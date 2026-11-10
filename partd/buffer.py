from .core import Interface
from threading import Lock
from toolz import topk, accumulate, pluck
from operator import add
from bisect import bisect
from collections import defaultdict


def zero():
    return 0


class BufferErrorState(Exception):
    """Raised when an operation is attempted on a poisoned Buffer.

    A Buffer enters this state after a backing ``fast``/``slow`` append has
    failed.  The failure may have been only partially applied (for example a
    File backend can have written some bytes before raising), so silently
    retrying the same append/flush would duplicate records.  Callers must
    inspect the underlying stores and rebuild the Buffer instead.
    """


class Buffer(Interface):
    def __init__(self, fast, slow, available_memory=1e9):
        self.lock = Lock()
        self.fast = fast
        self.slow = slow
        self.available_memory = available_memory
        self.lengths = defaultdict(zero)
        self.memory_usage = 0
        self._error = None
        Interface.__init__(self)

    def __getstate__(self):
        return {'fast': self.fast,
                'slow': self.slow,
                'memory_usage': self.memory_usage,
                'lengths': self.lengths,
                'available_memory': self.available_memory,
                '_error': self._error}

    def __setstate__(self, state):
        Interface.__setstate__(self, state)
        self.lock = Lock()
        self.__dict__.update(state)

    def _check_healthy(self):
        if self._error is not None:
            raise BufferErrorState(
                'Buffer is in an error state after %s; refusing to retry '
                'because part of the data may already have been written. '
                'Inspect the backing stores and rebuild the Buffer.'
                % self._error)

    def _poison(self, operation, exc):
        if self._error is None:
            self._error = operation
        return exc

    def append(self, data, lock=True, **kwargs):
        if lock: self.lock.acquire()
        try:
            self._check_healthy()
            for k, v in data.items():
                self.lengths[k] += len(v)
                self.memory_usage += len(v)
            try:
                self.fast.append(data, lock=False, **kwargs)
            except Exception as e:
                # The fast append either succeeded or failed outright; it is
                # not allowed to leave our bookkeeping ahead of reality.
                for k, v in data.items():
                    self.lengths[k] -= len(v)
                    if self.lengths[k] == 0:
                        del self.lengths[k]
                    self.memory_usage -= len(v)
                self._poison('fast.append', e)
                raise

            while self.memory_usage > self.available_memory:
                keys = keys_to_flush(self.lengths, 0.1, maxcount=20)
                self.flush(keys, lock=False)

        finally:
            if lock: self.lock.release()

    def _get(self, keys, lock=True, **kwargs):
        if lock: self.lock.acquire()
        try:
            # slow holds the older records, fast the newer ones: the global
            # append order is slow + fast, never the other way around.
            result = list(map(add, self.slow.get(keys, lock=False),
                                   self.fast.get(keys, lock=False)))
        finally:
            if lock: self.lock.release()
        return result

    def _iset(self, key, value, lock=True):
        """ Idempotent set """
        if lock: self.lock.acquire()
        try:
            self._check_healthy()
            self.fast.iset(key, value, lock=False)
        finally:
            if lock: self.lock.release()

    def _delete(self, keys, lock=True):
        if lock: self.lock.acquire()
        try:
            self._check_healthy()
            self.fast.delete(keys, lock=False)
            self.slow.delete(keys, lock=False)
        finally:
            if lock: self.lock.release()

    def drop(self):
        self._iset_seen.clear()
        self.fast.drop()
        self.slow.drop()

    def __exit__(self, *args):
        self.drop()

    def flush(self, keys=None, block=None, lock=True):
        """ Flush keys to disk

        Parameters
        ----------

        keys: list or None
            list of keys to flush
        block: bool (defaults to None)
            Whether or not to block until all writing is complete
        lock: bool
            Acquire the Buffer lock; pass False when already holding it (e.g.
            from :meth:`append`, which flushes automatically).

        If no keys are given then flush all keys
        """
        if lock:
            self.lock.acquire()
        try:
            self._check_healthy()
            if keys is None:
                keys = list(self.lengths)
            keys = [key for key in keys if key in self.lengths]
            if not keys:
                return
            payload = dict(zip(keys, self.fast.get(keys, lock=False)))

            try:
                self.slow.append(payload, lock=False)
            except Exception as e:
                # Some records may already be on the slow store; do not roll
                # back and do not allow a blind retry that would duplicate
                # them.  Bookkeeping is left describing the fast copies which
                # are still intact.
                self._poison('slow.append during flush', e)
                raise

            try:
                self.fast.delete(keys, lock=False)
            except Exception as e:
                # Data is now safely on the slow store.  Keep the Buffer
                # usable for reads (slow still holds the records) but mark the
                # error so nobody assumes the fast copies were removed.
                self._poison('fast.delete during flush', e)
                raise

            for key in keys:
                self.memory_usage -= self.lengths[key]
                del self.lengths[key]
        finally:
            if lock:
                self.lock.release()



def keys_to_flush(lengths, fraction=0.1, maxcount=100000):
    """ Which keys to remove

    >>> lengths = {'a': 20, 'b': 10, 'c': 15, 'd': 15,
    ...            'e': 10, 'f': 25, 'g': 5}
    >>> keys_to_flush(lengths, 0.5)
    ['f', 'a']
    """
    top = topk(max(len(lengths) // 2, 1),
               lengths.items(),
               key=1)
    total = sum(lengths.values())
    cutoff = min(maxcount, max(1,
                   bisect(list(accumulate(add, pluck(1, top))),
                          total * fraction)))
    result = [k for k, v in top[:cutoff]]
    assert result
    return result
