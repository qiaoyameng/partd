from .core import Interface
from threading import Lock


class Dict(Interface):
    def __init__(self):
        self.lock = Lock()
        self.data = dict()
        Interface.__init__(self)

    def __getstate__(self):
        return {'data': self.data}

    def __setstate__(self, state):
        Interface.__setstate__(self, state)
        Dict.__init__(self)
        self.data = state['data']

    def append(self, data, lock=True, **kwargs):
        if lock: self.lock.acquire()
        try:
            for k, v in data.items():
                if k not in self.data:
                    self.data[k] = []
                self.data[k].append(v)
        finally:
            if lock: self.lock.release()

    def _get(self, keys, lock=True, **kwargs):
        assert isinstance(keys, (list, tuple, set))
        if lock:
            self.lock.acquire()
        try:
            result = [b''.join(self.data.get(key, [])) for key in keys]
        finally:
            if lock:
                self.lock.release()
        return result

    def _iter_chunks(self, key, chunksize=1024 * 1024, lock=True):
        return _DictChunks(self, key, chunksize, lock)

    def _iset(self, key, value, lock=True):
        """ Idempotent set """
        if lock:
            self.lock.acquire()
        try:
            self.data[key] = [value]
        finally:
            if lock:
                self.lock.release()

    def _delete(self, keys, lock=True):
        if lock:
            self.lock.acquire()
        try:
            for key in keys:
                if key in self.data:
                    del self.data[key]
        finally:
            if lock:
                self.lock.release()

    def drop(self):
        self._iset_seen.clear()
        self.data.clear()

    def __exit__(self, *args):
        self.drop()


class _DictChunks:
    __slots__ = ('_chunks', '_current', '_offset', '_chunksize', '_closed')

    def __init__(self, partd, key, chunksize, lock):
        self._current = b''
        self._offset = 0
        self._chunksize = chunksize
        self._closed = False

        if lock:
            partd.lock.acquire()
        try:
            self._chunks = iter(tuple(partd.data.get(key, ())))
        finally:
            if lock:
                partd.lock.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration

        while self._offset >= len(self._current):
            try:
                self._current = next(self._chunks)
            except StopIteration:
                self.close()
                raise
            self._offset = 0

        end = self._offset + self._chunksize
        chunk = self._current[self._offset:end]
        self._offset = end
        return chunk

    def close(self):
        self._chunks = iter(())
        self._current = b''
        self._offset = 0
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
