import json
import os

from .core import Interface, token


METADATA_FILENAME = '.quota-usage.json'


class QuotaExceeded(Exception):
    """ Raised when a write would exceed a storage quota """


def _storage_path(partd, seen=None):
    """ Find the on-disk directory of a possibly wrapped partd, if any """
    if seen is None:
        seen = set()
    if id(partd) in seen:
        return None
    seen.add(id(partd))
    path = getattr(partd, 'path', None)
    if path is not None:
        return path
    for attr in ('partd', 'slow', 'fast'):
        child = getattr(partd, attr, None)
        if child is not None:
            path = _storage_path(child, seen)
            if path is not None:
                return path
    return None


class Quota(Interface):
    """ Quota-limiting wrapper around another partd

    Parameters
    ----------
    partd: Interface
        The partd to wrap
    partition_quota: int or None
        Maximum logical bytes stored per key.  None means unlimited.
    total_quota: int or None
        Maximum logical bytes stored across all keys.  None means unlimited.

    Quotas are counted in logical bytes: the lengths of the values passed to
    ``append`` and ``iset``, regardless of how the wrapped partd encodes or
    moves them (e.g. a Buffer flushing from fast to slow storage changes no
    accounting).

    Batches are validated in full before anything is written.  If any key in
    the batch would exceed its partition quota, or the batch as a whole would
    exceed the total quota, the entire batch is rejected with
    ``QuotaExceeded`` and nothing is written.

    Accounting is updated only after the wrapped partd succeeds, so failed
    deletes or writes never free up phantom space.  When wrapping an on-disk
    partd such as File, usage is persisted to a metadata file and, on reopen,
    rebuilt from that metadata or, failing that, from the stored content.
    """
    def __init__(self, partd, partition_quota=None, total_quota=None):
        self.partd = partd
        self.partition_quota = partition_quota
        self.total_quota = total_quota
        self.path = _storage_path(partd)
        self.usage = self._rebuild_usage()
        Interface.__init__(self)

    def __getstate__(self):
        return {'partd': self.partd,
                'partition_quota': self.partition_quota,
                'total_quota': self.total_quota,
                'path': self.path,
                'usage': self.usage}

    __setstate__ = Interface.__setstate__

    @property
    def lock(self):
        return self.partd.lock

    @property
    def total_usage(self):
        return sum(self.usage.values())

    def _metadata_filename(self):
        if self.path is None:
            return None
        return os.path.join(self.path, METADATA_FILENAME)

    def _rebuild_usage(self):
        if self.path is None or not os.path.isdir(self.path):
            return {}
        meta = self._metadata_filename()
        if os.path.exists(meta):
            with open(meta) as f:
                return {k: int(v) for k, v in json.load(f).items()}
        usage = {}
        for dirpath, dirnames, filenames in os.walk(self.path):
            for fn in filenames:
                if fn in ('.lock', METADATA_FILENAME):
                    continue
                full = os.path.join(dirpath, fn)
                usage[os.path.relpath(full, self.path)] = os.path.getsize(full)
        return usage

    def _persist_usage(self):
        meta = self._metadata_filename()
        if meta is None or not os.path.isdir(self.path):
            return
        with open(meta, 'w') as f:
            json.dump(self.usage, f)

    def _check_append(self, additions):
        for k, n in additions.items():
            if (self.partition_quota is not None and
                    self.usage.get(k, 0) + n > self.partition_quota):
                raise QuotaExceeded(
                    "Key %r would exceed partition quota: %d + %d > %d" %
                    (k, self.usage.get(k, 0), n, self.partition_quota))
        if (self.total_quota is not None and
                self.total_usage + sum(additions.values()) > self.total_quota):
            raise QuotaExceeded(
                "Batch would exceed total quota: %d + %d > %d" %
                (self.total_usage, sum(additions.values()), self.total_quota))

    def append(self, data, lock=True, **kwargs):
        if lock:
            self.lock.acquire()
        try:
            additions = {token(k): len(v) for k, v in data.items()}
            self._check_append(additions)
            self.partd.append(data, lock=False, **kwargs)
            for k, n in additions.items():
                self.usage[k] = self.usage.get(k, 0) + n
            self._persist_usage()
        finally:
            if lock:
                self.lock.release()

    def _get(self, keys, **kwargs):
        return self.partd._get(keys, **kwargs)

    def _iset(self, key, value, lock=True):
        if lock:
            self.lock.acquire()
        try:
            t = token(key)
            if (self.partition_quota is not None and
                    len(value) > self.partition_quota):
                raise QuotaExceeded(
                    "Key %r would exceed partition quota: %d > %d" %
                    (t, len(value), self.partition_quota))
            delta = len(value) - self.usage.get(t, 0)
            if (self.total_quota is not None and
                    self.total_usage + delta > self.total_quota):
                raise QuotaExceeded(
                    "Set of %r would exceed total quota: %d + %d > %d" %
                    (t, self.total_usage, delta, self.total_quota))
            self.partd._iset(key, value, lock=False)
            self.usage[t] = len(value)
            self._persist_usage()
        finally:
            if lock:
                self.lock.release()

    def _delete(self, keys, lock=True):
        if lock:
            self.lock.acquire()
        try:
            self.partd.delete(keys, lock=False)
            for key in keys:
                self.usage.pop(token(key), None)
            self._persist_usage()
        finally:
            if lock:
                self.lock.release()

    def flush(self, *args, **kwargs):
        return self.partd.flush(*args, **kwargs)

    def drop(self):
        self.partd.drop()
        self.usage.clear()
        self._iset_seen.clear()
        self._persist_usage()

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(*args)
