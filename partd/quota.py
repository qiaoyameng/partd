""" Byte quotas for local shuffle storage.

This module wraps any partd backend with a quota enforcement layer.  Two
independent limits are tracked, both measured in logical payload bytes
(``len(value)``) rather than bytes on disk:

* a per-partition quota: the stored size of any single key
* a global quota: the size of the whole storage directory

Enforcement semantics
---------------------

* An ``append`` batch is inspected in full and checked against *both* quotas
  before the wrapped backend is touched.  If the batch would exceed either
  quota, ``QuotaExceeded`` is raised and no key in the batch is written.
* The ledger is updated only after the wrapped operation reports success.
  If appending, overwriting or deleting raises, the ledger is left untouched
  so a failure never leaves phantom free space.
* The layer is transparent to movement between storage tiers: when a
  ``Buffer`` flushes its fast store into its slow store it calls the
  underlying backend directly, so a flush neither charges the bytes again
  nor releases the total.
* Quotas are charges for *stored* bytes, so an overwrite (``iset``) of an
  existing key first frees the old size before reserving the new one.
* Deleting a missing key is a no-op that does not change the ledger.

Persistence
-----------

When the wrapped backend (possibly nested) has a ``path`` (i.e. it stores to
a directory), the size ledger is persisted next to the data in
``.quota.json``.  On reopen a ``Quota`` recovers its counters from that file;
if the metadata is missing or corrupt the ledger is rebuilt by walking the
storage directory and summing file contents.  Backends without a directory
(for example a bare :class:`~partd.dict.Dict`) keep the ledger in memory
only.
"""

import json
import os
from contextlib import suppress
from threading import Lock

from .core import Interface, token

META_FILENAME = '.quota.json'
LOCK_FILENAME = '.lock'


class QuotaExceeded(Exception):
    """ Raised when an append or overwrite would exceed a quota. """


def _storage_path(partd):
    """ Find the on-disk directory of a (possibly nested) backend.

    Backends compose through ``partd`` (e.g. :class:`~partd.encode.Encode`)
    or through ``fast``/``slow`` (e.g. :class:`~partd.buffer.Buffer`).
    """
    seen = set()
    pending = [partd]
    while pending:
        backend = pending.pop()
        if id(backend) in seen:
            continue
        seen.add(id(backend))
        path = getattr(backend, 'path', None)
        if isinstance(path, str):
            return path
        pending.extend(child for child in (getattr(backend, 'partd', None),
                                           getattr(backend, 'slow', None),
                                           getattr(backend, 'fast', None))
                       if child is not None)
    return None


class Quota(Interface):
    """ Restrict stored bytes per partition and for the whole store.

    Parameters
    ----------
    partd:
        Backend being wrapped, e.g. ``Buffer(Dict(), File(path))`` or a
        plain ``File(path)``.
    per_partition:
        Maximum logical bytes stored under a single key.
    global_bytes:
        Maximum logical bytes stored in the whole storage directory.
    """

    def __init__(self, partd, per_partition=None, global_bytes=None):
        if per_partition is not None and per_partition < 0:
            raise ValueError("per_partition quota must be non-negative")
        if global_bytes is not None and global_bytes < 0:
            raise ValueError("global_bytes quota must be non-negative")
        self.partd = partd
        self.per_partition = per_partition
        self.global_bytes = global_bytes
        self.path = _storage_path(partd)
        self.lock = Lock()
        self.lengths = self._load_lengths()
        self.total = sum(self.lengths.values())
        Interface.__init__(self)

    def __getstate__(self):
        return {'partd': self.partd,
                'per_partition': self.per_partition,
                'global_bytes': self.global_bytes}

    def __setstate__(self, state):
        Interface.__setstate__(self, state)
        Quota.__init__(self, state['partd'],
                       state.get('per_partition'),
                       state.get('global_bytes'))

    # ------------------------------------------------------------------
    # ledger persistence
    # ------------------------------------------------------------------

    def _meta_filename(self):
        return os.path.join(self.path, META_FILENAME)

    def _load_lengths(self):
        if self.path is None or not os.path.exists(self.path):
            return {}
        try:
            with open(self._meta_filename()) as f:
                payload = json.load(f)
            lengths = payload['lengths']
            if not isinstance(lengths, dict) or                     not all(isinstance(k, str) and isinstance(v, int) and
                            v >= 0 for k, v in lengths.items()):
                raise ValueError("invalid quota metadata")
            if payload.get('total') != sum(lengths.values()):
                raise ValueError("inconsistent quota metadata")
            return lengths
        except (OSError, ValueError, KeyError, TypeError):
            return self._scan_lengths()

    def _scan_lengths(self):
        """ Rebuild the ledger by reading every data file in the store. """
        lengths = {}
        if not self.path or not os.path.exists(self.path):
            return lengths
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if filename == LOCK_FILENAME or \
                        filename == META_FILENAME or \
                        filename.startswith(META_FILENAME + '.'):
                    continue
                full = os.path.join(dirpath, filename)
                rel = os.path.relpath(full, self.path)
                key = token(rel)
                with suppress(OSError):
                    with open(full, 'rb') as f:
                        lengths[key] = lengths.get(key, 0) + \
                            len(f.read())
        return lengths

    def _write_meta(self):
        if self.path is None:
            return
        payload = {'total': self.total,
                   'lengths': dict(self.lengths)}
        tmp = self._meta_filename() + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, self._meta_filename())

    # ------------------------------------------------------------------
    # quota checks
    # ------------------------------------------------------------------

    def _check(self, sizes):
        """ Validate a mapping of projected per-key stored sizes. """
        if self.per_partition is not None:
            for key, size in sizes.items():
                if size > self.per_partition:
                    raise QuotaExceeded(
                        "partition %r would use %d bytes, exceeding the "
                        "per-partition quota of %d"
                        % (key, size, self.per_partition))
        if self.global_bytes is not None and \
                sum(sizes.values()) > self.global_bytes:
            raise QuotaExceeded(
                "store would use %d bytes, exceeding the global quota "
                "of %d" % (sum(sizes.values()), self.global_bytes))

    def _reserve(self, additions):
        """ New per-key and total sizes after an append of ``additions``. """
        sizes = dict(self.lengths)
        for key, size in additions.items():
            sizes[key] = sizes.get(key, 0) + size
        return sizes

    # ------------------------------------------------------------------
    # partd Interface
    # ------------------------------------------------------------------

    def append(self, data, lock=True, **kwargs):
        if lock:
            self.lock.acquire()
        try:
            sizes = self._reserve(
                {token(key): len(value) for key, value in data.items()})
            self._check(sizes)
            self.partd.append(data, lock=False, **kwargs)
            self.lengths = sizes
            self.total = sum(sizes.values())
            self._write_meta()
        finally:
            if lock:
                self.lock.release()

    def _get(self, keys, lock=True, **kwargs):
        return self.partd._get(keys, lock=lock, **kwargs)

    def _iset(self, key, value, lock=True):
        if lock:
            self.lock.acquire()
        try:
            token_key = token(key)
            new_size = len(value)
            sizes = dict(self.lengths)
            sizes[token_key] = new_size
            self._check(sizes)
            self.partd.iset(key, value, lock=False)
            self.lengths = sizes
            self.total = sum(sizes.values())
            self._write_meta()
        finally:
            if lock:
                self.lock.release()

    def _delete(self, keys, lock=True):
        if lock:
            self.lock.acquire()
        try:
            existing = dict(self.lengths)
            for key in keys:
                existing.pop(token(key), None)
            self.partd.delete(keys, lock=False)
            self.lengths = existing
            self.total = sum(existing.values())
            self._write_meta()
        finally:
            if lock:
                self.lock.release()

    def used(self, key=None):
        """ Logical bytes currently stored for ``key`` (or the total). """
        if key is None:
            return self.total
        return self.lengths.get(token(key), 0)

    def drop(self):
        self._iset_seen.clear()
        self.partd.drop()
        self.lengths = {}
        self.total = 0
        if self.path is not None:
            with suppress(OSError):
                os.remove(self._meta_filename())

    def flush(self, keys=None, block=None):
        if hasattr(self.partd, 'flush'):
            self.partd.flush(keys, block)

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(*args)
