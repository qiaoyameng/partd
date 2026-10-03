import atexit
from contextlib import suppress
import json
import os
import shutil
import string
import tempfile

from .core import Interface
import locket


TXN_JOURNAL_NAME = '.txn-journal'
TXN_COMMIT_MARKER = 'COMMITTED'


class File(Interface):
    def __init__(self, path=None, dir=None):
        if not path:
            path = tempfile.mkdtemp(suffix='.partd', dir=dir)
            cleanup_files.append(path)
            self._explicitly_given_path = False
        else:
            self._explicitly_given_path = True
        self.path = path
        if not os.path.exists(path):
            with suppress(OSError):
                os.makedirs(path)
        self.lock = locket.lock_file(self.filename('.lock'))
        self._recover_transaction()
        Interface.__init__(self)

    def __getstate__(self):
        return {'path': self.path}

    def __setstate__(self, state):
        Interface.__setstate__(self, state)
        File.__init__(self, state['path'])

    def append(self, data, lock=True, fsync=False, **kwargs):
        if lock: self.lock.acquire()
        try:
            for k, v in data.items():
                fn = self.filename(k)
                if not os.path.exists(os.path.dirname(fn)):
                    os.makedirs(os.path.dirname(fn))
                with open(fn, 'ab') as f:
                    f.write(v)
                    if fsync:
                        os.fsync(f)
        finally:
            if lock: self.lock.release()

    def append_transactional(self, data, fsync=True, lock=True):
        """ Atomically append a batch of ``{key: bytes}`` to many partitions

        Opt-in variant of ``append`` for shuffle-style batches that touch
        several partition files at once.  Either the whole batch becomes
        visible or, if the process dies midway, none of it does; the next
        ``File`` opened on the same directory rolls back the partial batch.

        Persistence order (each step durable before the next starts):

        1. A journal file recording every target file's original length is
           written, flushed and fsynced.
        2. Each partition's payload is appended, flushed and (if ``fsync``)
           fsynced.
        3. A commit marker is appended to the journal, flushed and fsynced.
        4. The journal is removed.

        On open, a journal without a commit marker truncates each target
        back to its recorded length (files first created by the batch are
        removed); a committed journal is simply discarded.
        """
        if lock:
            self.lock.acquire()
        try:
            entries = []
            for k in data:
                fn = self.filename(k)
                existed = os.path.exists(fn)
                entries.append((os.path.relpath(fn, self.path), existed,
                                os.path.getsize(fn) if existed else 0))
            self._write_txn_journal(entries)
            for (relpath, _, _), v in zip(entries, data.values()):
                self._append_txn_file(os.path.join(self.path, relpath), v,
                                      fsync)
            self._commit_txn_journal()
            self._remove_txn_journal()
        finally:
            if lock:
                self.lock.release()

    def _txn_journal_path(self):
        return os.path.join(self.path, TXN_JOURNAL_NAME)

    def _write_txn_journal(self, entries):
        with open(self._txn_journal_path(), 'w') as f:
            f.write(json.dumps({'entries': [list(e) for e in entries]}))
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        self._fsync_dir()

    def _append_txn_file(self, fn, value, fsync):
        if not os.path.exists(os.path.dirname(fn)):
            os.makedirs(os.path.dirname(fn))
        with open(fn, 'ab') as f:
            f.write(value)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

    def _commit_txn_journal(self):
        with open(self._txn_journal_path(), 'a') as f:
            f.write(TXN_COMMIT_MARKER + '\n')
            f.flush()
            os.fsync(f.fileno())

    def _remove_txn_journal(self):
        os.remove(self._txn_journal_path())
        self._fsync_dir()

    def _fsync_dir(self):
        with suppress(OSError):
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _recover_transaction(self):
        journal = self._txn_journal_path()
        if not os.path.exists(journal):
            return
        self.lock.acquire()
        try:
            if not os.path.exists(journal):
                return
            with open(journal, 'r') as f:
                lines = f.read().splitlines()
            committed = bool(lines) and lines[-1] == TXN_COMMIT_MARKER
            if not committed:
                # A journal that cannot be parsed was never fully fsynced,
                # so no partition data was appended yet; discard it.
                entries = []
                if lines:
                    with suppress(ValueError, KeyError):
                        entries = json.loads(lines[0])['entries']
                for entry in entries:
                    self._rollback_txn_entry(entry)
            os.remove(journal)
            self._fsync_dir()
        finally:
            self.lock.release()

    def _rollback_txn_entry(self, entry):
        relpath, existed, length = entry
        fn = os.path.join(self.path, relpath)
        if existed:
            with suppress(OSError):
                with open(fn, 'r+b') as f:
                    f.truncate(length)
                    f.flush()
                    os.fsync(f.fileno())
        else:
            with suppress(OSError):
                os.remove(fn)

    def _get(self, keys, lock=True, **kwargs):
        assert isinstance(keys, (list, tuple, set))
        if lock:
            self.lock.acquire()
        try:
            result = []
            for key in keys:
                try:
                    with open(self.filename(key), 'rb') as f:
                        result.append(f.read())
                except OSError:
                    result.append(b'')
        finally:
            if lock:
                self.lock.release()
        return result

    def _iset(self, key, value, lock=True):
        """ Idempotent set """
        fn = self.filename(key)
        if not os.path.exists(os.path.dirname(fn)):
            os.makedirs(os.path.dirname(fn))
        if lock:
            self.lock.acquire()
        try:
            with open(self.filename(key), 'wb') as f:
                f.write(value)
        finally:
            if lock:
                self.lock.release()

    def _delete(self, keys, lock=True):
        if lock:
            self.lock.acquire()
        try:
            for key in keys:
                path = filename(self.path, key)
                if os.path.exists(path):
                    os.remove(path)
        finally:
            if lock:
                self.lock.release()

    def drop(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        self._iset_seen.clear()
        os.mkdir(self.path)

    def filename(self, key):
        return filename(self.path, key)

    def __exit__(self, *args):
        self.drop()
        os.rmdir(self.path)

    def __del__(self):
        if not self._explicitly_given_path:
            self.drop()
            os.rmdir(self.path)


def filename(path, key):
    return os.path.join(path, escape_filename(token(key)))


# http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename-in-python
valid_chars = "-_.() " + string.ascii_letters + string.digits + os.path.sep


def escape_filename(fn):
    """ Escape text so that it is a valid filename

    >>> escape_filename('Foo!bar?')
    'Foobar'

    """
    return ''.join(filter(valid_chars.__contains__, fn))



def token(key):
    """

    >>> token('hello')
    'hello'
    >>> token(('hello', 'world'))  # doctest: +SKIP
    'hello/world'
    """
    if isinstance(key, str):
        return key
    elif isinstance(key, tuple):
        return os.path.join(*map(token, key))
    else:
        return str(key)


cleanup_files = list()

@atexit.register
def cleanup():
    for fn in cleanup_files:
        if os.path.exists(fn):
            shutil.rmtree(fn)
