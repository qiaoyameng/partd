import atexit
import json
import os
import shutil
import string
import tempfile
import uuid

from .core import Interface
import locket


class File(Interface):
    def __init__(self, path=None, dir=None):
        if not path:
            path = tempfile.mkdtemp(suffix='.partd', dir=dir)
            cleanup_files.append(path)
            self._explicitly_given_path = False
        else:
            self._explicitly_given_path = True
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.lock = locket.lock_file(self.filename('.lock'))
        with self.lock:
            self._recover_transactions(lock=False)
        Interface.__init__(self)

    def __getstate__(self):
        return {'path': self.path}

    def __setstate__(self, state):
        Interface.__setstate__(self, state)
        File.__init__(self, state['path'])

    def append(self, data, lock=True, fsync=False, transaction=False, **kwargs):
        if transaction:
            return self._append_transaction(data, lock=lock)
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

    def _append_transaction(self, data, lock=True):
        if not data:
            return

        if lock:
            self.lock.acquire()
        try:
            entries = self._transaction_entries(data)
            transaction_id = uuid.uuid4().hex
            temporary = self.transaction_path(transaction_id, 'tmp')
            pending = self.transaction_path(transaction_id, 'pending')
            committed = self.transaction_path(transaction_id, 'log')
            payload = json.dumps({
                'version': 1,
                'files': entries,
            }, sort_keys=True).encode('utf-8')

            journal = None
            try:
                journal = self._transaction_step('journal.tmp.open', open,
                                                 temporary, 'xb')
                self._transaction_step('journal.tmp.write', journal.write, payload)
                self._transaction_step('journal.tmp.flush', journal.flush)
                self._transaction_step('journal.tmp.fsync', os.fsync,
                                       journal.fileno())
            finally:
                if journal is not None:
                    self._transaction_step('journal.tmp.close', journal.close)
            self._fsync_directory(self.path, 'journal.tmp.directory')

            self._transaction_step('journal.prepare.replace', os.replace,
                                   temporary, pending)
            self._fsync_directory(self.path, 'journal.prepare.directory')

            for entry in entries:
                self._append_transaction_entry(entry, data)

            self._transaction_step('journal.commit.replace', os.replace,
                                   pending, committed)
            self._fsync_directory(self.path, 'journal.commit.directory')

            self._transaction_step('journal.cleanup.remove', os.remove, committed)
            self._fsync_directory(self.path, 'journal.cleanup.directory')
        finally:
            if lock:
                self.lock.release()

    def _transaction_entries(self, data):
        entries = []
        seen = set()
        for key, value in sorted(data.items(),
                                 key=lambda item: self.filename(item[0])):
            if not isinstance(value, bytes):
                raise TypeError('transactional File appends require bytes values')

            target = self.filename(key)
            relative = os.path.relpath(target, self.path)
            if self._transaction_path_unsafe(relative):
                raise ValueError('unsafe transactional File key')
            if relative in seen:
                raise ValueError('duplicate transactional File target')
            seen.add(relative)

            parent = os.path.dirname(target)
            if parent != self.path and not os.path.isdir(parent):
                self._transaction_step(
                    'directory.create:' + os.path.relpath(parent, self.path),
                    os.makedirs, parent, 0o777, True)
                self._sync_directory_chain(parent)

            existed = os.path.exists(target)
            entries.append({
                'path': relative,
                'length': os.path.getsize(target) if existed else 0,
                'existed': existed,
            })
        return entries

    def _append_transaction_entry(self, entry, data):
        target = os.path.join(self.path, entry['path'])
        key = entry['path']
        output = None
        try:
            output = self._transaction_step(f'file.open:{key}', open,
                                            target, 'ab')
            self._transaction_step(f'file.write:{key}', output.write,
                                   self._transaction_value(data, entry['path']))
            self._transaction_step(f'file.flush:{key}', output.flush)
            self._transaction_step(f'file.fsync:{key}', os.fsync,
                                   output.fileno())
        finally:
            if output is not None:
                self._transaction_step(f'file.close:{key}', output.close)
        if not entry['existed']:
            self._fsync_directory(os.path.dirname(target),
                                  f'file.directory:{key}')

    def _transaction_value(self, data, relative):
        for key, value in data.items():
            if os.path.relpath(self.filename(key), self.path) == relative:
                return value
        raise KeyError(relative)

    def _sync_directory_chain(self, directory):
        directories = []
        current = directory
        while current != self.path:
            directories.append(current)
            current = os.path.dirname(current)
        directories.append(self.path)
        for current in reversed(directories):
            if current == self.path:
                label = 'directory.sync'
            else:
                label = 'directory.sync:' + os.path.relpath(current, self.path)
            self._fsync_directory(current, label)

    def _fsync_directory(self, directory, prefix):
        if os.name == 'nt':
            return

        flags = os.O_RDONLY
        if hasattr(os, 'O_DIRECTORY'):
            flags |= os.O_DIRECTORY
        descriptor = None
        try:
            descriptor = self._transaction_step(prefix + '.open', os.open,
                                                 directory, flags)
            self._transaction_step(prefix + '.fsync', os.fsync, descriptor)
        finally:
            if descriptor is not None:
                self._transaction_step(prefix + '.close', os.close, descriptor)

    def transaction_path(self, transaction_id, state):
        return os.path.join(self.path, f'.~txn.{transaction_id}.{state}')

    def _recover_transactions(self, lock=True):
        if lock:
            self.lock.acquire()
        try:
            names = sorted(os.listdir(self.path))
            for name in names:
                path = os.path.join(self.path, name)
                if os.path.isfile(path) and name.startswith('.~txn.') \
                        and name.endswith('.tmp'):
                    self._transaction_step('recovery.tmp.remove', os.remove, path)
                    self._fsync_directory(self.path, 'recovery.tmp.directory')

            for name in names:
                path = os.path.join(self.path, name)
                if not os.path.isfile(path) or not name.startswith('.~txn.'):
                    continue
                if name.endswith('.pending'):
                    self._recover_pending_transaction(path)
                elif name.endswith('.log'):
                    self._recover_committed_transaction(path)
        finally:
            if lock:
                self.lock.release()

    def _read_transaction(self, path):
        with open(path, 'rb') as journal:
            document = json.loads(journal.read().decode('utf-8'))
        if document.get('version') != 1 or not isinstance(document.get('files'), list):
            raise ValueError('invalid partd transaction journal')
        for entry in document['files']:
            relative = entry.get('path')
            length = entry.get('length')
            existed = entry.get('existed')
            if not isinstance(relative, str) or not isinstance(length, int) \
                    or isinstance(length, bool) or length < 0 \
                    or not isinstance(existed, bool) \
                    or self._transaction_path_unsafe(relative):
                raise ValueError('invalid partd transaction journal')
        return document['files']

    def _recover_pending_transaction(self, path):
        for entry in self._read_transaction(path):
            self._rollback_transaction_entry(entry)

        name = os.path.basename(path)
        self._transaction_step('recovery.journal.remove', os.remove, path)
        self._fsync_directory(self.path,
                              'recovery.journal.directory:' + name)

    def _recover_committed_transaction(self, path):
        name = os.path.basename(path)
        self._transaction_step('recovery.committed.remove:' + name,
                               os.remove, path)
        self._fsync_directory(
            self.path, 'recovery.committed.directory:' + name)

    def _rollback_transaction_entry(self, entry):
        relative = entry['path']
        target = os.path.join(self.path, relative)
        if self._transaction_path_unsafe(relative):
            raise ValueError('invalid partd transaction journal')
        if not os.path.exists(target):
            return

        original_length = entry['length']
        if not entry['existed']:
            self._transaction_step(f'recovery.file.remove:{relative}',
                                   os.remove, target)
            self._fsync_directory(os.path.dirname(target),
                                  f'recovery.file.directory:{relative}')
        elif os.path.getsize(target) >= original_length:
            output = None
            try:
                output = self._transaction_step(
                    f'recovery.file.open:{relative}', open, target, 'r+b')
                self._transaction_step(
                    f'recovery.file.truncate:{relative}',
                    output.truncate, original_length)
                self._transaction_step(
                    f'recovery.file.flush:{relative}', output.flush)
                self._transaction_step(
                    f'recovery.file.fsync:{relative}', os.fsync,
                    output.fileno())
            finally:
                if output is not None:
                    self._transaction_step(
                        f'recovery.file.close:{relative}', output.close)

    def _transaction_step(self, step, function, *args):
        return function(*args)

    def _transaction_path_unsafe(self, relative):
        return os.path.isabs(relative) or relative == os.pardir or \
            relative.startswith(os.pardir + os.path.sep)

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
