import atexit
from contextlib import suppress
import os
import re
import shutil
import string
import tempfile

from .core import Interface
from .key_encoding import (
    FORMAT_MARKER,
    FORMAT_VERSION,
    LEGACY_FORMAT_VERSION,
    INDEX_DIRNAME,
    MARKER_TEMPLATE,
    key_to_relpath,
)
import locket


class MigrationConflictError(RuntimeError):
    """ Raised when a v1 -> v2 migration would overwrite or lose data """


class File(Interface):
    """ On-disk mapping of keys to bytes

    Two storage layouts exist:

    - **v1 (legacy)**: keys are mangled with ``escape_filename``; lossy and
      kept only for reading/writing pre-existing directories.
    - **v2 (current)**: keys use the reversible typed encoding from
      ``partd.key_encoding``.  New directories are created as v2 and carry
      a ``.partd-format`` marker file.

    Existing directories without a marker are recognized as v1 and are
    never silently reinterpreted; use ``migrate_to_v2`` to convert them
    explicitly.
    """
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
        self.format_version = _detect_format(path)
        if self.format_version == FORMAT_VERSION:
            _write_marker(path)
        self._index_dir = os.path.join(path, INDEX_DIRNAME)
        self.lock = locket.lock_file(os.path.join(path, '.lock'))
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
                path = self.filename(key)
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
        if self.format_version == FORMAT_VERSION:
            _write_marker(self.path)

    def filename(self, key):
        if self.format_version == FORMAT_VERSION:
            return os.path.join(self.path, key_to_relpath(key,
                                                          self._index_dir))
        return filename(self.path, key)

    def __exit__(self, *args):
        self.drop()
        with suppress(OSError):
            os.remove(os.path.join(self.path, FORMAT_MARKER))
        os.rmdir(self.path)

    def __del__(self):
        if not self._explicitly_given_path:
            self.__exit__()


_MARKER_RE = re.compile(r'partd-file-format: (\d+)$')


def _detect_format(path):
    """ Determine the storage layout version of the directory at ``path``

    Returns ``FORMAT_VERSION`` (2) for marked or empty directories and
    ``LEGACY_FORMAT_VERSION`` (1) for pre-existing unmarked directories
    that already contain data.
    """
    marker = os.path.join(path, FORMAT_MARKER)
    if os.path.exists(marker):
        with open(marker, encoding='utf-8', errors='replace') as f:
            match = _MARKER_RE.match(f.read().strip())
        if match:
            version = int(match.group(1))
            if version != FORMAT_VERSION:
                raise ValueError(
                    'unsupported File storage format version %d in %r'
                    % (version, path))
            return FORMAT_VERSION
        # A data file that merely shares the marker's name: treat the
        # directory as the legacy layout rather than misreading it.
        return LEGACY_FORMAT_VERSION
    if os.path.isdir(path):
        entries = [e for e in os.listdir(path) if e != '.lock']
        if entries:
            return LEGACY_FORMAT_VERSION
    return FORMAT_VERSION


def _write_marker(path):
    marker = os.path.join(path, FORMAT_MARKER)
    if not os.path.exists(marker):
        with open(marker, 'w', encoding='utf-8') as f:
            f.write(MARKER_TEMPLATE % FORMAT_VERSION)


def migrate_to_v2(path):
    """ Migrate a legacy (v1) File directory to the v2 layout

    Each existing file is re-stored under the v2 encoding of its key.  A
    single-segment legacy name becomes a ``str`` key; a nested legacy path
    becomes a tuple of its segments (mirroring the v1 ``token`` scheme).

    The whole plan is validated before anything is written: if two legacy
    files would map to the same v2 name, or a v2 name collides with
    something that is not being migrated, ``MigrationConflictError`` is
    raised and the directory is left untouched.

    The directory must not be open as a ``File`` while migrating.
    """
    if _detect_format(path) != LEGACY_FORMAT_VERSION:
        raise ValueError('%r is not a legacy (v1) File directory' % path)
    lock = locket.lock_file(os.path.join(path, '.lock'))
    lock.acquire()
    try:
        if _detect_format(path) != LEGACY_FORMAT_VERSION:
            raise ValueError('%r is not a legacy (v1) File directory' % path)

        sources = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d != INDEX_DIRNAME]
            for name in filenames:
                if name in ('.lock', FORMAT_MARKER):
                    continue
                sources.append(os.path.relpath(os.path.join(dirpath, name),
                                               path))

        source_set = set(sources)
        plan = []
        targets = {}
        conflicts = []
        for rel in sorted(sources):
            segments = rel.split(os.sep)
            key = segments[0] if len(segments) == 1 else tuple(segments)
            dst = key_to_relpath(key)
            if dst in targets:
                conflicts.append('%r and %r both map to %r'
                                 % (targets[dst], rel, dst))
            elif (os.path.lexists(os.path.join(path, dst))
                    and dst not in source_set):
                conflicts.append('%r would overwrite existing path %r'
                                 % (rel, dst))
            targets[dst] = rel
            plan.append((rel, dst))
        if conflicts:
            raise MigrationConflictError(
                'refusing to migrate %r:\n' % path
                + '\n'.join(conflicts))

        # Two-phase move through temporary names so that targets which
        # themselves are still-unmigrated sources cannot be clobbered.
        temps = []
        for i, (rel, dst) in enumerate(plan):
            tmp = os.path.join(path, '.partd-migrating-%d' % i)
            os.rename(os.path.join(path, rel), tmp)
            temps.append((tmp, dst))
        for tmp, dst in temps:
            final = os.path.join(path, dst)
            dirname = os.path.dirname(final)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            os.rename(tmp, final)
        for rel, dst in plan:
            dirname = os.path.dirname(os.path.join(path, rel))
            if dirname:
                with suppress(OSError):
                    os.removedirs(dirname)
        _write_marker(path)
    finally:
        lock.release()


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
