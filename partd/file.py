import atexit
from contextlib import suppress
import os
import shutil
import string
import tempfile

from .core import Interface
from .keycodec import (INDEX_DIR, KeyCollisionError, KeyEncodingError,
                       indexed_relpath, key_to_relpath, relpath_needs_index)
import locket


#: Name of the marker file recording the storage layout version.
FORMAT_MARKER = '.partd-format'

#: Legacy layout: lossy filename cleaning, kept for pre-existing stores.
FORMAT_V1 = '1'

#: Versioned layout: typed, reversible key encoding (see partd.keycodec).
FORMAT_V2 = '2'

LOCK_NAME = '.lock'

#: Root entries that are never user data.
_RESERVED = frozenset([FORMAT_MARKER, LOCK_NAME])


class MigrationConflictError(RuntimeError):
    """A legacy store cannot be migrated without ambiguity or data loss."""


class File(Interface):
    """File-based storage with a versioned on-disk layout.

    New and empty directories use format version 2: keys are encoded with
    a typed, reversible encoding (see ``partd.keycodec``) so that distinct
    keys never collide, and a ``.partd-format`` marker file records the
    layout version.  Pre-existing directories without a marker are treated
    as legacy format 1 stores: keys keep the old lossy filename mapping so
    existing data stays readable and nothing is silently reinterpreted.
    Call ``migrate()`` to convert a legacy store explicitly.
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
        self.format_version = self._detect_format()
        self.lock = locket.lock_file(os.path.join(self.path, LOCK_NAME))
        Interface.__init__(self)

    def _detect_format(self):
        marker = os.path.join(self.path, FORMAT_MARKER)
        if os.path.exists(marker):
            with open(marker) as f:
                version = f.read().strip()
            if version not in (FORMAT_V1, FORMAT_V2):
                raise ValueError(
                    'unsupported partd format version %r in %s'
                    % (version, marker))
            return version
        for entry in os.listdir(self.path):
            if entry not in _RESERVED:
                # Existing data and no marker: legacy layout.  Keep
                # reading it with the legacy mapping; do not reinterpret.
                return FORMAT_V1
        self._write_marker(FORMAT_V2)
        return FORMAT_V2

    def _write_marker(self, version):
        with open(os.path.join(self.path, FORMAT_MARKER), 'w') as f:
            f.write(version + '\n')

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
                self._register_index(k)
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
            self._register_index(key)
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
                if self.format_version == FORMAT_V2:
                    mapfile = path + '.key'
                    if os.path.exists(mapfile):
                        os.remove(mapfile)
        finally:
            if lock:
                self.lock.release()

    def drop(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        self._iset_seen.clear()
        os.mkdir(self.path)
        if self.format_version == FORMAT_V2:
            self._write_marker(FORMAT_V2)

    def filename(self, key):
        if self.format_version == FORMAT_V1:
            return filename(self.path, key)
        rel = key_to_relpath(key)
        if relpath_needs_index(rel):
            rel = indexed_relpath(rel)
        full = os.path.join(self.path, rel)
        root = os.path.abspath(self.path)
        if os.path.commonpath((root, os.path.abspath(full))) != root:
            raise KeyEncodingError(
                'key %r escapes the store root' % (key,))
        return full

    def _register_index(self, key):
        """Record the canonical key for an index-mapped (overlong) key.

        The mapping lets us detect hash collisions explicitly instead of
        silently merging two distinct overlong keys.
        """
        if self.format_version != FORMAT_V2:
            return
        canonical = key_to_relpath(key)
        if not relpath_needs_index(canonical):
            return
        mapfile = self.filename(key) + '.key'
        if os.path.exists(mapfile):
            with open(mapfile) as f:
                existing = f.read()
            if existing != canonical:
                raise KeyCollisionError(
                    'keys %r and %r share index entry %s'
                    % (existing, canonical, mapfile))
            return
        with open(mapfile, 'w') as f:
            f.write(canonical)

    def migrate(self):
        """Migrate a legacy (format 1) store in place to format 2.

        The full plan is computed and checked for conflicts before any
        file is written; on conflict ``MigrationConflictError`` is raised
        and the store is left untouched.  Keys are inferred from legacy
        filenames: path separators become tuple segments and every segment
        is assumed to be a string.  Because the legacy encoding is lossy,
        keys that were collapsed by character stripping cannot be
        recovered; files whose names cannot be reproduced from the
        inferred key, or whose new location is already taken, are reported
        as conflicts instead of being silently reinterpreted.
        """
        if self.format_version != FORMAT_V1:
            return
        self.lock.acquire()
        try:
            plan = []
            for dirpath, dirnames, filenames in os.walk(self.path):
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, self.path)
                    if rel in _RESERVED:
                        continue
                    key = tuple(rel.split(os.sep))
                    if len(key) == 1:
                        key = key[0]
                    if filename(self.path, key) != full:
                        raise MigrationConflictError(
                            'legacy file %r cannot be attributed to a '
                            'unique key; refusing to reinterpret it' % rel)
                    plan.append((rel, key))
            targets = {}
            for rel, key in plan:
                target_rel = key_to_relpath(key)
                if relpath_needs_index(target_rel):
                    target_rel = indexed_relpath(target_rel)
                previous = targets.setdefault(target_rel, rel)
                if previous != rel:
                    raise MigrationConflictError(
                        'legacy files %r and %r map to the same new '
                        'location %r' % (previous, rel, target_rel))
                target_full = os.path.join(self.path, target_rel)
                source_full = os.path.join(self.path, rel)
                if (os.path.exists(target_full) and os.path.abspath(
                        target_full) != os.path.abspath(source_full)):
                    raise MigrationConflictError(
                        'migrating key %r would overwrite existing file '
                        '%r' % (key, target_rel))
            self.format_version = FORMAT_V2
            try:
                for rel, key in plan:
                    source = os.path.join(self.path, rel)
                    target = self.filename(key)
                    if os.path.abspath(source) == os.path.abspath(target):
                        continue
                    dirname = os.path.dirname(target)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    shutil.move(source, target)
                    self._register_index(key)
                self._remove_empty_dirs()
                self._write_marker(FORMAT_V2)
            except Exception:
                self.format_version = FORMAT_V1
                raise
        finally:
            self.lock.release()

    def _remove_empty_dirs(self):
        for dirpath, dirnames, filenames in os.walk(self.path,
                                                    topdown=False):
            if dirpath != self.path and not os.listdir(dirpath):
                os.rmdir(dirpath)

    def __exit__(self, *args):
        self._iset_seen.clear()
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def __del__(self):
        if not self._explicitly_given_path:
            self.__exit__()


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
