from partd.file import (FORMAT_MARKER, FORMAT_V1, FORMAT_V2, File,
                        MigrationConflictError)
from partd.keycodec import INDEX_DIR, key_to_relpath

import hashlib
import pytest
import shutil
import os


def test_partd():
    with File() as p:
        p.append({'x': b'Hello', 'y': b'abc'})
        p.append({'x': b'World!', 'y': b'def'})
        assert os.path.exists(p.filename('x'))
        assert os.path.exists(p.filename('y'))

        result = p.get(['y', 'x'])
        assert result == [b'abcdef', b'HelloWorld!']

        assert p.get('z') == b''

        with p.lock:  # uh oh, possible deadlock
            result = p.get(['x'], lock=False)

    assert not os.path.exists(p.path)


def test_key_tuple():
    with File() as p:
        p.append({('a', 'b'): b'123'})
        assert os.path.exists(p.filename(('a', 'b')))


def test_iset():
    with File() as p:
        p.iset('x', b'123')
        assert 'x' in p._iset_seen
        assert 'y' not in p._iset_seen
        p.iset('x', b'123')
        p.iset('x', b'123')
        assert p.get('x') == b'123'


def test_nested_get():
    with File() as p:
        p.append({'x': b'1', 'y': b'2', 'z': b'3'})
        assert p.get(['x', ['y', 'z']]) == [b'1', [b'2', b'3']]


def test_drop():
    with File() as p:
        p.append({'x': b'123'})
        p.iset('y', b'abc')
        assert p.get('x') == b'123'
        assert p.get('y') == b'abc'

        p.drop()
        assert p.get('x') == b''
        assert p.get('y') == b''

        p.append({'x': b'123'})
        p.iset('y', b'def')
        assert p.get('x') == b'123'
        assert p.get('y') == b'def'


def test_del():
    f = File()

    assert f.path
    assert os.path.exists(f.path)

    f.__del__()
    assert not os.path.exists(f.path)

    with File('Foo') as p:
        p.__del__()
        assert os.path.exists(p.path)


def test_specify_dirname():
    with File(dir=os.getcwd()) as f:
        assert os.getcwd() in f.path


def test_new_store_is_format_v2():
    with File() as p:
        assert p.format_version == FORMAT_V2
        with open(os.path.join(p.path, FORMAT_MARKER)) as f:
            assert f.read().strip() == FORMAT_V2


def test_colliding_keys_roundtrip_after_reopen(tmp_path):
    """Keys that the legacy layout merged must stay distinct in v2.

    Three groups: punctuation removed by filename cleaning, string vs int
    (and bool) of the same name, and nested tuples that flatten to the
    same path.  Data is written, the store is closed and reopened, and
    every key must read back its own original bytes.
    """
    path = str(tmp_path / 'store')
    groups = [
        # punctuation that legacy cleaning collapsed to the same name
        ['Foo!bar?', 'Foobar', 'F!o@o#b$a%r^&'],
        # string vs int vs bool of the same name
        ['123', 123, '1', 1, True, False],
        # nested tuples that legacy tokenizing flattened to one path
        [('a', 'b', 'c'), ('a', ('b', 'c')), (('a', 'b'), 'c'),
         ('a/b', 'c'), 'a/b/c'],
    ]
    pairs = []
    p = File(path=path)
    assert p.format_version == FORMAT_V2
    for i, group in enumerate(groups):
        for j, key in enumerate(group):
            value = ('value-%d-%d' % (i, j)).encode()
            p.append({key: value})
            pairs.append((key, value))
    del p  # close

    q = File(path=path)  # reopen
    assert q.format_version == FORMAT_V2
    for key, value in pairs:
        assert q.get(key) == value
    q.__exit__()


def test_keys_stay_within_root(tmp_path):
    path = str(tmp_path / 'store')
    p = File(path=path)
    root = os.path.abspath(path)
    for key in ['..', '../..', 'a/../../b', ('..', '..'), '%2e%2e',
                '/absolute', ('.', '.')]:
        full = os.path.abspath(p.filename(key))
        assert os.path.commonpath((root, full)) == root
    p.__exit__()


def test_unsupported_key_type_raises(tmp_path):
    p = File(path=str(tmp_path / 'store'))
    with pytest.raises(TypeError):
        p.append({1.5: b'x'})
    p.__exit__()


def test_overlong_key_uses_index_mapping(tmp_path):
    path = str(tmp_path / 'store')
    key1 = 'k' * 500
    key2 = 'k' * 499 + 'z'
    key3 = ('d' * 300, 'e' * 300)
    p = File(path=path)
    p.append({key1: b'long1', key2: b'long2', key3: b'long3'})
    assert INDEX_DIR in os.listdir(path)
    # the index records the canonical key for collision detection
    digest = hashlib.sha256(key_to_relpath(key1).encode('utf-8')).hexdigest()
    with open(os.path.join(path, INDEX_DIR, digest + '.key')) as f:
        assert f.read() == key_to_relpath(key1)
    del p

    q = File(path=path)  # reopen
    assert q.get(key1) == b'long1'
    assert q.get(key2) == b'long2'
    assert q.get(key3) == b'long3'
    q.delete([key1])
    assert q.get(key1) == b''
    assert not os.path.exists(os.path.join(path, INDEX_DIR, digest + '.key'))
    q.__exit__()


def _write_legacy_store(path, files):
    os.makedirs(path)
    for rel, data in files.items():
        full = os.path.join(path, rel)
        dirname = os.path.dirname(full)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(full, 'wb') as f:
            f.write(data)


def test_legacy_directory_detected_not_reinterpreted(tmp_path):
    path = str(tmp_path / 'legacy')
    _write_legacy_store(path, {'x': b'hello',
                               os.path.join('a', 'b'): b'tuple'})
    p = File(path=path)
    assert p.format_version == FORMAT_V1
    assert not os.path.exists(os.path.join(path, FORMAT_MARKER))
    # legacy data is read with the legacy mapping
    assert p.get('x') == b'hello'
    assert p.get(('a', 'b')) == b'tuple'
    # legacy lossy behavior is retained, not silently upgraded
    p.append({'y!': b'Y'})
    assert os.path.exists(os.path.join(path, 'y'))
    assert p.get('y') == b'Y'
    p.__exit__()


def test_migrate_legacy_store(tmp_path):
    path = str(tmp_path / 'legacy')
    _write_legacy_store(path, {'x': b'hello',
                               os.path.join('a', 'b'): b'tuple'})
    p = File(path=path)
    assert p.format_version == FORMAT_V1
    p.migrate()
    assert p.format_version == FORMAT_V2
    with open(os.path.join(path, FORMAT_MARKER)) as f:
        assert f.read().strip() == FORMAT_V2
    assert p.get('x') == b'hello'
    assert p.get(('a', 'b')) == b'tuple'
    assert not os.path.exists(os.path.join(path, 'x'))
    del p

    q = File(path=path)  # reopen as v2
    assert q.format_version == FORMAT_V2
    assert q.get('x') == b'hello'
    assert q.get(('a', 'b')) == b'tuple'
    q.__exit__()


def test_migrate_conflict_unattributable_file(tmp_path):
    path = str(tmp_path / 'legacy')
    # '!' is stripped by the legacy cleaner, so this file cannot be
    # attributed to a unique key
    _write_legacy_store(path, {'Foo!bar': b'data'})
    p = File(path=path)
    assert p.format_version == FORMAT_V1
    with pytest.raises(MigrationConflictError):
        p.migrate()
    # nothing was written or reinterpreted
    assert p.format_version == FORMAT_V1
    assert not os.path.exists(os.path.join(path, FORMAT_MARKER))
    assert os.path.exists(os.path.join(path, 'Foo!bar'))
    p.__exit__()


def test_migrate_conflict_existing_target(tmp_path):
    path = str(tmp_path / 'legacy')
    # migrating legacy key 'x' would land on 'S1_x', which already exists
    _write_legacy_store(path, {'x': b'aaa', 'S1_x': b'bbb'})
    p = File(path=path)
    with pytest.raises(MigrationConflictError):
        p.migrate()
    assert p.format_version == FORMAT_V1
    assert not os.path.exists(os.path.join(path, FORMAT_MARKER))
    with open(os.path.join(path, 'x'), 'rb') as f:
        assert f.read() == b'aaa'
    with open(os.path.join(path, 'S1_x'), 'rb') as f:
        assert f.read() == b'bbb'
    p.__exit__()


def test_migrate_is_noop_on_v2(tmp_path):
    p = File(path=str(tmp_path / 'store'))
    p.append({'x': b'1'})
    p.migrate()
    assert p.format_version == FORMAT_V2
    assert p.get('x') == b'1'
    p.__exit__()
