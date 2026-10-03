import os
import shutil

import pytest

from partd.file import (
    File,
    MigrationConflictError,
    migrate_to_v2,
    _detect_format,
)
from partd.key_encoding import (
    FORMAT_MARKER,
    FORMAT_VERSION,
    LEGACY_FORMAT_VERSION,
    INDEX_DIRNAME,
    decode_key,
    encode_key,
    key_to_relpath,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / 'store')


def test_new_directory_defaults_to_v2(store_path):
    f = File(store_path)
    try:
        assert f.format_version == FORMAT_VERSION
        with open(os.path.join(store_path, FORMAT_MARKER)) as marker:
            assert marker.read().strip() == 'partd-file-format: 2'
    finally:
        shutil.rmtree(store_path)


def test_punctuation_collision_group_persists(store_path):
    # v1 collapsed all of these to the same file ('Foobar' etc.)
    data = {'Foo!bar?': b'first', 'Foobar': b'second', 'Foo bar!': b'third'}
    f = File(store_path)
    f.append(data)
    del f

    reopened = File(store_path)
    try:
        assert reopened.format_version == FORMAT_VERSION
        for key, expected in data.items():
            assert reopened.get(key) == expected
        # three physically distinct files
        filenames = {reopened.filename(k) for k in data}
        assert len(filenames) == 3
        for fn in filenames:
            assert os.path.exists(fn)
    finally:
        shutil.rmtree(store_path)


def test_string_int_bool_collision_group_persists(store_path):
    # v1 collapsed '1', 1 and True to the same file
    data = {'1': b'string', 1: b'int', True: b'bool', '-1': b'negstr',
            -1: b'negint'}
    f = File(store_path)
    for key, value in data.items():
        f.iset(key, value)
    del f

    reopened = File(store_path)
    try:
        for key, expected in data.items():
            assert reopened.get(key) == expected
        # bool and int are encoded as different types
        assert reopened.filename(True) != reopened.filename(1)
        assert reopened.filename('1') != reopened.filename(1)
    finally:
        shutil.rmtree(store_path)


def test_nested_tuple_collision_group_persists(store_path):
    # v1 flattened all of these to 'a/b' or 'a/b/c'
    data = {('a', 'b'): b'tuple',
            'a/b': b'string',
            (('a', 'b'), 'c'): b'nested-left',
            ('a', ('b', 'c')): b'nested-right',
            ('a', 'b', 'c'): b'flat'}
    f = File(store_path)
    f.append(data)
    del f

    reopened = File(store_path)
    try:
        for key, expected in data.items():
            assert reopened.get(key) == expected
        filenames = {reopened.filename(k) for k in data}
        assert len(filenames) == len(data)
    finally:
        shutil.rmtree(store_path)


def test_encoding_roundtrip():
    keys = ['', 'hello', 'a/b', 'a b', '%', '..', 'Foo!bar?',
            'unicode-é-key', 0, 1, -1, 123456789, True, False,
            (), ('a',), ('a', 'b'), (1, 'a', True), (('a', 'b'), ('c',)),
            ('nested', ('tuple', ('deep', 3)))]
    for key in keys:
        assert decode_key(encode_key(key)) == key


def test_bool_and_int_are_distinct_types():
    assert decode_key(encode_key(True)) is True
    assert decode_key(encode_key(1)) == 1
    assert not isinstance(decode_key(encode_key(1)), bool)
    assert encode_key(True) != encode_key(1)
    assert encode_key(False) != encode_key(0)


def test_keys_stay_within_root(store_path):
    hostile = ['..', '../..', '/absolute', 'a/../../x', '%2e%2e',
               '.partd-format', '.lock', 'h' + '0' * 64, '\x00null']
    f = File(store_path)
    try:
        root = os.path.realpath(store_path)
        f.append({key: b'data' for key in hostile})
        for key in hostile:
            fn = os.path.realpath(f.filename(key))
            assert fn.startswith(root + os.sep), (key, fn)
            assert f.get(key) == b'data'
        for dirpath, dirnames, filenames in os.walk(store_path):
            assert os.path.realpath(dirpath).startswith(root)
    finally:
        shutil.rmtree(store_path)


def test_long_key_uses_index_not_truncation(store_path):
    long_key = 'k' * 1000
    other_long = 'k' * 999 + 'j'
    f = File(store_path)
    try:
        f.iset(long_key, b'long')
        f.iset(other_long, b'other')
        rel = key_to_relpath(long_key, f._index_dir)
        assert rel.startswith('h')
        assert len(rel) < 255
        # recorded in the index and reversible
        index_dir = os.path.join(store_path, INDEX_DIRNAME)
        assert decode_key(rel, index_dir) == long_key
        # near-identical long keys do not collide
        assert f.get(long_key) == b'long'
        assert f.get(other_long) == b'other'
    finally:
        shutil.rmtree(store_path)


def test_unsupported_key_type_raises(store_path):
    f = File(store_path)
    try:
        with pytest.raises(TypeError):
            f.filename(1.5)
        with pytest.raises(TypeError):
            f.filename(None)
        with pytest.raises(TypeError):
            f.filename(b'bytes')
    finally:
        shutil.rmtree(store_path)


def test_legacy_directory_detected_and_not_reinterpreted(store_path):
    os.makedirs(store_path)
    # a v1-style file produced by escape_filename('Foo!bar?')
    with open(os.path.join(store_path, 'Foobar'), 'wb') as f:
        f.write(b'legacy')

    legacy = File(store_path)
    try:
        assert legacy.format_version == LEGACY_FORMAT_VERSION
        # legacy mangling still in effect for this directory
        assert legacy.get('Foo!bar?') == b'legacy'
        assert legacy.get('Foobar') == b'legacy'
        # no marker is written into legacy directories
        assert not os.path.exists(os.path.join(store_path, FORMAT_MARKER))
    finally:
        shutil.rmtree(store_path)


def test_migrate_legacy_directory(store_path):
    os.makedirs(os.path.join(store_path, 'a'))
    with open(os.path.join(store_path, 'x'), 'wb') as f:
        f.write(b'one')
    with open(os.path.join(store_path, 'a', 'b'), 'wb') as f:
        f.write(b'two')

    migrate_to_v2(store_path)

    migrated = File(store_path)
    try:
        assert migrated.format_version == FORMAT_VERSION
        assert migrated.get('x') == b'one'
        assert migrated.get(('a', 'b')) == b'two'
        # old layout is gone
        assert not os.path.exists(os.path.join(store_path, 'x'))
        assert not os.path.exists(os.path.join(store_path, 'a'))
    finally:
        shutil.rmtree(store_path)


def test_migrate_detects_conflict_before_writing(store_path):
    # 'x' migrates to 'sx', which already exists as a legacy directory
    os.makedirs(os.path.join(store_path, 'sx'))
    with open(os.path.join(store_path, 'x'), 'wb') as f:
        f.write(b'one')
    with open(os.path.join(store_path, 'sx', 'y'), 'wb') as f:
        f.write(b'two')

    with pytest.raises(MigrationConflictError):
        migrate_to_v2(store_path)

    # nothing was written or moved
    assert not os.path.exists(os.path.join(store_path, FORMAT_MARKER))
    with open(os.path.join(store_path, 'x'), 'rb') as f:
        assert f.read() == b'one'
    with open(os.path.join(store_path, 'sx', 'y'), 'rb') as f:
        assert f.read() == b'two'
    shutil.rmtree(store_path)


def test_migrate_rejects_non_legacy(store_path):
    f = File(store_path)  # fresh v2 directory
    try:
        with pytest.raises(ValueError):
            migrate_to_v2(store_path)
    finally:
        shutil.rmtree(store_path)


def test_migrated_directory_reopens_with_original_data(store_path):
    os.makedirs(store_path)
    with open(os.path.join(store_path, 'Foobar'), 'wb') as f:
        f.write(b'legacy-data')
    migrate_to_v2(store_path)

    reopened = File(store_path)
    try:
        # the legacy filename is now a plain string key
        assert reopened.get('Foobar') == b'legacy-data'
        # and punctuation keys are independent again
        reopened.iset('Foo!bar?', b'new-data')
        assert reopened.get('Foobar') == b'legacy-data'
        assert reopened.get('Foo!bar?') == b'new-data'
    finally:
        shutil.rmtree(store_path)
