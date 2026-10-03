import pytest

from partd.keycodec import (INDEX_DIR, KeyEncodingError, decode_key,
                            encode_key, indexed_relpath, key_to_relpath,
                            relpath_needs_index, relpath_to_key)


ROUNDTRIP_KEYS = [
    'hello', '', 'Foo!bar?', 'a/b', 'a\\b', '%', '%25', 'S1_x', '_index',
    '.', '..', '.lock', 'snowman ☃/雪', 'with space', 'x' * 500,
    0, 1, -1, 123, -987654321,
    True, False,
    (), ('a',), ('a', 'b'), ('a', 'b', 'c'),
    ('a', ('b', 'c')), (('a', 'b'), 'c'), (('a', 'b'),),
    (1, 'a', True, (2,)), ('S1_a', 'S1_b'), ('a/b', 'c'),
    (('deep', ('nesting', ('is', ('fine',))),),),
]


@pytest.mark.parametrize('key', ROUNDTRIP_KEYS)
def test_roundtrip(key):
    relpath = key_to_relpath(key)
    assert relpath_to_key(relpath) == key


def test_bool_is_distinct_from_int():
    assert encode_key(True) != encode_key(1)
    assert encode_key(False) != encode_key(0)
    assert decode_key(encode_key(True)) is True
    assert decode_key(encode_key(1)) == 1
    assert type(decode_key(encode_key(1))) is int


def test_distinct_keys_never_collide():
    groups = [
        ['Foo!bar?', 'Foobar', 'F!o@o#b$a%r^&'],
        ['123', 123, '1', 1, True],
        [('a', 'b', 'c'), ('a', ('b', 'c')), (('a', 'b'), 'c'),
         ('a/b', 'c'), 'a/b/c'],
    ]
    for group in groups:
        encoded = [key_to_relpath(key) for key in group]
        assert len(set(encoded)) == len(encoded)


def test_encoded_segments_are_filesystem_safe():
    for key in ROUNDTRIP_KEYS:
        for segment in key_to_relpath(key).split('/'):
            assert segment not in ('', '.', '..')
            assert '/' not in segment and '\\' not in segment


def test_unsupported_key_type_raises():
    for key in [1.5, b'bytes', None, ['list'], {'set'}, {'dict': 1}]:
        with pytest.raises(TypeError):
            key_to_relpath(key)


def test_decode_rejects_garbage():
    for bad in ['', 'X1', 'S', 'S3_ab', 'S4_abx', 'I', 'I-', 'B2', 'T2_S1_a',
                'S2_%zz', 'S1_a trailing']:
        with pytest.raises(KeyEncodingError):
            decode_key(bad)


def test_index_mapping_for_overlong_keys():
    key = 'k' * 500
    relpath = key_to_relpath(key)
    assert relpath_needs_index(relpath)
    indexed = indexed_relpath(relpath)
    assert indexed.startswith(INDEX_DIR + '/')
    assert not relpath_needs_index(indexed)
    # distinct overlong keys map to distinct index entries
    other = indexed_relpath(key_to_relpath('k' * 499 + 'z'))
    assert indexed != other
    # short keys never need the index
    assert not relpath_needs_index(key_to_relpath(('a', 'b')))
