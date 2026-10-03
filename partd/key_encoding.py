"""Reversible, typed key encoding for the v2 File storage layout.

The legacy (v1) layout mangles keys with ``escape_filename``, which simply
deletes characters it does not like.  That is lossy: ``'Foo!bar?'`` and
``'Foobar'`` collide, the string ``'1'`` and the integer ``1`` collide, and
the nested tuple ``('a', 'b')`` collides with the string ``'a/b'``.

The v2 layout instead encodes every key as a typed, reversible relative
path:

- ``str``   -> ``s`` + percent-encoded UTF-8 (only ``[A-Za-z0-9._-]`` kept)
- ``bool``  -> ``b0`` / ``b1`` (distinct from ``int`` even though
  ``isinstance(True, int)``)
- ``int``   -> ``i`` + decimal digits
- ``tuple`` -> ``T`` + length-prefixed encoded items, recursively

Every key maps to a single flat file name (tuples are *not* spread over
subdirectories, so a key can never conflict with a directory prefix of
another key).  Segment and type boundaries are preserved by the
length-prefixed encoding.  Names always carry a type tag, so they can
never be ``.``/``..`` or contain a path separator: encoded keys always
stay inside the store root.

Keys whose encoded path would exceed filesystem limits are stored under
``h<sha256>`` and the original encoding is recorded in an index directory,
keeping the mapping reversible without truncation.
"""

import hashlib
import os

FORMAT_MARKER = '.partd-format'
FORMAT_VERSION = 2
LEGACY_FORMAT_VERSION = 1
INDEX_DIRNAME = '.partd-key-index'

MARKER_TEMPLATE = 'partd-file-format: %d\n'

MAX_SEGMENT_BYTES = 255
MAX_PATH_BYTES = 4000

_SAFE_BYTES = frozenset(
    b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-')
_SAFE_CHARS = frozenset(chr(b) for b in _SAFE_BYTES)

_HASH_PREFIX = 'h'


def _quote(text):
    """ Percent-encode a string, keeping only unambiguous characters

    >>> _quote('a/b?')
    'a%2Fb%3F'
    >>> _unquote(_quote('a/b? \\u00e9')) == 'a/b? \\u00e9'
    True
    """
    out = []
    for byte in text.encode('utf-8'):
        if byte in _SAFE_BYTES:
            out.append(chr(byte))
        else:
            out.append('%%%02X' % byte)
    return ''.join(out)


def _unquote(text):
    out = bytearray()
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '%':
            hexpart = text[i + 1:i + 3]
            if len(hexpart) != 2:
                raise ValueError('invalid percent-encoding: %r' % text)
            out.append(int(hexpart, 16))  # raises ValueError on bad hex
            i += 3
        elif ch in _SAFE_CHARS:
            out.append(ord(ch))
            i += 1
        else:
            raise ValueError('invalid character %r in encoded key' % ch)
    return bytes(out).decode('utf-8')


def encode_segment(key):
    """ Encode a single (non top-level-tuple) key component

    >>> encode_segment('a/b')
    'sa%2Fb'
    >>> encode_segment(42)
    'i42'
    >>> encode_segment(True)
    'b1'
    >>> encode_segment(('a', 1))
    'T2.sa2.i1'
    """
    if isinstance(key, bool):
        return 'b1' if key else 'b0'
    if isinstance(key, int):
        return 'i' + str(key)
    if isinstance(key, str):
        return 's' + _quote(key)
    if isinstance(key, tuple):
        parts = [encode_segment(k) for k in key]
        return 'T' + ''.join('%d.%s' % (len(p), p) for p in parts)
    raise TypeError('unsupported key type for File storage: %r' % type(key))


def decode_segment(segment):
    """ Inverse of encode_segment

    >>> decode_segment('sa%2Fb')
    'a/b'
    >>> decode_segment('i42')
    42
    >>> decode_segment('b1')
    True
    >>> decode_segment('T2.sa2.i1')
    ('a', 1)
    """
    if not segment:
        raise ValueError('empty encoded key segment')
    tag, rest = segment[0], segment[1:]
    if tag == 's':
        return _unquote(rest)
    if tag == 'b':
        if rest == '1':
            return True
        if rest == '0':
            return False
        raise ValueError('invalid bool segment: %r' % segment)
    if tag == 'i':
        return int(rest)  # raises ValueError on garbage
    if tag == 'T':
        return tuple(decode_segment(item) for item in _parse_items(rest))
    raise ValueError('unknown type tag in encoded key: %r' % segment)


def _parse_items(text):
    items = []
    i = 0
    while i < len(text):
        j = text.find('.', i)
        if j < 0:
            raise ValueError('malformed tuple encoding: %r' % text)
        n = int(text[i:j])
        item = text[j + 1:j + 1 + n]
        if len(item) != n:
            raise ValueError('truncated tuple encoding: %r' % text)
        items.append(item)
        i = j + 1 + n
    return items


def encode_key(key):
    """ Encode a key as a relative path

    Top-level tuples become one path segment per element; nested tuples
    stay a single segment, so segment boundaries are preserved.

    >>> encode_key('a/b')
    'sa%2Fb'
    >>> encode_key(('a', 'b'))
    'T2.sa2.sb'
    >>> encode_key(('a',))
    'T2.sa'
    >>> encode_key((('a', 'b'), 'c'))
    'T9.T2.sa2.sb2.sc'
    """
    return encode_segment(key)


def decode_key(relpath, index_dir=None):
    """ Inverse of encode_key

    Hash-based names (overlong keys) are resolved through the index
    directory when given.

    >>> decode_key(encode_key(('a', 1, ('x', True))))
    ('a', 1, ('x', True))
    """
    if relpath.startswith(_HASH_PREFIX) and '/' not in relpath:
        if index_dir is None:
            raise ValueError('hash-based key %r requires the index directory'
                             % relpath)
        with open(os.path.join(index_dir, relpath[1:]),
                  encoding='utf-8') as f:
            relpath = f.read()
    return decode_segment(relpath)


def key_to_relpath(key, index_dir=None):
    """ Map a key to the relative path used inside the store

    Overlong encodings are replaced by ``h<sha256>`` and recorded in
    ``index_dir`` (when given) so the mapping stays reversible.  Keys are
    never truncated.
    """
    rel = encode_key(key)
    too_long = (len(rel.encode('utf-8')) > MAX_PATH_BYTES or
                any(len(segment.encode('utf-8')) > MAX_SEGMENT_BYTES
                    for segment in rel.split('/')))
    if not too_long:
        return rel
    digest = hashlib.sha256(rel.encode('utf-8')).hexdigest()
    if index_dir is not None:
        _record_index(index_dir, digest, rel)
    return _HASH_PREFIX + digest


def _record_index(index_dir, digest, relpath):
    if not os.path.isdir(index_dir):
        os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, digest)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(relpath)
