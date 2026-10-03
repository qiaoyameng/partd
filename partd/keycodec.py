"""Typed, reversible encoding of partd keys to filesystem-safe paths.

This module implements the key encoding used by the version 2 ``File``
storage layout.  The legacy layout (format 1) stripped characters that
were not "filename safe" and stringified everything, irreversibly merging
distinct keys (``'Foo!bar?'`` and ``'Foobar'``, ``'123'`` and ``123``,
``('a', ('b', 'c'))`` and ``('a', 'b', 'c')``).  The encoding here is a
bijection on supported keys, so distinct keys never share a filename.

Supported keys are ``str``, ``int``, ``bool`` and (arbitrarily nested)
tuples of these.  ``bool`` is deliberately a *different* type from
``int``: ``True``/``False`` encode as ``B1``/``B0`` and are not
interchangeable with the integers ``1``/``0``, even though they compare
equal in Python.

Encoding grammar (self-delimiting, recursive)::

    bool  -> "B1" | "B0"
    int   -> "I" + decimal ASCII ("-" for negatives)
    str   -> "S" + <payload length> + "_" + payload
             where payload percent-encodes the UTF-8 bytes of the string:
             bytes in [A-Za-z0-9-_. ] are kept, every other byte (including
             "%" and path separators) becomes %XX
    tuple -> "T" + <element count> + "_" + concatenated element encodings

A top-level tuple of two or more elements maps to one directory segment
per element (preserving the legacy habit of grouping tuple keys into
subdirectories); every other key maps to a single segment.  Encoded
segments never contain a path separator and always start with a type tag
(``B``, ``I``, ``S`` or ``T``), so no encoded key can be ``.`` or ``..``
or otherwise escape the store root.

Segments and whole relative paths have filesystem length limits.  Keys
whose encoding exceeds them are not truncated (that would create
collisions); callers route them through the content-addressed index
mapping provided by :func:`indexed_relpath` instead.
"""

import hashlib
import os

__all__ = ['INDEX_DIR', 'KeyEncodingError', 'KeyCollisionError',
           'MAX_SEGMENT_BYTES', 'MAX_RELPATH_BYTES',
           'encode_key', 'decode_key', 'key_to_relpath', 'relpath_to_key',
           'relpath_needs_index', 'indexed_relpath']

#: Directory (relative to the store root) holding index-mapped keys.
#: It cannot collide with an encoded key: every encoded segment starts
#: with an uppercase type tag.
INDEX_DIR = '_index'

#: Conservative limits (bytes) for one path segment and one relative path.
#: ext4 allows 255 bytes per filename and 4096 bytes per path.
MAX_SEGMENT_BYTES = 200
MAX_RELPATH_BYTES = 3800

_SAFE_BYTES = frozenset(
    b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_. ')

_HEX = '0123456789ABCDEF'


class KeyEncodingError(ValueError):
    """A key cannot be encoded or decoded safely."""


class KeyCollisionError(RuntimeError):
    """Two distinct keys map to the same storage location."""


def _encode_string(value):
    payload = []
    for byte in value.encode('utf-8'):
        if byte in _SAFE_BYTES:
            payload.append(chr(byte))
        else:
            payload.append('%' + _HEX[byte >> 4] + _HEX[byte & 0xF])
    payload = ''.join(payload)
    return 'S%d_%s' % (len(payload), payload)


def encode_key(key):
    """Encode a supported key to a single self-delimiting segment.

    >>> encode_key('hello')
    'S5_hello'
    >>> encode_key(('a', 1))
    'T2_S1_aI1'
    """
    if isinstance(key, bool):
        return 'B1' if key else 'B0'
    if isinstance(key, int):
        return 'I%d' % key
    if isinstance(key, str):
        return _encode_string(key)
    if isinstance(key, tuple):
        return 'T%d_%s' % (len(key), ''.join(encode_key(k) for k in key))
    raise TypeError(
        'unsupported key type %s; the version 2 File layout supports '
        'str, int, bool and (nested) tuples of these' % type(key).__name__)


def _decode_string(payload):
    data = bytearray()
    i = 0
    while i < len(payload):
        ch = payload[i]
        if ch == '%':
            try:
                data.append(int(payload[i + 1:i + 3], 16))
            except ValueError:
                raise KeyEncodingError(
                    'invalid percent escape in %r' % payload)
            i += 3
        else:
            data.append(ord(ch))
            i += 1
    try:
        return bytes(data).decode('utf-8')
    except UnicodeDecodeError:
        raise KeyEncodingError('invalid UTF-8 in %r' % payload)


def _parse(text, pos):
    if pos >= len(text):
        raise KeyEncodingError('truncated key encoding: %r' % text)
    tag = text[pos]
    if tag == 'B':
        bit = text[pos + 1:pos + 2]
        if bit == '0':
            return False, pos + 2
        if bit == '1':
            return True, pos + 2
        raise KeyEncodingError('invalid bool encoding: %r' % text)
    if tag == 'I':
        end = pos + 1
        if text[end:end + 1] == '-':
            end += 1
        digits = end
        while end < len(text) and text[end].isdigit():
            end += 1
        if end == digits:
            raise KeyEncodingError('invalid int encoding: %r' % text)
        return int(text[pos + 1:end]), end
    if tag == 'S' or tag == 'T':
        end = pos + 1
        while end < len(text) and text[end].isdigit():
            end += 1
        if end == pos + 1 or text[end:end + 1] != '_':
            raise KeyEncodingError('invalid length prefix: %r' % text)
        count = int(text[pos + 1:end])
        if tag == 'S':
            payload = text[end + 1:end + 1 + count]
            if len(payload) != count:
                raise KeyEncodingError('truncated key encoding: %r' % text)
            return _decode_string(payload), end + 1 + count
        items = []
        p = end + 1
        for _ in range(count):
            item, p = _parse(text, p)
            items.append(item)
        return tuple(items), p
    raise KeyEncodingError('unknown type tag %r in %r' % (tag, text))


def decode_key(text):
    """Inverse of :func:`encode_key` for a single segment."""
    key, end = _parse(text, 0)
    if end != len(text):
        raise KeyEncodingError('trailing data in key encoding: %r' % text)
    return key


def key_to_relpath(key):
    """Encode a key to a relative path below the store root.

    Top-level tuples of two or more elements become one path segment per
    element; all other keys become a single segment.
    """
    if isinstance(key, tuple) and len(key) >= 2:
        return os.sep.join(encode_key(element) for element in key)
    return encode_key(key)


def relpath_to_key(relpath):
    """Inverse of :func:`key_to_relpath`."""
    parts = relpath.split(os.sep)
    if len(parts) >= 2:
        return tuple(decode_key(part) for part in parts)
    return decode_key(parts[0])


def relpath_needs_index(relpath):
    """True if the encoded relative path exceeds filesystem limits."""
    if len(relpath.encode('utf-8')) > MAX_RELPATH_BYTES:
        return True
    return any(len(part.encode('utf-8')) > MAX_SEGMENT_BYTES
               for part in relpath.split(os.sep))


def indexed_relpath(relpath):
    """Content-addressed relative path for an overlong encoded key.

    The digest is derived from the full canonical encoding, so distinct
    keys map to distinct locations without any truncation.
    """
    digest = hashlib.sha256(relpath.encode('utf-8')).hexdigest()
    return os.sep.join((INDEX_DIR, digest))
