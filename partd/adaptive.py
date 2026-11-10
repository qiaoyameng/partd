""" Adaptive per-frame compression layer

Each appended block is stored as a self-describing frame:

    header:  magic (2 bytes) | version (1 byte) | codec (1 byte) |
             raw length (8 bytes) | payload length (8 bytes)
    payload: raw or compressed bytes

The codec is chosen independently for every frame, so a single partition
may freely mix raw, zlib, and bz2 frames.  Frame order on disk always
matches append order, so concatenated reads reproduce the logical byte
stream exactly.

Only standard library codecs are used.  Data written without frame
headers (legacy partd files) is rejected unless the layer is created
with ``legacy=True``.
"""
import bz2
import struct
import zlib

from .core import Interface
from .file import File

__all__ = ['Adaptive', 'AdaptiveError', 'read_frames',
           'CODEC_RAW', 'CODEC_ZLIB', 'CODEC_BZ2']

MAGIC = b'PA'
VERSION = 1

CODEC_RAW = 0
CODEC_ZLIB = 1
CODEC_BZ2 = 2

_codec_names = {CODEC_RAW: 'raw', CODEC_ZLIB: 'zlib', CODEC_BZ2: 'bz2'}

_header = struct.Struct('<2sBBQQ')
HEADER_SIZE = _header.size

DEFAULT_MIN_BLOCK_SIZE = 128
DEFAULT_MIN_SAVINGS = 0.05
DEFAULT_MAX_FRAME_SIZE = 1 << 31


class AdaptiveError(ValueError):
    """ Corrupt, truncated, or unsupported adaptive-compression frame """


def codec_name(codec):
    return _codec_names.get(codec, 'unknown(%d)' % codec)


def encode_frame(data, min_block_size=DEFAULT_MIN_BLOCK_SIZE,
                 min_savings=DEFAULT_MIN_SAVINGS):
    """ Encode one block as a framed payload, choosing the codec adaptively

    Blocks smaller than ``min_block_size`` are stored raw.  Otherwise the
    best of zlib/bz2 is used, but only if it shrinks the block by at least
    the ``min_savings`` fraction; otherwise the block stays raw.
    """
    codec = CODEC_RAW
    payload = data
    if len(data) >= min_block_size:
        best_codec, best_payload = CODEC_RAW, data
        for candidate, compress in ((CODEC_ZLIB, zlib.compress),
                                    (CODEC_BZ2, bz2.compress)):
            compressed = compress(data)
            if len(compressed) < len(best_payload):
                best_codec, best_payload = candidate, compressed
        if len(best_payload) < len(data) * (1 - min_savings):
            codec, payload = best_codec, best_payload
    return (_header.pack(MAGIC, VERSION, codec, len(data), len(payload))
            + payload)


def _decompress_zlib(payload, limit, key, index):
    decompressor = zlib.decompressobj()
    out = []
    total = 0
    data = payload
    while True:
        piece = decompressor.decompress(data, limit - total + 1)
        out.append(piece)
        total += len(piece)
        if total > limit:
            raise AdaptiveError(
                'partition %r frame %d: decompressed size exceeds limit of '
                '%d bytes' % (key, index, limit))
        data = decompressor.unconsumed_tail
        if not data:
            break
    if not decompressor.eof:
        raise AdaptiveError(
            'partition %r frame %d: truncated zlib stream' % (key, index))
    return b''.join(out)


def _decompress_bz2(payload, limit, key, index):
    decompressor = bz2.BZ2Decompressor()
    out = []
    total = 0
    data = payload
    while True:
        piece = decompressor.decompress(data, limit - total + 1)
        data = b''
        out.append(piece)
        total += len(piece)
        if total > limit:
            raise AdaptiveError(
                'partition %r frame %d: decompressed size exceeds limit of '
                '%d bytes' % (key, index, limit))
        if decompressor.eof:
            return b''.join(out)
        if decompressor.needs_input:
            raise AdaptiveError(
                'partition %r frame %d: truncated bz2 stream' % (key, index))


def decompress_frame(codec, payload, raw_length, max_frame_size, key=None,
                     index=None):
    """ Decompress one frame payload, enforcing the output cap in-stream """
    limit = min(raw_length, max_frame_size)
    if codec == CODEC_RAW:
        result = payload
    elif codec == CODEC_ZLIB:
        result = _decompress_zlib(payload, limit, key, index)
    elif codec == CODEC_BZ2:
        result = _decompress_bz2(payload, limit, key, index)
    else:
        raise AdaptiveError(
            'partition %r frame %d: unknown codec %d' % (key, index, codec))
    if len(result) != raw_length:
        raise AdaptiveError(
            'partition %r frame %d: decompressed length %d does not match '
            'declared length %d' % (key, index, len(result), raw_length))
    return result


def read_frames(data, key=None):
    """ Yield (index, codec, raw_length, payload) for each frame in ``data`` """
    i = 0
    n = len(data)
    index = 0
    while i < n:
        if data[i:i + len(MAGIC)] != MAGIC:
            raise AdaptiveError(
                'partition %r frame %d: missing adaptive frame magic; this '
                'looks like legacy unmarked data, read it with legacy=True'
                % (key, index))
        if n - i < HEADER_SIZE:
            raise AdaptiveError(
                'partition %r frame %d: truncated header (%d bytes left)'
                % (key, index, n - i))
        magic, version, codec, raw_length, payload_length = \
            _header.unpack_from(data, i)
        if version != VERSION:
            raise AdaptiveError(
                'partition %r frame %d: unsupported version %d'
                % (key, index, version))
        if codec not in _codec_names:
            raise AdaptiveError(
                'partition %r frame %d: unknown codec %d'
                % (key, index, codec))
        i += HEADER_SIZE
        payload = data[i:i + payload_length]
        if len(payload) != payload_length:
            raise AdaptiveError(
                'partition %r frame %d: truncated payload (expected %d '
                'bytes, found %d)' % (key, index, payload_length,
                                      len(payload)))
        i += payload_length
        yield index, codec, raw_length, payload
        index += 1


class Adaptive(Interface):
    """ Per-frame adaptive compression wrapper

    Parameters
    ----------

    partd: Interface or str or None
        Inner partd to wrap; defaults to a temporary File
    min_block_size: int
        Blocks smaller than this are always stored raw
    min_savings: float in [0, 1)
        Compressed payloads must save at least this fraction of the
        original size, otherwise the block is stored raw
    max_frame_size: int
        Hard cap on the decompressed size of a single frame, enforced
        while decompressing rather than after allocation
    legacy: bool
        Read partitions as raw unmarked bytes (old data without frame
        headers).  Off by default; new appends always use framed format.
    """
    def __init__(self, partd=None, min_block_size=DEFAULT_MIN_BLOCK_SIZE,
                 min_savings=DEFAULT_MIN_SAVINGS,
                 max_frame_size=DEFAULT_MAX_FRAME_SIZE, legacy=False):
        if not 0 <= min_savings < 1:
            raise ValueError('min_savings must be in [0, 1)')
        if min_block_size < 0:
            raise ValueError('min_block_size must be non-negative')
        if max_frame_size < 0:
            raise ValueError('max_frame_size must be non-negative')
        if not partd or isinstance(partd, str):
            partd = File(partd)
        self.partd = partd
        self.min_block_size = min_block_size
        self.min_savings = min_savings
        self.max_frame_size = max_frame_size
        self.legacy = legacy
        Interface.__init__(self)

    def __getstate__(self):
        return self.__dict__

    __setstate__ = Interface.__setstate__

    def append(self, data, **kwargs):
        framed = {k: encode_frame(v, self.min_block_size, self.min_savings)
                  for k, v in data.items()}
        self.partd.append(framed, **kwargs)

    def _get(self, keys, **kwargs):
        raw = self.partd._get(keys, **kwargs)
        return [self._decode(key, chunk) for key, chunk in zip(keys, raw)]

    def _decode(self, key, chunk):
        if self.legacy:
            return chunk
        return b''.join(
            decompress_frame(codec, payload, raw_length,
                             self.max_frame_size, key, index)
            for index, codec, raw_length, payload
            in read_frames(chunk, key))

    def delete(self, keys, **kwargs):
        return self.partd.delete(keys, **kwargs)

    def _iset(self, key, value, **kwargs):
        framed = encode_frame(value, self.min_block_size, self.min_savings)
        return self.partd.iset(key, framed, **kwargs)

    def drop(self):
        return self.partd.drop()

    @property
    def lock(self):
        return self.partd.lock

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(*args)
