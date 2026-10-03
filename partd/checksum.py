"""Checksummed framing layer for on-disk partitions.

Each appended payload is stored as a self-describing frame::

    +-------+---------+--------+=========+-------+
    | magic | version | length | payload | crc32 |
    +-------+---------+--------+=========+-------+
      4 B     1 B       8 B      length    4 B

The explicit magic marker distinguishes framed data from legacy
length-prefixed data (see ``partd.utils.frame``); old files are rejected
with :class:`FormatError` rather than silently mis-decoded.  The decoder
is incremental and tolerates arbitrary read-block boundaries, caps the
declared payload length to avoid huge allocations from corrupt headers,
and detects both truncation and bit flips.  By default any corruption
raises :class:`CorruptionError`; with ``recover=True`` only an
*incomplete* trailing frame is ignored -- checksum failures anywhere in
the stream always raise.
"""
import struct
import zlib

from toolz import valmap

from .core import Interface
from .file import File

MAGIC = b'PDCK'
VERSION = 1

_HEADER = struct.Struct('<4sBQ')  # magic, format version, payload length
_CRC = struct.Struct('<I')
HEADER_SIZE = _HEADER.size
CRC_SIZE = _CRC.size
FRAME_OVERHEAD = HEADER_SIZE + CRC_SIZE

#: Upper bound for a single frame's declared payload length.  A corrupt
#: header can claim an absurd length; refusing anything above this limit
#: keeps a bad header from triggering huge allocations or waits.
DEFAULT_MAX_FRAME_SIZE = 2 ** 31 - 1


class CorruptionError(Exception):
    """Raised when framed data is truncated, corrupt, or out of spec."""

    def __init__(self, message, offset=None, frame_index=None):
        super().__init__(message)
        self.offset = offset
        self.frame_index = frame_index


class FormatError(CorruptionError):
    """Raised when data lacks the framing magic or uses an unknown version."""


def pack_frame(payload, version=VERSION):
    """Pack a payload into a checksummed frame."""
    payload = bytes(payload)
    header = _HEADER.pack(MAGIC, version, len(payload))
    return header + payload + _CRC.pack(zlib.crc32(payload))


class FrameDecoder:
    """Incrementally decode a stream of checksummed frames.

    Bytes are supplied through :meth:`feed` in arbitrarily sized blocks;
    complete payloads are returned as they become available.  Call
    :meth:`finish` at end of stream to surface truncation.

    With ``recover=True``, :meth:`finish` ignores a single incomplete
    trailing frame (e.g. from a crashed append).  Complete frames with
    bad checksums, bad headers, or trailing garbage always raise
    :class:`CorruptionError` regardless of ``recover``.
    """

    def __init__(self, max_frame_size=DEFAULT_MAX_FRAME_SIZE, recover=False):
        self.max_frame_size = max_frame_size
        self.recover = recover
        self._buffer = bytearray()
        self.offset = 0  # stream offset of self._buffer[0]
        self.frames_decoded = 0

    def feed(self, data):
        """Consume a block of bytes; return any newly completed payloads."""
        self._buffer.extend(data)
        return self._drain()

    def finish(self):
        """Flush end of stream; return remaining complete payloads.

        Raises :class:`CorruptionError` on a truncated trailing frame
        unless ``recover`` was enabled.
        """
        out = self._drain()
        if self._buffer:
            # _drain only stops with a non-empty buffer when the leftover
            # is a valid prefix of a frame, i.e. an incomplete last frame.
            leftover = len(self._buffer)
            if self.recover:
                self.offset += leftover
                self._buffer.clear()
            else:
                raise CorruptionError(
                    'truncated frame %d at offset %d: %d bytes missing' % (
                        self.frames_decoded, self.offset, leftover),
                    offset=self.offset, frame_index=self.frames_decoded)
        return out

    def _drain(self):
        out = []
        buf = self._buffer
        while True:
            if len(buf) < HEADER_SIZE:
                head = bytes(buf[:len(MAGIC)])
                if head != MAGIC[:len(head)]:
                    raise FormatError(
                        'data at offset %d does not start with the framing '
                        'magic %r; it may be legacy unframed data' % (
                            self.offset, MAGIC),
                        offset=self.offset, frame_index=self.frames_decoded)
                break
            magic, version, length = _HEADER.unpack(bytes(buf[:HEADER_SIZE]))
            if magic != MAGIC:
                raise FormatError(
                    'bad frame magic %r at offset %d; expected %r. This may '
                    'be legacy unframed data' % (magic, self.offset, MAGIC),
                    offset=self.offset, frame_index=self.frames_decoded)
            if version != VERSION:
                raise FormatError(
                    'unsupported frame format version %d at offset %d' % (
                        version, self.offset),
                    offset=self.offset, frame_index=self.frames_decoded)
            if length > self.max_frame_size:
                raise CorruptionError(
                    'frame %d at offset %d declares length %d, above the '
                    'limit of %d bytes' % (
                        self.frames_decoded, self.offset, length,
                        self.max_frame_size),
                    offset=self.offset, frame_index=self.frames_decoded)
            need = FRAME_OVERHEAD + length
            if len(buf) < need:
                break
            payload = bytes(buf[HEADER_SIZE:HEADER_SIZE + length])
            expected_crc, = _CRC.unpack(
                buf[HEADER_SIZE + length:need])
            if zlib.crc32(payload) != expected_crc:
                raise CorruptionError(
                    'checksum mismatch in frame %d at offset %d' % (
                        self.frames_decoded, self.offset),
                    offset=self.offset, frame_index=self.frames_decoded)
            out.append(payload)
            del buf[:need]
            self.offset += need
            self.frames_decoded += 1
        return out


def unpack_frames(data, max_frame_size=DEFAULT_MAX_FRAME_SIZE,
                  recover=False):
    """Decode a full in-memory stream of frames into a list of payloads."""
    decoder = FrameDecoder(max_frame_size=max_frame_size, recover=recover)
    out = decoder.feed(data)
    out.extend(decoder.finish())
    return out


class Checksum(Interface):
    """Composable partd layer adding per-append integrity frames.

    Wraps any other partd (a :class:`partd.File` by default) and stores
    every appended payload as a length/version/checksum frame, so
    truncation and bit flips on disk are detected on read.

    With ``recover=True`` an incomplete trailing frame (e.g. from a
    crashed write) is ignored on read; any other corruption still raises
    :class:`CorruptionError`.
    """

    def __init__(self, partd=None, max_frame_size=DEFAULT_MAX_FRAME_SIZE,
                 recover=False):
        if not partd or isinstance(partd, str):
            partd = File(partd)
        self.partd = partd
        self.max_frame_size = max_frame_size
        self.recover = recover
        Interface.__init__(self)

    def __getstate__(self):
        return self.__dict__

    __setstate__ = Interface.__setstate__

    def append(self, data, **kwargs):
        self.partd.append(valmap(pack_frame, data), **kwargs)

    def _get(self, keys, **kwargs):
        raw = self.partd._get(keys, **kwargs)
        return [b''.join(unpack_frames(chunk,
                                       max_frame_size=self.max_frame_size,
                                       recover=self.recover))
                for chunk in raw]

    def delete(self, keys, **kwargs):
        return self.partd.delete(keys, **kwargs)

    def _iset(self, key, value, **kwargs):
        return self.partd.iset(key, pack_frame(value), **kwargs)

    def drop(self):
        return self.partd.drop()

    @property
    def lock(self):
        return self.partd.lock

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(*args)
