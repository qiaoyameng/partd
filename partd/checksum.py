import struct
import zlib

from toolz import valmap

from .core import Interface
from .file import File


MAGIC = b'PDCK'
FORMAT_VERSION = 1
HEADER_DATA = struct.Struct('>4sBQ')
HEADER = struct.Struct('>4sBQI')
CHECKSUM = struct.Struct('>I')
HEADER_SIZE = HEADER.size
CHECKSUM_SIZE = CHECKSUM.size
DEFAULT_MAX_PAYLOAD_SIZE = 1 << 30


class ChecksumError(ValueError):
    pass


class InvalidChecksumFrame(ChecksumError):
    def __init__(self, message, offset=None, frame=None):
        self.offset = offset
        self.frame = frame
        location = ''
        if offset is not None:
            location = f' at byte {offset}'
        if frame is not None:
            location += f' (frame {frame})'
        super().__init__(message + location)


class InvalidFormat(InvalidChecksumFrame):
    pass


class UnsupportedFormatVersion(InvalidChecksumFrame):
    pass


class FrameTooLarge(InvalidChecksumFrame):
    pass


class TruncatedFrame(InvalidChecksumFrame):
    pass


class ChecksumMismatch(InvalidChecksumFrame):
    pass


def encode_frame(payload, *, max_payload_size=DEFAULT_MAX_PAYLOAD_SIZE):
    payload_size = len(payload)
    if payload_size > max_payload_size:
        raise FrameTooLarge(
            f'payload of size {payload_size} exceeds limit {max_payload_size}'
        )

    header_data = HEADER_DATA.pack(MAGIC, FORMAT_VERSION, payload_size)
    header = HEADER.pack(
        MAGIC, FORMAT_VERSION, payload_size,
        zlib.crc32(header_data)
    )
    return header + payload + CHECKSUM.pack(zlib.crc32(payload))


class ChecksumFrameDecoder:
    def __init__(self, *, max_payload_size=DEFAULT_MAX_PAYLOAD_SIZE,
                 recover=False):
        if max_payload_size < 0:
            raise ValueError('max_payload_size must be non-negative')
        self.max_payload_size = max_payload_size
        self.recover = recover
        self.frame_count = 0
        self._buffer = bytearray()
        self._position = 0
        self._offset = 0

    def feed(self, chunk):
        if chunk:
            self._buffer.extend(chunk)

        frames = []
        while True:
            available = len(self._buffer) - self._position
            if available < HEADER_SIZE:
                break

            frame_offset = self._offset + self._position
            magic, version, payload_size, header_digest = HEADER.unpack_from(
                self._buffer, self._position
            )
            if magic != MAGIC:
                raise InvalidFormat(
                    'invalid checksum frame marker', frame_offset,
                    self.frame_count
                )
            header_data = bytes(
                self._buffer[
                    self._position:self._position + HEADER_DATA.size
                ]
            )
            if zlib.crc32(header_data) != header_digest:
                raise ChecksumMismatch(
                    'frame header checksum mismatch', frame_offset,
                    self.frame_count
                )
            if version != FORMAT_VERSION:
                raise UnsupportedFormatVersion(
                    f'unsupported format version {version}', frame_offset,
                    self.frame_count
                )
            if payload_size > self.max_payload_size:
                raise FrameTooLarge(
                    f'declared payload size {payload_size} exceeds limit '
                    f'{self.max_payload_size}',
                    frame_offset, self.frame_count
                )

            frame_size = HEADER_SIZE + payload_size + CHECKSUM_SIZE
            if available < frame_size:
                break

            payload_start = self._position + HEADER_SIZE
            checksum_start = payload_start + payload_size
            view = memoryview(self._buffer)
            try:
                digest = zlib.crc32(view[payload_start:checksum_start])
                expected = CHECKSUM.unpack_from(
                    self._buffer, checksum_start
                )[0]
                if digest != expected:
                    raise ChecksumMismatch(
                        'payload checksum mismatch', frame_offset,
                        self.frame_count
                    )
            finally:
                view.release()

            frames.append(bytes(
                self._buffer[payload_start:checksum_start]
            ))
            self.frame_count += 1
            self._position += frame_size

        self._consume_buffer()
        return frames

    def finish(self):
        remaining = len(self._buffer) - self._position
        if not remaining:
            return []

        frame_offset = self._offset + self._position
        if self.recover:
            return []

        raise TruncatedFrame(
            f'truncated final frame: {remaining} bytes available',
            frame_offset, self.frame_count
        )

    def _consume_buffer(self):
        if self._position:
            del self._buffer[:self._position]
            self._offset += self._position
            self._position = 0


class Checksum(Interface):
    def __init__(self, partd=None, *,
                 max_payload_size=DEFAULT_MAX_PAYLOAD_SIZE,
                 recover=False):
        if not partd or isinstance(partd, str):
            partd = File(partd)
        self.partd = partd
        self.max_payload_size = max_payload_size
        self.recover = recover
        Interface.__init__(self)

    def __getstate__(self):
        return self.__dict__

    __setstate__ = Interface.__setstate__

    def append(self, data, **kwargs):
        data = valmap(
            lambda payload: encode_frame(
                payload, max_payload_size=self.max_payload_size
            ),
            data
        )
        self.partd.append(data, **kwargs)

    def _get(self, keys, recover=None, **kwargs):
        if recover is None:
            recover = self.recover
        raw = self.partd._get(keys, **kwargs)

        result = []
        for chunk in raw:
            decoder = ChecksumFrameDecoder(
                max_payload_size=self.max_payload_size,
                recover=recover
            )
            result.append(b''.join(decoder.feed(chunk)))
            decoder.finish()
        return result

    def delete(self, keys, **kwargs):
        return self.partd.delete(keys, **kwargs)

    def _iset(self, key, value, **kwargs):
        return self.partd.iset(
            key,
            encode_frame(value, max_payload_size=self.max_payload_size),
            **kwargs
        )

    def drop(self):
        return self.partd.drop()

    @property
    def lock(self):
        return self.partd.lock

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(*args)
