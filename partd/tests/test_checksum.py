import zlib
import pickle
import struct

import pytest

from partd import Checksum, ZLib
from partd.checksum import (
    CHECKSUM_SIZE,
    HEADER,
    HEADER_DATA,
    HEADER_SIZE,
    MAGIC,
    ChecksumMismatch,
    FrameTooLarge,
    InvalidFormat,
    TruncatedFrame,
    UnsupportedFormatVersion,
    encode_frame,
    ChecksumFrameDecoder,
)


def decode_in_blocks(data, block_size, *, recover=False,
                     max_payload_size=1 << 20):
    decoder = ChecksumFrameDecoder(
        recover=recover,
        max_payload_size=max_payload_size
    )
    frames = []
    for start in range(0, len(data), block_size):
        frames.extend(decoder.feed(data[start:start + block_size]))
    frames.extend(decoder.finish())
    return frames


def test_checksum_round_trip():
    with Checksum() as p:
        p.append({'x': b'Hello', 'y': b'abc'})
        p.append({'x': b'World!', 'y': b'def'})

        assert p.get(['y', 'x']) == [
            b'abcdef', b'HelloWorld!'
        ]
        assert p.get('z') == b''


def test_checksum_composes_around_compression():
    with Checksum(ZLib()) as p:
        p.append({'x': b'Hello'})
        p.append({'x': b'World!'})

        assert p.get('x') == b'HelloWorld!'


def test_checksum_pickles_nested_partd(tmp_path):
    p = Checksum(ZLib(str(tmp_path / 'store')), recover=True)
    q = pickle.loads(pickle.dumps(p))

    q.append({'x': b'one'})
    assert q.get('x') == b'one'
    assert q.recover is True


def test_checksum_decoder_crosses_arbitrary_block_boundaries():
    payloads = [b'', b'a', b'abcd', b'x' * 37]
    data = b''.join(encode_frame(payload) for payload in payloads)

    for block_size in range(1, len(data) + 2):
        assert decode_in_blocks(data, block_size) == payloads


def test_recover_ignores_only_an_incomplete_final_frame():
    prefix_payloads = [b'one', b'two']
    prefix = b''.join(map(encode_frame, prefix_payloads))
    final = encode_frame(b'three')

    for cut in range(1, len(final)):
        truncated = prefix + final[:cut]
        for block_size in range(1, len(truncated) + 1):
            assert decode_in_blocks(
                truncated, block_size, recover=True
            ) == prefix_payloads

            with pytest.raises(TruncatedFrame) as exc_info:
                decode_in_blocks(truncated, block_size)

            assert exc_info.value.offset == len(prefix)
            assert exc_info.value.frame == len(prefix_payloads)


def test_middle_payload_corruption_is_not_recoverable():
    first = encode_frame(b'one')
    middle = encode_frame(b'two')
    third = encode_frame(b'three')

    payload_offset = len(first) + HEADER_SIZE
    for bit in range(8):
        corrupted = bytearray(first + middle + third)
        corrupted[payload_offset] ^= 1 << bit
        corrupted = bytes(corrupted)

        for block_size in range(1, len(corrupted) + 1):
            with pytest.raises(ChecksumMismatch) as strict:
                decode_in_blocks(corrupted, block_size)
            assert strict.value.offset == len(first)
            assert strict.value.frame == 1

            with pytest.raises(ChecksumMismatch) as recovered:
                decode_in_blocks(corrupted, block_size, recover=True)
            assert recovered.value.offset == len(first)
            assert recovered.value.frame == 1


def test_trailing_corrupt_complete_frame_is_not_recoverable():
    data = bytearray(encode_frame(b'one') + encode_frame(b'two'))
    data[-1] ^= 1

    with pytest.raises(ChecksumMismatch) as exc_info:
        decode_in_blocks(bytes(data), 3, recover=True)
    assert exc_info.value.frame == 1


def test_legacy_and_unknown_formats_require_marker_and_version():
    legacy = struct.pack('Q', 9) + b'abcdefghi'
    with pytest.raises(InvalidFormat) as exc_info:
        decode_in_blocks(legacy, 2)
    assert exc_info.value.offset == 0

    unsupported = HEADER.pack(
        MAGIC, 99, 0, zlib.crc32(HEADER_DATA.pack(MAGIC, 99, 0))
    ) + struct.pack('>I', 0)
    with pytest.raises(UnsupportedFormatVersion) as exc_info:
        decode_in_blocks(unsupported, 5)
    assert exc_info.value.offset == 0


def test_declared_payload_size_is_limited_before_body_is_available():
    oversized = HEADER.pack(
        MAGIC, 1, 10, zlib.crc32(HEADER_DATA.pack(MAGIC, 1, 10))
    )
    with pytest.raises(FrameTooLarge) as exc_info:
        decode_in_blocks(oversized, 5, max_payload_size=9)
    assert exc_info.value.offset == 0
    assert exc_info.value.frame == 0

    with pytest.raises(FrameTooLarge):
        encode_frame(b'x' * 10, max_payload_size=9)


def test_checksum_validates_written_file_format(tmp_path):
    path = tmp_path / 'store'
    p = Checksum(str(path))
    p.append({'x': b'old-style bytes'})

    with open(p.partd.filename('x'), 'wb') as f:
        f.write(struct.pack('Q', 15) + b'old-style bytes')

    with pytest.raises(InvalidFormat):
        p.get('x')

    with pytest.raises(InvalidFormat):
        p.get('x', recover=True)


def test_file_recovery_returns_valid_prefix(tmp_path):
    path = tmp_path / 'store'
    p = Checksum(str(path))
    p.append({'x': b'one'})
    p.append({'x': b'two'})
    p.append({'x': b'three'})

    filename = p.partd.filename('x')
    with open(filename, 'rb') as f:
        data = f.read()
    final_size = len(encode_frame(b'three'))
    with open(filename, 'wb') as f:
        f.write(data[:-(final_size // 2)])

    with pytest.raises(TruncatedFrame):
        p.get('x')
    assert p.get('x', recover=True) == b'onetwo'


def test_checksum_detects_corrupt_header_length(tmp_path):
    path = tmp_path / 'store'
    p = Checksum(str(path))
    p.append({'x': b'one'})

    filename = p.partd.filename('x')
    with open(filename, 'r+b') as f:
        f.seek(HEADER_DATA.size - 1)
        f.write(b'\xff')

    with pytest.raises((ChecksumMismatch, FrameTooLarge)):
        p.get('x')


def test_iset_uses_checksum_frame():
    with Checksum() as p:
        p.iset('x', b'123')
        raw = p.partd.get('x')
        assert raw.startswith(MAGIC)
        assert len(raw) == HEADER_SIZE + 3 + CHECKSUM_SIZE
        assert p.get('x') == b'123'
