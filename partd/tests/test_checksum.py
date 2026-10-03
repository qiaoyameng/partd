import os
import struct
import zlib

import pytest

from partd import Checksum, File, Pickle
from partd.checksum import (MAGIC, VERSION, HEADER_SIZE, CRC_SIZE,
                            FRAME_OVERHEAD, CorruptionError, FormatError,
                            FrameDecoder, pack_frame, unpack_frames)
from partd.encode import Encode
from partd.utils import frame as legacy_frame


payloads = [b'', b'a', b'Hello World!', b'x' * 1000, bytes(range(256))]


def stream_of(items):
    return b''.join(pack_frame(p) for p in items)


def test_frame_layout():
    frame = pack_frame(b'abc')
    assert frame[:4] == MAGIC
    assert frame[4] == VERSION
    assert struct.unpack('<Q', frame[5:13])[0] == 3
    assert frame[13:16] == b'abc'
    assert struct.unpack('<I', frame[16:])[0] == zlib.crc32(b'abc')
    assert len(frame) == FRAME_OVERHEAD + 3


def test_roundtrip():
    assert unpack_frames(stream_of(payloads)) == payloads
    assert unpack_frames(b'') == []


def test_arbitrary_block_boundaries():
    stream = stream_of(payloads)
    for block in [1, 2, 3, 5, 7, 13, 64, len(stream)]:
        decoder = FrameDecoder()
        out = []
        for i in range(0, len(stream), block):
            out.extend(decoder.feed(stream[i:i + block]))
        out.extend(decoder.finish())
        assert out == payloads


def test_declared_length_is_capped():
    # A corrupt header claiming an absurd length fails immediately
    bad = MAGIC + bytes([VERSION]) + struct.pack('<Q', 2 ** 40)
    with pytest.raises(CorruptionError, match='declares length'):
        unpack_frames(bad)

    # A legitimate frame above a configured limit is also refused
    frame = pack_frame(b'ten bytes!')
    with pytest.raises(CorruptionError, match='declares length'):
        unpack_frames(frame, max_frame_size=4)
    assert unpack_frames(frame, max_frame_size=10) == [b'ten bytes!']


def test_truncation_byte_by_byte():
    items = [b'aaa', b'bbbb', b'ccccc']
    stream = stream_of(items)
    boundaries = [len(stream_of(items[:i + 1])) for i in range(len(items))]

    for cut in range(len(stream) + 1):
        prefix = stream[:cut]
        complete = sum(cut >= b for b in boundaries)
        frame_start = boundaries[complete - 1] if complete else 0

        if cut == 0 or cut in boundaries:  # ends on a frame boundary: intact
            assert unpack_frames(prefix) == items[:complete]
        else:
            with pytest.raises(CorruptionError) as exc:
                unpack_frames(prefix)
            assert exc.value.offset == frame_start
            assert exc.value.frame_index == complete

        # recovery mode always yields exactly the intact prefix
        assert unpack_frames(prefix, recover=True) == items[:complete]


def test_truncation_at_decoder_level_byte_by_byte():
    items = [b'aaa', b'bbbb', b'ccccc']
    stream = stream_of(items)
    for cut in range(len(stream) + 1):
        prefix = stream[:cut]
        decoder = FrameDecoder(recover=True)
        out = decoder.feed(prefix)
        out.extend(decoder.finish())
        expected = sum(cut >= len(stream_of(items[:i + 1]))
                       for i in range(len(items)))
        assert out == items[:expected]


def test_mid_stream_bit_flip_never_recoverable():
    items = [b'aaa', b'bbbb', b'ccccc']
    stream = stream_of(items)
    first = len(pack_frame(items[0]))
    # flip a payload byte inside the middle frame
    pos = first + HEADER_SIZE + 1
    bad = stream[:pos] + bytes([stream[pos] ^ 0xFF]) + stream[pos + 1:]

    for recover in (False, True):
        with pytest.raises(CorruptionError) as exc:
            unpack_frames(bad, recover=recover)
        assert exc.value.offset == first
        assert exc.value.frame_index == 1

    # frames before the corruption are still delivered incrementally
    decoder = FrameDecoder(recover=True)
    assert decoder.feed(stream[:first]) == [b'aaa']
    with pytest.raises(CorruptionError):
        decoder.feed(bad[first:])


def test_bit_flip_in_last_frame_not_recoverable():
    stream = stream_of([b'aaa', b'bbbb'])
    pos = len(stream) - CRC_SIZE - 1  # last payload byte
    bad = stream[:pos] + bytes([stream[pos] ^ 0x01]) + stream[pos + 1:]
    for recover in (False, True):
        with pytest.raises(CorruptionError):
            unpack_frames(bad, recover=recover)


def test_bit_flip_in_crc_detected():
    stream = pack_frame(b'payload')
    pos = len(stream) - 1  # inside the crc trailer
    bad = stream[:pos] + bytes([stream[pos] ^ 0x01]) + stream[pos + 1:]
    with pytest.raises(CorruptionError, match='checksum mismatch'):
        unpack_frames(bad)


def test_legacy_data_rejected_by_marker():
    legacy = legacy_frame(b'hello') + legacy_frame(b'world')
    for recover in (False, True):
        with pytest.raises(FormatError, match='legacy'):
            unpack_frames(legacy, recover=recover)

    # also detected when fed in tiny pieces
    decoder = FrameDecoder()
    with pytest.raises(FormatError):
        for i in range(len(legacy)):
            decoder.feed(legacy[i:i + 1])


def test_unknown_version_rejected():
    frame = pack_frame(b'abc', version=VERSION + 1)
    with pytest.raises(FormatError, match='version'):
        unpack_frames(frame)


def test_partd_roundtrip():
    with Checksum() as p:
        p.append({'x': b'Hello', 'y': b'abc'})
        p.append({'x': b'World!', 'y': b'def'})
        assert p.get(['y', 'x']) == [b'abcdef', b'HelloWorld!']
        assert p.get('z') == b''


def test_partd_iset():
    with Checksum() as p:
        p.iset('x', b'123')
        p.iset('x', b'123')
        assert p.get('x') == b'123'


def test_partd_composable():
    with Pickle(Checksum()) as p:
        p.append({'x': [1, 2, 3], 'y': ['a']})
        p.append({'x': [4], 'y': ['b', 'c']})
        assert p.get('x') == [1, 2, 3, 4]
        assert p.get('y') == ['a', 'b', 'c']

    with Encode(zlib.compress, zlib.decompress, b''.join, Checksum()) as p:
        p.append({'x': b'Hello'})
        p.append({'x': b'World!'})
        assert p.get('x') == b'HelloWorld!'


def test_partd_detects_disk_corruption():
    with Checksum() as p:
        p.append({'x': b'Hello'})
        p.append({'x': b'World!'})
        fn = p.partd.filename('x')
        with open(fn, 'r+b') as f:
            f.seek(HEADER_SIZE + 2)  # inside the first frame's payload
            f.write(b'X')
        with pytest.raises(CorruptionError):
            p.get('x')


def test_partd_recovers_truncated_tail():
    with Checksum(recover=True) as p:
        p.append({'x': b'Hello'})
        p.append({'x': b'World!'})
        fn = p.partd.filename('x')
        size = os.path.getsize(fn)
        with open(fn, 'r+b') as f:
            f.truncate(size - 3)  # cut into the last frame
        assert p.get('x') == b'Hello'

    with Checksum() as p:  # strict mode reports the truncation
        p.append({'x': b'Hello'})
        p.append({'x': b'World!'})
        fn = p.partd.filename('x')
        size = os.path.getsize(fn)
        with open(fn, 'r+b') as f:
            f.truncate(size - 3)
        with pytest.raises(CorruptionError, match='truncated'):
            p.get('x')


def test_partd_rejects_legacy_file():
    with File() as f:
        f.append({'x': legacy_frame(b'hello')})
        p = Checksum(f)
        with pytest.raises(FormatError):
            p.get('x')
