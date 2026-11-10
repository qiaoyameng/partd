import os
import pickle
import struct

import pytest

from partd.adaptive import (Adaptive, AdaptiveError, HEADER_SIZE, MAGIC,
                            VERSION, CODEC_RAW, CODEC_ZLIB, CODEC_BZ2,
                            encode_frame, read_frames)
from partd.buffer import Buffer
from partd.dict import Dict
from partd.encode import Encode
from partd.file import File


compressible = (b'the quick brown fox jumps over the lazy dog ' * 300)
incompressible = os.urandom(5000)
tiny = b'hi'


def frame_codes(path, key):
    with open(path, 'rb') as f:
        data = f.read()
    return [(index, codec, raw_length, len(payload))
            for index, codec, raw_length, payload in read_frames(data, key)]


def test_mixed_frames_roundtrip():
    with Adaptive() as p:
        p.append({'x': compressible})
        p.append({'x': incompressible})
        p.append({'x': tiny})
        p.append({'x': compressible})
        p.append({'x': tiny})

        frames = frame_codes(p.partd.filename('x'), 'x')
        codecs = [codec for _, codec, _, _ in frames]
        assert codecs[0] in (CODEC_ZLIB, CODEC_BZ2)   # compressible
        assert codecs[1] == CODEC_RAW                  # incompressible
        assert codecs[2] == CODEC_RAW                  # tiny
        assert codecs[3] in (CODEC_ZLIB, CODEC_BZ2)
        assert codecs[4] == CODEC_RAW

        # compressed frame is actually smaller than its input
        _, _, raw_length, payload_length = frames[0]
        assert raw_length == len(compressible)
        assert payload_length < len(compressible) // 2

        # raw frames are stored verbatim
        assert frames[1][2] == frames[1][3] == len(incompressible)
        assert frames[2][2] == frames[2][3] == len(tiny)

        expected = compressible + incompressible + tiny + compressible + tiny
        assert p.get('x') == expected
        assert p.get('missing') == b''


def test_stored_size_smaller_than_logical():
    with Adaptive() as p:
        p.append({'x': compressible})
        stored = os.path.getsize(p.partd.filename('x'))
        assert stored < len(compressible)
        assert p.get('x') == compressible


def test_policy_thresholds():
    # blocks below min_block_size stay raw even if compressible
    with Adaptive(min_block_size=len(compressible) + 1) as p:
        p.append({'x': compressible})
        assert frame_codes(p.partd.filename('x'), 'x')[0][1] == CODEC_RAW
        assert p.get('x') == compressible

    # an impossible savings requirement disables compression
    with Adaptive(min_savings=0.999) as p:
        p.append({'x': compressible})
        assert frame_codes(p.partd.filename('x'), 'x')[0][1] == CODEC_RAW
        assert p.get('x') == compressible

    # zero savings accepts any non-expanding codec
    with Adaptive(min_savings=0.0) as p:
        p.append({'x': compressible})
        assert frame_codes(p.partd.filename('x'), 'x')[0][1] != CODEC_RAW
        assert p.get('x') == compressible

    with pytest.raises(ValueError):
        Adaptive(min_savings=1.0)
    with pytest.raises(ValueError):
        Adaptive(min_block_size=-1)


def corrupt_header(path, frame_offset, field_offset, fmt, value):
    with open(path, 'r+b') as f:
        f.seek(frame_offset + field_offset)
        f.write(struct.pack(fmt, value))


def test_corrupt_declared_length_locates_partition_and_frame():
    with Adaptive() as p:
        p.append({'x': tiny})           # frame 0, raw
        p.append({'x': compressible})   # frame 1
        path = p.partd.filename('x')

        second_frame_offset = HEADER_SIZE + len(tiny)
        # raw_length is a Q at offset 4 within the header
        corrupt_header(path, second_frame_offset, 4, '<Q',
                       len(compressible) + 1)

        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        message = str(excinfo.value)
        assert "'x'" in message
        assert 'frame 1' in message
        assert 'does not match declared length' in message


def test_unknown_codec_and_version():
    with Adaptive() as p:
        p.append({'x': compressible})
        path = p.partd.filename('x')

        corrupt_header(path, 0, 3, '<B', 99)  # codec byte
        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        assert 'unknown codec 99' in str(excinfo.value)
        assert "'x'" in str(excinfo.value)

        corrupt_header(path, 0, 3, '<B', CODEC_ZLIB)  # restore
        corrupt_header(path, 0, 2, '<B', VERSION + 1)  # version byte
        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        assert 'unsupported version' in str(excinfo.value)


def test_max_frame_size_enforced_during_decompression():
    with Adaptive() as p:
        p.append({'x': compressible})
        path = p.partd.filename('x')
        # lie about the declared length: claim something enormous
        corrupt_header(path, 0, 4, '<Q', 1 << 40)

        p.max_frame_size = 1000
        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        assert 'exceeds limit' in str(excinfo.value)
        assert 'frame 0' in str(excinfo.value)

        # with a generous cap the lie surfaces as a length mismatch
        p.max_frame_size = 1 << 31
        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        assert 'does not match declared length' in str(excinfo.value)


def test_truncated_payload():
    with Adaptive() as p:
        p.append({'x': compressible})
        path = p.partd.filename('x')
        with open(path, 'r+b') as f:
            f.truncate(HEADER_SIZE + 10)
        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        assert 'truncated payload' in str(excinfo.value)


def test_legacy_mode_explicit_only():
    with Adaptive() as p:
        p.partd.append({'x': b'old unmarked data'})  # bypass framing
        with pytest.raises(AdaptiveError) as excinfo:
            p.get('x')
        assert 'legacy=True' in str(excinfo.value)

        q = Adaptive(p.partd, legacy=True)
        assert q.get('x') == b'old unmarked data'


def test_buffer_composition():
    # Buffer sits inside Adaptive: its byte statistics track the framed
    # payloads that actually flow through it, and frames survive the
    # flush from fast to slow storage intact and in order.
    with Adaptive(Buffer(Dict(), File(), available_memory=2000)) as p:
        blocks = [compressible, incompressible, tiny,
                      compressible, incompressible, tiny]
        for block in blocks:
            p.append({'x': block})

        inner = p.partd
        framed_total = sum(len(encode_frame(b, p.min_block_size,
                                            p.min_savings))
                           for b in blocks)
        slow_bytes = os.path.getsize(inner.slow.filename('x'))
        assert sum(inner.lengths.values()) == inner.memory_usage
        # every framed byte is accounted for: cached in fast or on disk
        assert inner.memory_usage + slow_bytes == framed_total

        inner.flush()
        assert inner.memory_usage == 0
        assert not inner.lengths
        assert p.get('x') == b''.join(blocks)


def test_encode_composition():
    # Encode frames its values before Adaptive sees them; decoding must
    # peel the layers in the opposite order without confusion.
    with Encode(lambda s: s.encode('utf-8'),
                lambda b: b.decode('utf-8'),
                lambda parts: ''.join(parts),
                partd=Adaptive()) as p:
        p.append({'x': 'hello ' * 100})
        p.append({'x': 'world ' * 100})
        p.append({'x': '!'})
        assert p.get('x') == 'hello ' * 100 + 'world ' * 100 + '!'


def test_iset_and_pickle():
    with Adaptive() as p:
        p.iset('x', compressible)
        p.iset('x', compressible)
        assert p.get('x') == compressible

        q = pickle.loads(pickle.dumps(p))
        assert q.get('x') == compressible
