from partd.dict import Dict
from partd.file import File
from partd.buffer import Buffer, BufferErrorState, keys_to_flush
from partd.pickle import Pickle
from partd.encode import Encode
import pickle
import zlib

import shutil
import os
import tempfile

import pytest


def test_partd():
    a = Dict()
    b = Dict()
    with Buffer(a, b, available_memory=10) as p:
        p.append({'x': b'Hello', 'y': b'abc'})
        assert a.get(['x', 'y']) == [b'Hello', b'abc']

        p.append({'x': b'World!', 'y': b'def'})
        assert a.get(['x', 'y']) == [b'', b'abcdef']
        assert b.get(['x', 'y']) == [b'HelloWorld!', b'']

        result = p.get(['y', 'x'])
        assert result == [b'abcdef', b'HelloWorld!']

        assert p.get('z') == b''

        with p.lock:  # uh oh, possible deadlock
            result = p.get(['x'], lock=False)


def test_keys_to_flush():
    lengths = {'a': 20, 'b': 10, 'c': 15, 'd': 15, 'e': 10, 'f': 25, 'g': 5}
    assert keys_to_flush(lengths, 0.5) == ['f', 'a']


def test_pickle():
    with Dict() as a:
        with File() as b:
            c = Buffer(a, b)

            c.append({'x': b'123'})

            d = pickle.loads(pickle.dumps(c))

            assert d.get('x') == c.get('x')

            pickled_attrs = ('memory_usage', 'lengths', 'available_memory')
            for attr in pickled_attrs:
                assert hasattr(d, attr)
                assert getattr(d, attr) == getattr(c, attr)
            # special case Dict and File -- some attrs do not pickle
            assert hasattr(d, 'fast')
            assert d.fast.data == c.fast.data
            assert hasattr(d, 'slow')
            assert d.slow.path == c.slow.path


def test_ordering_across_flush_dict():
    # New data lives in fast, old data in slow: get must keep global order.
    with Buffer(Dict(), Dict(), available_memory=10) as p:
        p.append({'x': b'Hello', 'y': b'abc'})
        p.flush()
        p.append({'x': b'World!', 'y': b'def'})

        assert p.fast.get(['x', 'y']) == [b'World!', b'def']
        assert p.slow.get(['x', 'y']) == [b'Hello', b'abc']
        assert p.get(['x', 'y']) == [b'HelloWorld!', b'abcdef']
        assert dict(p.lengths) == {'x': 6, 'y': 3}
        assert p.memory_usage == 9


def test_ordering_across_flush_file():
    with tempfile.TemporaryDirectory() as tmp:
        with Buffer(Dict(), File(os.path.join(tmp, 'slow')),
                    available_memory=10) as p:
            p.append({'x': b'Hello'})
            p.flush()
            p.append({'x': b'World!'})
            assert p.get('x') == b'HelloWorld!'


def test_repeated_append_flush_cycles():
    with Buffer(Dict(), Dict(), available_memory=1) as p:
        expected = b''
        for i in range(5):
            chunk = ('%d' % i).encode()
            p.append({'k': chunk})
            expected += chunk
            p.flush()
        assert p.get('k') == expected
        assert p.memory_usage == 0
        assert dict(p.lengths) == {}


def test_duplicate_key_within_one_request():
    with Buffer(Dict(), Dict(), available_memory=1000) as p:
        p.append({'k': b'AA'})
        p.flush()
        p.append({'k': b'BB'})
        p.append({'k': b'CC'})
        assert p.get('k') == b'AABBCC'
        assert p.lengths['k'] == 4
        assert p.memory_usage == 4


def test_partial_key_flush_and_bookkeeping():
    with Buffer(Dict(), Dict(), available_memory=1000) as p:
        p.append({'a': b'1', 'b': b'22', 'c': b'333'})
        p.flush(['a', 'b'])

        assert p.fast.get(['a', 'b', 'c']) == [b'', b'', b'333']
        assert p.slow.get(['a', 'b', 'c']) == [b'1', b'22', b'']
        assert dict(p.lengths) == {'c': 3}
        assert p.memory_usage == 3

        p.append({'a': b'4'})
        p.flush()
        assert p.get(['a', 'b', 'c']) == [b'14', b'22', b'333']
        assert dict(p.lengths) == {}
        assert p.memory_usage == 0


def test_flush_unknown_key_is_noop():
    with Buffer(Dict(), Dict()) as p:
        p.append({'a': b'1'})
        p.flush(['a', 'missing'])
        assert p.get('a') == b'1'


def test_pickle_ordering_across_flush():
    with tempfile.TemporaryDirectory() as tmp:
        fast = Pickle(Dict())
        slow = Pickle(File(os.path.join(tmp, 'slow')))
        with Buffer(fast, slow, available_memory=10) as p:
            p.append({'k': [1, 2]})
            p.flush()
            p.append({'k': [3, 4]})
            assert p.get('k') == [1, 2, 3, 4]


def test_encode_ordering_across_flush():
    with tempfile.TemporaryDirectory() as tmp:
        fast = Encode(zlib.compress, zlib.decompress, b''.join, Dict())
        slow = Encode(zlib.compress, zlib.decompress, b''.join,
                      File(os.path.join(tmp, 'slow')))
        with Buffer(fast, slow, available_memory=10) as p:
            p.append({'k': b'hello-'})
            p.flush()
            p.append({'k': b'world'})
            assert p.get('k') == b'hello-world'


class ExplodingDict(Dict):
    def __init__(self, partial=False):
        super().__init__()
        self.partial = partial

    def append(self, data, lock=True, **kwargs):
        if self.partial:
            first_key = next(iter(data))
            super().append({first_key: data[first_key]}, lock=lock)
        raise RuntimeError('injected fast.append failure')


class ExplodingFile(File):
    def __init__(self, path, fail_after=1):
        super().__init__(path=path)
        self.fail_after = fail_after

    def append(self, data, lock=True, **kwargs):
        items = list(data.items())
        super().append(dict(items[:self.fail_after]), lock=lock)
        raise RuntimeError('injected slow.append failure')


def test_fast_append_failure_leaves_no_phantom_state():
    fast, slow = ExplodingDict(), Dict()
    with Buffer(fast, slow, available_memory=1000) as p:
        with pytest.raises(RuntimeError):
            p.append({'k': b'AA', 'j': b'BB'})

        assert dict(p.lengths) == {}
        assert p.memory_usage == 0
        assert fast.data == {}
        assert slow.data == {}

        with pytest.raises(BufferErrorState):
            p.append({'k': b'CC'})
        with pytest.raises(BufferErrorState):
            p.flush()
        assert p.get(['k', 'j']) == [b'', b'']


def test_slow_partial_write_blocks_blind_retry():
    with tempfile.TemporaryDirectory() as tmp:
        fast = Dict()
        slow = ExplodingFile(os.path.join(tmp, 'slow'))
        with Buffer(fast, slow, available_memory=1000) as p:
            p.append({'a': b'11', 'b': b'22', 'c': b'33'})
            with pytest.raises(RuntimeError):
                p.flush()

            # fast copies and bookkeeping survive; slow keeps the partial write
            assert fast.get(['a', 'b', 'c']) == [b'11', b'22', b'33']
            assert slow.get(['a', 'b', 'c']) == [b'11', b'', b'']
            assert dict(p.lengths) == {'a': 2, 'b': 2, 'c': 2}
            assert p.memory_usage == 6

            # a blind retry would duplicate b'11' on slow: refuse it
            with pytest.raises(BufferErrorState):
                p.flush()
            with pytest.raises(BufferErrorState):
                p.append({'a': b'zz'})


def test_fast_delete_failure_marks_error():
    class FailDeleteDict(Dict):
        def _delete(self, keys, lock=True):
            raise RuntimeError('injected fast.delete failure')

    fast, slow = FailDeleteDict(), Dict()
    with Buffer(fast, slow, available_memory=1000) as p:
        p.append({'k': b'AA'})
        with pytest.raises(RuntimeError):
            p.flush()
        assert slow.get('k') == b'AA'
        with pytest.raises(BufferErrorState):
            p.append({'k': b'BB'})


def test_auto_flush_failure_during_append_poisons_buffer():
    with tempfile.TemporaryDirectory() as tmp:
        fast = Dict()
        slow = ExplodingFile(os.path.join(tmp, 'slow'))
        with Buffer(fast, slow, available_memory=4) as p:
            p.append({'a': b'11'})
            with pytest.raises(RuntimeError):
                p.append({'b': b'22', 'c': b'33'})
            assert p._error is not None
            assert dict(p.lengths) == {'a': 2, 'b': 2, 'c': 2}
            assert p.memory_usage == 6
            with pytest.raises(BufferErrorState):
                p.flush()
