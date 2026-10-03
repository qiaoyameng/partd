from partd.dict import Dict
from partd.file import File
from partd.buffer import Buffer, keys_to_flush
import pytest
import pickle

import shutil
import os


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


class BoundedHandle:
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.bytes_read = 0
        self.closed = False

    def read(self, size=-1):
        if size is None or size < 0:
            raise AssertionError("unbounded read")
        chunk = self.file.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True


class BoundedFile(File):
    def _open(self, filename):
        return BoundedHandle(filename)


def test_iter_chunks_fast_then_slow():
    fast = Dict()
    slow = Dict()
    with Buffer(fast, slow, available_memory=10) as p:
        fast.append({'x': b'ab'})
        slow.append({'x': b'cdef'})

        assert list(p.iter_chunks('x', chunksize=2)) == [
            b'ab', b'cd', b'ef']
        assert list(p.iter_chunks('missing', chunksize=2)) == []


def test_iter_chunks_uses_bounded_reads_and_closes_handles():
    with BoundedFile() as fast, BoundedFile() as slow:
        fast.append({'x': b'ab'})
        slow.append({'x': b'cdef'})

        with Buffer(fast, slow, available_memory=10) as p:
            chunks = p.iter_chunks('x', chunksize=2)
            fast_handle = chunks._readers[0]._file
            slow_handle = chunks._readers[1]._file

            assert next(chunks) == b'ab'
            assert next(chunks) == b'cd'
            assert fast_handle.closed
            assert not slow_handle.closed
            chunks.close()

        assert fast_handle.bytes_read == 2
        assert slow_handle.bytes_read == 2
        assert fast_handle.closed
        assert slow_handle.closed


def test_iter_chunks_snapshot_ignores_later_flush():
    fast = Dict()
    slow = Dict()
    with Buffer(fast, slow, available_memory=100) as p:
        fast.append({'x': b'ab'})
        chunks = p.iter_chunks('x', chunksize=2)

        p.append({'x': b'cd'})
        p.flush(['x'])

        assert list(chunks) == [b'ab']


def test_iter_chunks_concurrent_delete_finishes():
    fast = Dict()
    slow = Dict()
    with Buffer(fast, slow, available_memory=10) as p:
        fast.append({'x': b'ab'})
        slow.append({'x': b'cd'})
        chunks = p.iter_chunks('x', chunksize=2)

        assert next(chunks) == b'ab'
        p.delete('x')

        assert list(chunks) == [b'cd']


class TrackedDict(Dict):
    def __init__(self):
        super().__init__()
        self.chunk_reader = None

    def _iter_chunks(self, key, chunksize=1024 * 1024, lock=True):
        self.chunk_reader = super()._iter_chunks(
            key, chunksize=chunksize, lock=lock)
        return self.chunk_reader


class FailingDict(Dict):
    def _iter_chunks(self, key, chunksize=1024 * 1024, lock=True):
        raise OSError


def test_iter_chunks_closes_reader_when_layer_opening_fails():
    with TrackedDict() as fast, FailingDict() as slow:
        fast.append({'x': b'ab'})
        with Buffer(fast, slow, available_memory=10) as p:
            with pytest.raises(OSError):
                p.iter_chunks('x', chunksize=2)

        assert fast.chunk_reader._closed
