from partd.dict import Dict
from partd.file import File
from partd.buffer import Buffer, keys_to_flush
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


class BoundedReadFile:
    """ File double that refuses unbounded reads and tracks its usage """
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.closed = False
        self.read_sizes = []

    def read(self, n=-1):
        if n is None or n < 0:
            raise AssertionError("unbounded read")
        self.read_sizes.append(n)
        chunk = self.data[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk

    def close(self):
        self.closed = True


def test_iter_chunks_fast_then_slow():
    with Dict() as a:
        with File() as b:
            with Buffer(a, b, available_memory=1000) as p:
                p.append({'x': b'NEW'})      # stays in the fast layer
                b.append({'x': b'old-data'})  # directly in the slow layer

                chunks = list(p.iter_chunks('x', chunk_size=2))
                assert chunks == [b'NE', b'W', b'ol', b'd-', b'da', b'ta']
                assert b''.join(chunks) == p.get('x')

                assert list(p.iter_chunks('z')) == []


def test_iter_chunks_snapshot_boundary():
    with Dict() as a:
        with File() as b:
            with Buffer(a, b, available_memory=1000) as p:
                p.append({'x': b'abc'})
                it = p.iter_chunks('x', chunk_size=2)
                p.append({'x': b'def'})  # not part of the ongoing iteration
                assert b''.join(it) == b'abc'
                assert p.get('x') == b'abcdef'


def test_iter_chunks_early_close_releases_handle(monkeypatch):
    with Dict() as a:
        with File() as b:
            with Buffer(a, b, available_memory=1000) as p:
                p.append({'x': b'NE'})        # one chunk in the fast layer
                b.append({'x': b'old-data'})  # directly in the slow layer
                with open(b.filename('x'), 'rb') as f:
                    content = f.read()
                handle = BoundedReadFile(content)
                monkeypatch.setattr('partd.file.open',
                                    lambda path, mode: handle, raising=False)

                it = p.iter_chunks('x', chunk_size=2)
                assert next(it) == b'NE'  # fast layer
                assert next(it) == b'ol'  # slow layer, first bounded read
                it.close()

                assert handle.read_sizes == [2]
                assert handle.pos == 2
                assert handle.closed
