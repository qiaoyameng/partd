from partd.file import File

import pytest
import shutil
import os


def test_partd():
    with File() as p:
        p.append({'x': b'Hello', 'y': b'abc'})
        p.append({'x': b'World!', 'y': b'def'})
        assert os.path.exists(p.filename('x'))
        assert os.path.exists(p.filename('y'))

        result = p.get(['y', 'x'])
        assert result == [b'abcdef', b'HelloWorld!']

        assert p.get('z') == b''

        with p.lock:  # uh oh, possible deadlock
            result = p.get(['x'], lock=False)

    assert not os.path.exists(p.path)


def test_key_tuple():
    with File() as p:
        p.append({('a', 'b'): b'123'})
        assert os.path.exists(p.filename(('a', 'b')))


def test_iset():
    with File() as p:
        p.iset('x', b'123')
        assert 'x' in p._iset_seen
        assert 'y' not in p._iset_seen
        p.iset('x', b'123')
        p.iset('x', b'123')
        assert p.get('x') == b'123'


def test_nested_get():
    with File() as p:
        p.append({'x': b'1', 'y': b'2', 'z': b'3'})
        assert p.get(['x', ['y', 'z']]) == [b'1', [b'2', b'3']]


def test_drop():
    with File() as p:
        p.append({'x': b'123'})
        p.iset('y', b'abc')
        assert p.get('x') == b'123'
        assert p.get('y') == b'abc'

        p.drop()
        assert p.get('x') == b''
        assert p.get('y') == b''

        p.append({'x': b'123'})
        p.iset('y', b'def')
        assert p.get('x') == b'123'
        assert p.get('y') == b'def'


def test_del():
    f = File()

    assert f.path
    assert os.path.exists(f.path)

    f.__del__()
    assert not os.path.exists(f.path)

    with File('Foo') as p:
        p.__del__()
        assert os.path.exists(p.path)


def test_specify_dirname():
    with File(dir=os.getcwd()) as f:
        assert os.getcwd() in f.path


def test_iter_chunks():
    with File() as p:
        p.append({'x': b'abcde'})
        iterator = p.iter_chunks('x', chunksize=2)
        assert next(iterator) == b'ab'

        p.append({'x': b'fgh'})

        assert list(iterator) == [b'cd', b'e']
        assert b''.join(p.iter_chunks('x', chunksize=2)) == b'abcdefgh'
        assert list(p.iter_chunks('missing', chunksize=2)) == []


def test_iter_chunks_invalid_chunksize():
    with File() as p:
        for chunksize in (0, -1):
            with pytest.raises(ValueError):
                p.iter_chunks('x', chunksize=chunksize)
        with pytest.raises(TypeError):
            p.iter_chunks('x', chunksize=1.5)


def test_iter_chunks_closes_after_error():
    class FailingFile(File):
        def _open(self, filename):
            return FailingHandle()

    class FailingHandle:
        closed = False

        def read(self, size=-1):
            raise OSError

        def close(self):
            self.closed = True

    with FailingFile() as p:
        p.append({'x': b'abcde'})
        iterator = p.iter_chunks('x', chunksize=2)
        handle = iterator._file

        with pytest.raises(OSError):
            next(iterator)

        assert handle.closed


def test_iter_chunks_concurrent_delete_finishes():
    with File() as p:
        p.append({'x': b'abcdef'})
        iterator = p.iter_chunks('x', chunksize=2)

        assert next(iterator) == b'ab'
        p.delete('x')

        assert list(iterator) == [b'cd', b'ef']
