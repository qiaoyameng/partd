from partd.file import File, TXN_JOURNAL_NAME, TXN_COMMIT_MARKER

import shutil
import os

import pytest


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


def journal_path(p):
    return os.path.join(p.path, TXN_JOURNAL_NAME)


def test_append_transactional(tmp_path):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'x': b'Hello', 'y': b'abc'})

    p.append_transactional({'x': b'1', 'y': b'23', 'z': b'new'})

    assert p.get(['x', 'y', 'z']) == [b'Hello1', b'abc23', b'new']
    assert not os.path.exists(journal_path(p))


def test_append_transactional_empty(tmp_path):
    p = File(path=str(tmp_path / 'db'))
    p.append_transactional({})
    assert not os.path.exists(journal_path(p))
    assert os.listdir(p.path) == ['.lock']


def test_append_transactional_no_fsync(tmp_path):
    p = File(path=str(tmp_path / 'db'))
    p.append_transactional({'x': b'1'}, fsync=False)
    assert p.get('x') == b'1'
    assert not os.path.exists(journal_path(p))


def test_transactional_crash_before_journal(tmp_path, monkeypatch):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'x': b'Hello'})

    monkeypatch.setattr(File, '_write_txn_journal',
                        lambda self, entries: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        p.append_transactional({'x': b'!', 'y': b'new'})

    monkeypatch.undo()
    q = File(path=path)
    assert q.get(['x', 'y']) == [b'Hello', b'']
    assert not os.path.exists(journal_path(q))


def test_transactional_crash_mid_append(tmp_path, monkeypatch):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'a': b'AA', 'b': b'BB'})

    calls = []
    original = File._append_txn_file

    def flaky(self, fn, value, fsync):
        calls.append(fn)
        if len(calls) == 2:
            raise RuntimeError('simulated crash')
        return original(self, fn, value, fsync)

    monkeypatch.setattr(File, '_append_txn_file', flaky)
    with pytest.raises(RuntimeError):
        p.append_transactional({'a': b'1', 'b': b'2', 'c': b'3'})

    # first partition picked up an uncommitted tail on disk
    assert p.get('a', lock=False) == b'AA1'

    monkeypatch.undo()
    q = File(path=path)
    # only the batch's tail is truncated; earlier data is untouched
    assert q.get(['a', 'b', 'c']) == [b'AA', b'BB', b'']
    assert not os.path.exists(journal_path(q))


def test_transactional_crash_before_commit(tmp_path, monkeypatch):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'a': b'AA'})

    monkeypatch.setattr(File, '_commit_txn_journal',
                        lambda self: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        p.append_transactional({'a': b'tail', 'b': b'new'})

    monkeypatch.undo()
    q = File(path=path)
    # data was fsynced but never committed: whole batch is rolled back
    assert q.get(['a', 'b']) == [b'AA', b'']
    assert not os.path.exists(journal_path(q))


def test_transactional_crash_after_commit(tmp_path, monkeypatch):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'a': b'AA'})

    monkeypatch.setattr(File, '_remove_txn_journal',
                        lambda self: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        p.append_transactional({'a': b'tail', 'b': b'new'})

    with open(journal_path(p)) as f:
        assert f.read().splitlines()[-1] == TXN_COMMIT_MARKER

    monkeypatch.undo()
    q = File(path=path)
    # committed batch survives; stale journal is cleaned up
    assert q.get(['a', 'b']) == [b'AAtail', b'new']
    assert not os.path.exists(journal_path(q))


def test_transactional_rollback_removes_new_files(tmp_path, monkeypatch):
    path = str(tmp_path / 'db')
    p = File(path=path)

    monkeypatch.setattr(File, '_commit_txn_journal',
                        lambda self: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        p.append_transactional({'fresh': b'data', ('nest', 'ed'): b'x'})

    monkeypatch.undo()
    q = File(path=path)
    # files first created by the aborted batch disappear entirely
    assert not os.path.exists(q.filename('fresh'))
    assert not os.path.exists(q.filename(('nest', 'ed')))
    assert q.get(['fresh']) == [b'']


def test_transactional_recovery_interrupted(tmp_path, monkeypatch):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'a': b'AA', 'b': b'BB'})

    monkeypatch.setattr(File, '_commit_txn_journal',
                        lambda self: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        p.append_transactional({'a': b'1', 'b': b'2'})
    monkeypatch.undo()

    # first reopen dies in the middle of rolling back
    calls = []
    original = File._rollback_txn_entry

    def flaky(self, entry):
        calls.append(entry)
        if calls:
            raise RuntimeError('crash during recovery')
        return original(self, entry)

    monkeypatch.setattr(File, '_rollback_txn_entry', flaky)
    with pytest.raises(RuntimeError):
        File(path=path)
    monkeypatch.undo()

    # journal survived, so the next open finishes the (idempotent) rollback
    assert os.path.exists(journal_path(p))
    q = File(path=path)
    assert q.get(['a', 'b']) == [b'AA', b'BB']
    assert not os.path.exists(journal_path(q))


def test_transactional_corrupt_journal_is_discarded(tmp_path):
    path = str(tmp_path / 'db')
    p = File(path=path)
    p.append({'a': b'AA'})
    # process died while writing the journal itself: no data was appended
    with open(journal_path(p), 'w') as f:
        f.write('{"entries": [')

    q = File(path=path)
    assert q.get('a') == b'AA'
    assert not os.path.exists(journal_path(q))
