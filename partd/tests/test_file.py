import glob
import os
import shutil

from partd.file import File

import pytest


class InjectedFailure(RuntimeError):
    pass


class FaultFile(File):
    def __init__(self, path, fail_step=None):
        self.fail_step = fail_step
        self.steps = []
        super().__init__(path)
        self._explicitly_given_path = True

    def _transaction_step(self, step, function, *args):
        target = self.fail_step
        before = False
        if isinstance(target, str) and target.endswith(':before'):
            target = target[:-len(':before')]
            before = True

        should_fail = target(step) if callable(target) else step == target
        is_open_step = step.endswith('.open') or '.open' in step
        if should_fail and (before or is_open_step):
            raise InjectedFailure(step)

        result = function(*args)
        self.steps.append(step)
        if should_fail:
            raise InjectedFailure(step)
        return result


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


def test_transactional_append_is_opt_in():
    with File() as p:
        p.append({'x': b'1'}, transaction=True)
        assert p.get('x') == b'1'
        assert not glob.glob(os.path.join(p.path, '.~txn.*'))


def test_transactional_append_empty_batch():
    with File() as p:
        p.append({}, transaction=True)
        assert not glob.glob(os.path.join(p.path, '.~txn.*'))
        assert p.get('x') == b''


def test_transaction_recovery_ignores_user_file_with_journal_prefix(tmp_path):
    user_file = File(str(tmp_path))
    user_file.append({'.~txn.user.tmp': b'user-data'})

    recovered = File(str(tmp_path))
    assert recovered.get('.~txn.user.tmp') == b'user-data'


def test_transactional_append_commits_all_partitions():
    with File() as p:
        p.append({'x': b'Hello', 'y': b'abc'}, transaction=True)
        p.append({'x': b'World!', 'y': b'def'}, transaction=True)
        assert p.get(['x', 'y']) == [b'HelloWorld!', b'abcdef']
        assert not glob.glob(os.path.join(p.path, '.~txn.*'))


def test_transactional_append_preserves_existing_empty_file(tmp_path):
    source = File(str(tmp_path))
    source.append({'x': b''})
    inode_before = os.stat(source.filename('x')).st_ino

    faults = FaultFile(str(tmp_path),
                       fail_step='journal.commit.replace:before')
    with pytest.raises(InjectedFailure):
        faults.append({'x': b'new'}, transaction=True)

    recovered = File(str(tmp_path))
    assert recovered.get('x') == b''
    assert os.stat(recovered.filename('x')).st_ino == inode_before
    assert not glob.glob(os.path.join(str(tmp_path), '.~txn.*'))


def test_transactional_rollback_removes_new_nested_file(tmp_path):
    File(str(tmp_path))
    faults = FaultFile(str(tmp_path),
                       fail_step='journal.commit.replace:before')
    with pytest.raises(InjectedFailure):
        faults.append({('a', 'b'): b'new'}, transaction=True)

    recovered = File(str(tmp_path))
    assert recovered.get(('a', 'b')) == b''
    assert not os.path.exists(recovered.filename(('a', 'b')))
    assert not glob.glob(os.path.join(str(tmp_path), '.~txn.*'))


@pytest.mark.parametrize('fail_step', [
    'directory.create:a',
    'directory.sync.open',
    'directory.sync.fsync',
    'directory.sync.close',
    'directory.sync:a.open',
    'directory.sync:a.fsync',
    'directory.sync:a.close',
])
def test_transactional_nested_directory_crash(tmp_path, fail_step):
    _seed_transaction(str(tmp_path))
    faults = FaultFile(str(tmp_path), fail_step=fail_step)
    with pytest.raises(InjectedFailure):
        faults.append({('a', 'b'): b'new'}, transaction=True)

    recovered = File(str(tmp_path))
    assert recovered.get(['x', ('a', 'b')]) == [b'old', b'']
    assert not glob.glob(os.path.join(str(tmp_path), '.~txn.*'))


def _seed_transaction(path):
    p = File(path)
    p._explicitly_given_path = True
    p.append({'x': b'old'}, transaction=True)


APPEND_STEPS = [
    'journal.tmp.open',
    'journal.tmp.write',
    'journal.tmp.flush',
    'journal.tmp.fsync',
    'journal.tmp.close',
    'journal.tmp.directory.open',
    'journal.tmp.directory.fsync',
    'journal.tmp.directory.close',
    'journal.prepare.replace',
    'journal.prepare.directory.open',
    'journal.prepare.directory.fsync',
    'journal.prepare.directory.close',
    'file.open:x',
    'file.write:x',
    'file.flush:x',
    'file.fsync:x',
    'file.close:x',
    'file.open:y',
    'file.write:y',
    'file.flush:y',
    'file.fsync:y',
    'file.close:y',
    'file.directory:y.open',
    'file.directory:y.fsync',
    'file.directory:y.close',
    'journal.commit.replace',
    'journal.commit.directory.open',
    'journal.commit.directory.fsync',
    'journal.commit.directory.close',
    'journal.cleanup.remove',
    'journal.cleanup.directory.open',
    'journal.cleanup.directory.fsync',
    'journal.cleanup.directory.close',
]


@pytest.mark.parametrize('fail_step', APPEND_STEPS)
def test_transactional_append_crash_at_each_step(tmp_path, fail_step):
    path = tmp_path / 'data'
    _seed_transaction(str(path))
    faults = FaultFile(str(path), fail_step=fail_step)
    with pytest.raises(InjectedFailure):
        faults.append({'x': b'new', 'y': b'new-y'}, transaction=True)

    recovered = File(str(path))
    committed = APPEND_STEPS.index(fail_step) >= APPEND_STEPS.index(
        'journal.commit.replace')
    expected_x = b'oldnew' if committed else b'old'
    expected_y = b'new-y' if committed else b''
    assert recovered.get(['x', 'y']) == [expected_x, expected_y]
    assert not glob.glob(os.path.join(str(path), '.~txn.*'))


def _copy_path(source, destination):
    shutil.copytree(source, destination)


def _pending_recovery_steps(transaction_name):
    return [
        'recovery.file.open:x',
        'recovery.file.truncate:x',
        'recovery.file.flush:x',
        'recovery.file.fsync:x',
        'recovery.file.close:x',
        'recovery.file.remove:y',
        'recovery.file.directory:y.open',
        'recovery.file.directory:y.fsync',
        'recovery.file.directory:y.close',
        'recovery.journal.remove',
        f'recovery.journal.directory:.~txn.{transaction_name}.pending.open',
        f'recovery.journal.directory:.~txn.{transaction_name}.pending.fsync',
        f'recovery.journal.directory:.~txn.{transaction_name}.pending.close',
    ]


@pytest.mark.parametrize('fail_step_index', range(13))
def test_pending_transaction_recovery_can_be_interrupted(
        tmp_path, fail_step_index):
    source = tmp_path / 'pending'
    _seed_transaction(str(source))
    faults = FaultFile(str(source),
                       fail_step='journal.commit.replace:before')
    with pytest.raises(InjectedFailure):
        faults.append({'x': b'new', 'y': b'new-y'}, transaction=True)

    pending = glob.glob(os.path.join(str(source), '.~txn.*.pending'))
    assert len(pending) == 1
    transaction_name = os.path.basename(pending[0]).split('.')[2]
    fail_step = _pending_recovery_steps(transaction_name)[fail_step_index]

    interrupted = tmp_path / 'interrupted'
    _copy_path(str(source), str(interrupted))
    with pytest.raises(InjectedFailure):
        FaultFile(str(interrupted), fail_step=fail_step)

    recovered = File(str(interrupted))
    assert recovered.get(['x', 'y']) == [b'old', b'']
    assert not glob.glob(os.path.join(str(interrupted), '.~txn.*'))


def _committed_recovery_steps(transaction_name):
    return [
        f'recovery.committed.remove:.~txn.{transaction_name}.log',
        f'recovery.committed.directory:.~txn.{transaction_name}.log.open',
        f'recovery.committed.directory:.~txn.{transaction_name}.log.fsync',
        f'recovery.committed.directory:.~txn.{transaction_name}.log.close',
    ]


@pytest.mark.parametrize('fail_step_index', range(4))
def test_committed_transaction_recovery_can_be_interrupted(
        tmp_path, fail_step_index):
    source = tmp_path / 'committed'
    _seed_transaction(str(source))
    faults = FaultFile(str(source),
                       fail_step='journal.cleanup.remove:before')
    with pytest.raises(InjectedFailure):
        faults.append({'x': b'new', 'y': b'new-y'}, transaction=True)

    committed = glob.glob(os.path.join(str(source), '.~txn.*.log'))
    assert len(committed) == 1
    transaction_name = os.path.basename(committed[0]).split('.')[2]
    fail_step = _committed_recovery_steps(transaction_name)[fail_step_index]

    interrupted = tmp_path / 'interrupted'
    _copy_path(str(source), str(interrupted))
    with pytest.raises(InjectedFailure):
        FaultFile(str(interrupted), fail_step=fail_step)

    recovered = File(str(interrupted))
    assert recovered.get(['x', 'y']) == [b'oldnew', b'new-y']
    assert not glob.glob(os.path.join(str(interrupted), '.~txn.*'))


@pytest.mark.parametrize('fail_step', [
    'recovery.tmp.remove',
    'recovery.tmp.directory.open',
    'recovery.tmp.directory.fsync',
    'recovery.tmp.directory.close',
])
def test_temporary_transaction_recovery_can_be_interrupted(
        tmp_path, fail_step):
    source = tmp_path / 'temporary'
    _seed_transaction(str(source))
    faults = FaultFile(str(source),
                       fail_step='journal.prepare.replace:before')
    with pytest.raises(InjectedFailure):
        faults.append({'x': b'new', 'y': b'new-y'}, transaction=True)
    assert glob.glob(os.path.join(str(source), '.~txn.*.tmp'))

    interrupted = tmp_path / 'interrupted'
    _copy_path(str(source), str(interrupted))
    with pytest.raises(InjectedFailure):
        FaultFile(str(interrupted), fail_step=fail_step)

    recovered = File(str(interrupted))
    assert recovered.get(['x', 'y']) == [b'old', b'']
    assert not glob.glob(os.path.join(str(interrupted), '.~txn.*'))
