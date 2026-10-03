import os
import shutil

import pytest

from partd import Dict, File, Buffer, Quota, QuotaExceeded


# --------------------------------------------------------------------------
# basic quota enforcement: exactly fits, one byte over, multi-key batches
# --------------------------------------------------------------------------

def test_append_exactly_fills_global_quota(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        q.append({'x': b'abcd', 'y': b'efgh'})
        assert q.used() == 8
        assert q.used('x') == 4

        q.append({'x': b'ij'})
        assert q.used('x') == 6
        assert q.used() == 10

        with pytest.raises(QuotaExceeded):
            q.append({'z': b'k'})
        assert q.used() == 10
        assert q.get('z') == b''


def test_append_one_byte_over_rejects_whole_batch(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        q.append({'x': b'a' * 9})
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'ab'})
        # nothing from the rejected batch was written
        assert q.get('y') == b''
        assert q.used() == 9


def test_per_partition_quota(tmp_path):
    with Quota(File(str(tmp_path / 'p')),
               per_partition=5, global_bytes=100) as q:
        q.append({'x': b'ab'})
        q.append({'x': b'cde'})
        assert q.get('x') == b'abcde'
        with pytest.raises(QuotaExceeded):
            q.append({'x': b'f'})
        assert q.get('x') == b'abcde'
        assert q.used('x') == 5


def test_multi_key_batch_is_all_or_nothing(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        with pytest.raises(QuotaExceeded):
            q.append({'a': b'12345', 'b': b'123456'})
        assert q.get('a') == b''
        assert q.get('b') == b''
        assert q.used() == 0
        assert not os.path.exists(q.partd.filename('a'))
        assert not os.path.exists(q.partd.filename('b'))


def test_multi_key_batch_per_partition_all_or_nothing(tmp_path):
    with Quota(File(str(tmp_path / 'p')),
               per_partition=4, global_bytes=1000) as q:
        with pytest.raises(QuotaExceeded):
            q.append({'a': b'12', 'b': b'12345'})
        assert q.get('a') == b''
        assert q.get('b') == b''


def test_empty_payload_is_allowed(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=0) as q:
        q.append({'x': b''})
        assert q.used('x') == 0
        assert q.used() == 0
        assert q.get('x') == b''


def test_empty_batch_is_noop(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=0) as q:
        q.append({})
        assert q.used() == 0


def test_negative_quota_rejected():
    with pytest.raises(ValueError):
        Quota(Dict(), per_partition=-1)
    with pytest.raises(ValueError):
        Quota(Dict(), global_bytes=-1)


# --------------------------------------------------------------------------
# flush from fast to slow must not double-charge / release bytes
# --------------------------------------------------------------------------

def test_flush_does_not_change_accounting():
    with Dict() as fast, File() as slow:
        buf = Buffer(fast, slow, available_memory=1000)
        with Quota(buf, per_partition=100, global_bytes=20) as q:
            q.append({'x': b'Hello', 'y': b'abc'})
            assert q.used() == 8

            q.append({'x': b'World!', 'y': b'def'})
            assert q.used() == 17
            assert q.used('x') == 11
            assert q.used('y') == 6

            # force everything from fast to slow
            q.flush()
            assert fast.get(['x', 'y']) == [b'', b'']
            assert slow.get(['x', 'y']) == [b'HelloWorld!', b'abcdef']

            # accounting is unchanged, and flushed bytes still count
            assert q.used() == 17
            assert q.used('x') == 11
            assert q.used('y') == 6
            assert q.get(['y', 'x']) == [b'abcdef', b'HelloWorld!']

            with pytest.raises(QuotaExceeded):
                q.append({'z': b'aaaa'})
            assert q.get('z') == b''


def test_automatic_flush_does_not_change_accounting():
    with Dict() as fast, File() as slow:
        buf = Buffer(fast, slow, available_memory=10)
        with Quota(buf, per_partition=100, global_bytes=20) as q:
            q.append({'x': b'Hello', 'y': b'abc'})
            q.append({'x': b'World!', 'y': b'def'})
            # the second append pushed memory over the limit, triggering an
            # automatic fast->slow flush that must stay invisible to quota
            assert buf.memory_usage <= 10
            assert q.used() == 17
            assert q.used('x') == 11
            assert q.used('y') == 6
            assert q.get(['y', 'x']) == [b'abcdef', b'HelloWorld!']


def test_flush_then_fill_exactly():
    with Dict() as fast, File() as slow:
        with Quota(Buffer(fast, slow, available_memory=5),
                   global_bytes=10) as q:
            q.append({'x': b'123456'})
            q.flush()
            q.append({'y': b'1234'})
            assert q.used() == 10
            with pytest.raises(QuotaExceeded):
                q.append({'z': b'1'})


# --------------------------------------------------------------------------
# deletes free space only after success; deleting a missing key is a no-op
# --------------------------------------------------------------------------

def test_delete_frees_quota(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        q.append({'x': b'abcdef', 'y': b'ghij'})
        assert q.used() == 10
        with pytest.raises(QuotaExceeded):
            q.append({'z': b'k'})

        q.delete('x')
        assert q.used() == 4
        assert q.used('x') == 0
        assert q.get('x') == b''
        q.append({'z': b'k' * 6})
        assert q.used() == 10


def test_delete_missing_key_is_noop(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        q.append({'x': b'abcd'})
        q.delete(['missing', 'also-missing'])
        assert q.used() == 4
        # repeated deletes (retries) stay consistent
        q.delete('missing')
        q.delete(['x'])
        q.delete('x')
        assert q.used() == 0


def test_delete_multiple_keys_frees_all(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        q.append({'a': b'1234', 'b': b'5678'})
        q.delete(['a', 'b', 'missing'])
        assert q.used() == 0
        assert q.get(['a', 'b']) == [b'', b'']
        q.append({'c': b'0123456789'})
        assert q.used() == 10


def test_failed_delete_does_not_release_quota(tmp_path):
    class FailingFile(File):
        def _delete(self, keys, lock=True):
            raise IOError("disk exploded")

    with FailingFile(str(tmp_path / 'p')) as f:
        q = Quota(f, global_bytes=10)
        q.append({'x': b'abcdef'})
        assert q.used() == 6
        with pytest.raises(IOError):
            q.delete('x')
        # failed delete must not invent free space
        assert q.used() == 6
        assert q.used('x') == 6
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'12345'})
        assert q.get('x') == b'abcdef'


# --------------------------------------------------------------------------
# overwrites free old bytes before charging new ones
# --------------------------------------------------------------------------

def test_iset_overwrite_frees_old_bytes(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        q.append({'x': b'abcdefgh'})
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'123'})
        q.iset('x', b'12')
        assert q.get('x') == b'12'
        assert q.used('x') == 2
        q.append({'y': b'12345678'})
        assert q.used() == 10


def test_failed_overwrite_does_not_release_quota(tmp_path):
    class FailingFile(File):
        def _iset(self, key, value, lock=True):
            raise IOError("disk exploded")

    with FailingFile(str(tmp_path / 'p')) as f:
        q = Quota(f, global_bytes=10)
        q.append({'x': b'abcdef'})
        with pytest.raises(IOError):
            q.iset('x', b'z')
        # old bytes must remain charged and present
        assert q.used('x') == 6
        assert q.used() == 6
        assert q.get('x') == b'abcdef'


def test_failed_append_does_not_change_ledger(tmp_path):
    class FlakyFile(File):
        fail = False

        def append(self, data, lock=True, **kwargs):
            if self.fail:
                raise IOError("disk exploded")
            return File.append(self, data, lock=lock, **kwargs)

    with FlakyFile(str(tmp_path / 'p')) as f:
        q = Quota(f, global_bytes=10)
        q.append({'x': b'abcd'})
        assert q.used() == 4

        f.fail = True
        with pytest.raises(IOError):
            q.append({'y': b'efgh'})
        assert q.used() == 4
        assert q.used('y') == 0
        assert q.get('y') == b''

        f.fail = False
        q.append({'y': b'ef'})
        assert q.used() == 6


# --------------------------------------------------------------------------
# reopen: rebuild counters from persisted metadata or from contents
# --------------------------------------------------------------------------

def test_reopen_rebuilds_from_persisted_metadata(tmp_path):
    path = str(tmp_path / 'p')
    q = Quota(File(path), per_partition=100, global_bytes=20)
    q.append({'x': b'abcdef', 'y': b'gh'})
    assert os.path.exists(os.path.join(path, '.quota.json'))

    reopened = Quota(File(path), per_partition=100, global_bytes=20)
    assert reopened.used('x') == 6
    assert reopened.used('y') == 2
    assert reopened.used() == 8
    with pytest.raises(QuotaExceeded):
        reopened.append({'z': b'a' * 13})
    reopened.append({'z': b'a' * 12})
    assert reopened.used() == 20


def test_reopen_rebuilds_from_contents_when_metadata_missing(tmp_path):
    path = str(tmp_path / 'p')
    q = Quota(File(path), per_partition=100, global_bytes=20)
    q.append({('a', 'b'): b'abc', 'x': b'defgh'})

    os.remove(os.path.join(path, '.quota.json'))
    reopened = Quota(File(path), per_partition=100, global_bytes=20)
    assert reopened.used(('a', 'b')) == 3
    assert reopened.used('x') == 5
    assert reopened.used() == 8
    with pytest.raises(QuotaExceeded):
        reopened.append({'z': b'a' * 13})


def test_reopen_rebuilds_from_contents_when_metadata_corrupt(tmp_path):
    path = str(tmp_path / 'p')
    q = Quota(File(path), global_bytes=20)
    q.append({'x': b'abcdef'})

    with open(os.path.join(path, '.quota.json'), 'w') as f:
        f.write('{not json')
    reopened = Quota(File(path), global_bytes=20)
    assert reopened.used('x') == 6
    assert reopened.used() == 6

    with open(os.path.join(path, '.quota.json'), 'w') as f:
        f.write('{"total": 999, "lengths": {"78": 6}}')
    reopened = Quota(File(path), global_bytes=20)
    assert reopened.used('x') == 6
    assert reopened.used() == 6


def test_persists_through_buffer_composition(tmp_path):
    path = str(tmp_path / 'p')
    q = Quota(Buffer(Dict(), File(path), available_memory=1000),
              global_bytes=20)
    assert q.path == path
    q.append({'x': b'abcdef'})
    assert os.path.exists(os.path.join(path, '.quota.json'))

    reopened = Quota(Buffer(Dict(), File(path), available_memory=1000),
                     global_bytes=20)
    assert reopened.used('x') == 6
    assert reopened.used() == 6
    with pytest.raises(QuotaExceeded):
        reopened.append({'y': b'a' * 15})


def test_persists_through_nested_encoders(tmp_path):
    from partd import ZLib
    path = str(tmp_path / 'p')
    q = Quota(ZLib(File(path)), global_bytes=20)
    assert q.path is not None
    q.append({'x': b'abcdef'})
    assert q.used('x') == 6
    assert os.path.exists(os.path.join(q.path, '.quota.json'))


def test_drop_clears_ledger_and_metadata(tmp_path):
    path = str(tmp_path / 'p')
    q = Quota(File(path), global_bytes=20)
    q.append({'x': b'abcdef'})
    q.drop()
    assert q.used() == 0
    assert not os.path.exists(os.path.join(path, '.quota.json'))
    q.append({'x': b'a' * 20})
    assert q.used() == 20


def test_in_memory_backend_has_no_metadata():
    q = Quota(Dict(), global_bytes=10)
    q.append({'x': b'abc'})
    assert q.used() == 3
    with pytest.raises(QuotaExceeded):
        q.append({'y': b'a' * 8})


# --------------------------------------------------------------------------
# misc interface behavior
# --------------------------------------------------------------------------

def test_pickle_roundtrip(tmp_path):
    import pickle
    path = str(tmp_path / 'p')
    q = Quota(File(path), per_partition=100, global_bytes=20)
    q.append({'x': b'abcdef'})

    reopened = pickle.loads(pickle.dumps(q))
    assert reopened.used('x') == 6
    assert reopened.used() == 6
    with pytest.raises(QuotaExceeded):
        reopened.append({'y': b'a' * 15})


def test_get_missing_key_is_empty(tmp_path):
    with Quota(File(str(tmp_path / 'p')), global_bytes=10) as q:
        assert q.get('nope') == b''
        assert q.get(['nope', 'also']) == [b'', b'']
