import os
import pickle

import pytest

from partd import Buffer, Dict, File, Quota, QuotaExceeded
from partd.quota import METADATA_FILENAME


def test_fill_exactly_then_one_byte_over():
    with Quota(Dict(), partition_quota=10, total_quota=20) as q:
        q.append({'x': b'aaaaaa'})
        q.append({'x': b'bbbb'})  # partition filled to exactly 10
        assert q.get('x') == b'aaaaaabbbb'
        assert q.total_usage == 10

        with pytest.raises(QuotaExceeded):
            q.append({'x': b'c'})  # one byte over the partition quota
        assert q.get('x') == b'aaaaaabbbb'

        q.append({'y': b'aaaaaaaaaa'})  # total filled to exactly 20
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'c'})  # one byte over the total quota
        assert q.get('y') == b'aaaaaaaaaa'


def test_rejected_batch_writes_nothing():
    with Quota(Dict(), total_quota=10) as q:
        q.append({'x': b'aaaaa'})
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'bbbb', 'z': b'cc'})
        assert q.get('y') == b''
        assert q.get('z') == b''
        assert q.get('x') == b'aaaaa'
        assert q.total_usage == 5


def test_partition_quota_rejects_whole_batch():
    with Quota(Dict(), partition_quota=5) as q:
        with pytest.raises(QuotaExceeded):
            q.append({'x': b'ok', 'y': b'too-long!'})
        assert q.get('x') == b''
        assert q.get('y') == b''
        assert q.total_usage == 0


def test_buffer_flush_does_not_change_accounting():
    buf = Buffer(Dict(), File(), available_memory=10)
    with Quota(buf, total_quota=20) as q:
        q.append({'x': b'aaaaaa', 'y': b'bbbbbb'})  # 12 > 10, triggers flush
        assert q.total_usage == 12

        q.flush()
        assert q.total_usage == 12  # flushing neither charges nor frees

        q.append({'z': b'cccccccc'})  # exactly 20
        with pytest.raises(QuotaExceeded):
            q.append({'z': b'd'})
        assert q.get('x') == b'aaaaaa'
        assert q.get('y') == b'bbbbbb'
        assert q.get('z') == b'cccccccc'


def test_failed_delete_keeps_accounting_then_retry():
    with Quota(Dict(), total_quota=10) as q:
        q.append({'x': b'aaaaaaaaaa'})
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'b'})

        real_delete = q.partd.delete

        def failing_delete(keys, **kwargs):
            raise RuntimeError('boom')

        q.partd.delete = failing_delete
        with pytest.raises(RuntimeError):
            q.delete('x')
        # the failed delete must not free up any quota
        assert q.total_usage == 10
        with pytest.raises(QuotaExceeded):
            q.append({'y': b'b'})

        q.partd.delete = real_delete
        q.delete('x')  # retry succeeds and frees the space
        assert q.total_usage == 0
        q.append({'y': b'bbbbbbbbbb'})
        assert q.get('y') == b'bbbbbbbbbb'


def test_missing_key_and_empty_payload():
    with Quota(Dict(), partition_quota=4, total_quota=4) as q:
        q.delete('missing')  # deleting a missing key is a no-op
        assert q.get('missing') == b''

        q.append({'x': b''})  # empty payloads are always allowed
        q.append({'x': b'aaaa'})
        q.append({'x': b''})  # still allowed when the quota is full
        with pytest.raises(QuotaExceeded):
            q.append({'x': b'b'})

        q.delete(['missing', 'x'])
        assert q.total_usage == 0


def test_iset():
    with Quota(Dict(), partition_quota=10, total_quota=12) as q:
        q.iset('x', b'aaaaa')
        assert q.get('x') == b'aaaaa'
        assert q.total_usage == 5

        q.iset('y', b'bbbbbbb')  # total of exactly 12
        with pytest.raises(QuotaExceeded):
            q.iset('z', b'c')
        assert q.get('z') == b''

        with pytest.raises(QuotaExceeded):
            q.iset('w', b'1' * 11)  # over the partition quota
        assert q.get('w') == b''


def test_iset_overwrite_replaces_usage():
    with Quota(Dict(), total_quota=10) as q:
        q._iset('x', b'aaaaaaaa')
        q._iset('x', b'bb')  # overwriting shrinks usage
        assert q.total_usage == 2

        q._iset('x', b'cccccccccc')  # grow back to exactly 10
        assert q.total_usage == 10
        with pytest.raises(QuotaExceeded):
            q._iset('x', b'd' * 11)
        assert q.get('x') == b'cccccccccc'


def test_file_reopen_rebuilds_usage_from_metadata(tmp_path):
    path = str(tmp_path / 'store')
    q = Quota(File(path), total_quota=10)
    q.append({'x': b'aaaaaa', 'y': b'bbb'})
    assert os.path.exists(os.path.join(path, METADATA_FILENAME))

    q2 = Quota(File(path), total_quota=10)
    assert q2.total_usage == 9
    q2.append({'y': b'c'})  # exactly 10
    with pytest.raises(QuotaExceeded):
        q2.append({'z': b'd'})
    assert q2.get('x') == b'aaaaaa'
    assert q2.get('y') == b'bbbc'


def test_file_reopen_rebuilds_usage_from_content(tmp_path):
    path = str(tmp_path / 'store')
    q = Quota(File(path), total_quota=10)
    q.append({'x': b'aaaaaa'})
    os.remove(os.path.join(path, METADATA_FILENAME))

    q2 = Quota(File(path), total_quota=10)
    assert q2.total_usage == 6
    with pytest.raises(QuotaExceeded):
        q2.append({'y': b'ccccc'})
    q2.append({'y': b'cccc'})
    assert q2.get('y') == b'cccc'


def test_drop_resets_usage():
    with Quota(Dict(), total_quota=10) as q:
        q.append({'x': b'aaaaaaaaaa'})
        q.drop()
        assert q.total_usage == 0
        q.append({'x': b'aaaaaaaaaa'})
        assert q.get('x') == b'aaaaaaaaaa'


def test_pickle():
    with Quota(Dict(), partition_quota=10, total_quota=20) as q:
        q.append({'x': b'aaaa'})
        q2 = pickle.loads(pickle.dumps(q))
        assert q2.total_usage == 4
        q2.append({'x': b'bbbbbb'})
        assert q2.get('x') == b'aaaabbbbbb'
        with pytest.raises(QuotaExceeded):
            q2.append({'x': b'c'})
