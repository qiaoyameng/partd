import json
import os
import pickle
import subprocess
import sys
from collections import Counter

import pytest

from partd import RepartitionError, repartition
from partd.repartition import bucket_of, bytes_partd, encode_key


KEYS = ['alpha', 'beta', 7, 42, ('a', 1), ('a', 2), 'alpha', 7]
NPARTITIONS_IN = 4
NPARTS_OUT = 7


def make_records():
    # small dataset with duplicate records, keys cover str/int/tuple
    return [(KEYS[i % len(KEYS)], i % 5) for i in range(40)]


def partition_records(records, p):
    return records[p::NPARTITIONS_IN]


def build_source(path):
    source = bytes_partd(str(path))
    for p in range(NPARTITIONS_IN):
        for record in partition_records(make_records(), p):
            source.append({p: pickle.dumps(record)})
    return source


def test_encode_key_typed_boundaries():
    assert encode_key(('ab', 'c')) != encode_key(('a', 'bc'))
    assert encode_key('1') != encode_key(1)
    assert encode_key((1,)) != encode_key(1)
    assert encode_key(('x', 3)) == encode_key(('x', 3))
    assert encode_key('héllo') == encode_key('héllo')
    with pytest.raises(TypeError):
        encode_key(1.5)
    with pytest.raises(TypeError):
        encode_key(None)
    with pytest.raises(TypeError):
        encode_key(True)
    with pytest.raises(TypeError):
        encode_key(b'bytes')


def test_repartition_multiset_and_order(tmp_path):
    source = build_source(tmp_path / 'src')
    before = [source.get(p) for p in range(NPARTITIONS_IN)]

    # tiny shared budget forces spilling to disk mid-run
    out = repartition(source, [0, 1, 2, 3], pickle.loads,
                      lambda record: record[0], NPARTS_OUT,
                      str(tmp_path / 'out'), available_memory=128)

    records = make_records()
    # output record multiset equals input record multiset
    expected = Counter(pickle.dumps(record) for record in records)
    got = Counter()
    for bucket in range(NPARTS_OUT):
        got.update(out.get(bucket))
    assert got == expected

    # within a bucket: source partition declaration order, then record order
    for bucket in range(NPARTS_OUT):
        got_records = [pickle.loads(chunk) for chunk in out.get(bucket)]
        expected_records = [
            record
            for p in range(NPARTITIONS_IN)
            for record in partition_records(records, p)
            if bucket_of(record[0], NPARTS_OUT) == bucket
        ]
        assert got_records == expected_records

    # source is untouched
    assert [source.get(p) for p in range(NPARTITIONS_IN)] == before


def test_repartition_refuses_to_overwrite_target(tmp_path):
    source = build_source(tmp_path / 'src')
    target = tmp_path / 'out'
    target.mkdir()
    sentinel = target / 'sentinel'
    sentinel.write_text('old')

    with pytest.raises(RepartitionError, match='already exists'):
        repartition(source, [0, 1, 2, 3], pickle.loads,
                    lambda record: record[0], NPARTS_OUT, str(target))

    assert sentinel.read_text() == 'old'
    assert not list(tmp_path.glob('*.tmp-*'))


def test_repartition_failure_cleans_up(tmp_path):
    source = build_source(tmp_path / 'src')

    def bad_decode(record):
        raise ValueError('cannot decode')

    with pytest.raises(RepartitionError, match='record 0 of source'):
        repartition(source, [0, 1, 2, 3], bad_decode,
                    lambda record: record[0], NPARTS_OUT,
                    str(tmp_path / 'out'))

    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('*.tmp-*'))

    def bad_key(record):
        return 1.5  # unsupported key type

    with pytest.raises(RepartitionError):
        repartition(source, [0, 1, 2, 3], pickle.loads, bad_key,
                    NPARTS_OUT, str(tmp_path / 'out'))

    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('*.tmp-*'))


def test_repartition_invalid_nparts(tmp_path):
    source = build_source(tmp_path / 'src')
    with pytest.raises(ValueError):
        repartition(source, [0], pickle.loads, lambda record: record[0],
                    0, str(tmp_path / 'out'))


TWO_PROCESS_SCRIPT = """
import json
import pickle
import sys

from partd.repartition import bytes_partd, repartition

run_dir = sys.argv[1]

keys = ['alpha', 'beta', 7, 42, ('a', 1), ('a', 2), 'alpha', 7]
records = [(keys[i % len(keys)], i % 5) for i in range(40)]

source = bytes_partd(run_dir + '/src')
for p in range(4):
    for record in records[p::4]:
        source.append({p: pickle.dumps(record)})

out = repartition(source, [0, 1, 2, 3], pickle.loads,
                  lambda record: record[0], 7, run_dir + '/out',
                  available_memory=128)

buckets = [[pickle.loads(chunk) for chunk in out.get(b)] for b in range(7)]
print(json.dumps(buckets))
"""


def test_repartition_deterministic_across_processes(tmp_path):
    outputs = []
    for i, seed in enumerate(['0', '1']):
        run_dir = tmp_path / ('run%d' % i)
        run_dir.mkdir()
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(subprocess.check_output(
            [sys.executable, '-c', TWO_PROCESS_SCRIPT, str(run_dir)],
            env=env))

    # identical bucket assignment and identical per-bucket order
    assert outputs[0] == outputs[1]

    buckets = json.loads(outputs[0])
    assert len(buckets) == NPARTS_OUT

    # output record multiset equals input record multiset
    expected = Counter(json.dumps(record) for record in make_records())
    got = Counter(json.dumps(record)
                  for bucket in buckets for record in bucket)
    assert got == expected
