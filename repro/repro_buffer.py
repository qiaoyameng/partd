"""Reproduce Buffer ordering / bookkeeping / failure semantics."""
import os
import shutil
import tempfile

from partd import Buffer, Dict, File, Pickle, Encode
import zlib


def line(msg):
    print(msg)


def scenario_bytes(fast, slow, label):
    line(f"\n=== {label}: append/flush/append/get on bytes ===")
    with Buffer(fast, slow, available_memory=10) as p:
        p.append({'k': b'AA', 'j': b'11'})
        line(f"after append #1  fast={dict(fast.data) if isinstance(fast, Dict) else fast.get(['k','j'])} "
             f"slow={p.slow.get(['k','j'])} lengths={dict(p.lengths)} mem={p.memory_usage}")
        p.flush()
        line(f"after flush #1  fast={p.fast.get(['k','j'])} slow={p.slow.get(['k','j'])} "
             f"lengths={dict(p.lengths)} mem={p.memory_usage}")
        p.append({'k': b'BB'})
        line(f"after append #2  fast={p.fast.get(['k','j'])} slow={p.slow.get(['k','j'])} "
             f"lengths={dict(p.lengths)} mem={p.memory_usage}")
        got = p.get('k')
        line(f"p.get('k') = {got!r}  (expected b'AABB')")
        assert got == b'AABB', f"ORDER BUG: {got!r}"


def scenario_duplicate_keys_in_request(fast, slow, label):
    line(f"\n=== {label}: duplicate keys inside one append request ===")
    with Buffer(fast, slow, available_memory=1000) as p:
        p.append({'k': b'AA'})
        p.flush()
        p.append({'k': b'BB'})
        p.append({'k': b'CC'})
        got = p.get('k')
        line(f"p.get('k') = {got!r}  (expected b'AABBCC')")
        assert got == b'AABBCC', got
        assert p.lengths['k'] == 4 and p.memory_usage == 4, (dict(p.lengths), p.memory_usage)


def scenario_partial_flush(fast, slow, label):
    line(f"\n=== {label}: partial-key flush, multiple flushes ===")
    with Buffer(fast, slow, available_memory=1000) as p:
        p.append({'a': b'1', 'b': b'2', 'c': b'3'})
        p.flush(['a', 'b'])
        line(f"after partial flush fast={p.fast.get(['a','b','c'])} slow={p.slow.get(['a','b','c'])} "
             f"lengths={dict(p.lengths)} mem={p.memory_usage}")
        assert p.fast.get(['a', 'b', 'c']) == [b'', b'', b'3']
        assert p.slow.get(['a', 'b', 'c']) == [b'1', b'2', b'']
        assert dict(p.lengths) == {'c': 1} and p.memory_usage == 1
        p.append({'a': b'4', 'c': b'5'})
        p.flush()
        line(f"after full flush    fast={p.fast.get(['a','b','c'])} slow={p.slow.get(['a','b','c'])} "
             f"lengths={dict(p.lengths)} mem={p.memory_usage}")
        assert p.get(['a', 'b', 'c']) == [b'14', b'2', b'35']
        assert p.memory_usage == 0 and dict(p.lengths) == {}


def scenario_pickle(tmp, label):
    line(f"\n=== {label}: Pickle (Encode-wrapped frames) ordering ===")
    fast = Pickle(Dict())
    slow = Pickle(File(os.path.join(tmp, 'slow')))
    with Buffer(fast, slow, available_memory=10) as p:
        p.append({'k': [1, 2]})
        p.flush()
        p.append({'k': [3, 4]})
        got = p.get('k')
        line(f"p.get('k') = {got}  (expected [1, 2, 3, 4])")
        assert got == [1, 2, 3, 4], got


def scenario_encode(tmp, label):
    line(f"\n=== {label}: Encode(zlib) wrapped ordering ===")
    fast = Encode(zlib.compress, zlib.decompress, b''.join, Dict())
    slow = Encode(zlib.compress, zlib.decompress, b''.join,
                  File(os.path.join(tmp, 'slowz')))
    with Buffer(fast, slow, available_memory=10) as p:
        p.append({'k': b'hello-'})
        p.flush()
        p.append({'k': b'world'})
        got = p.get('k')
        line(f"p.get('k') = {got!r}  (expected b'hello-world')")
        assert got == b'hello-world', got


if __name__ == '__main__':
    tmp = tempfile.mkdtemp(prefix='partd-repro-')
    try:
        scenario_bytes(Dict(), Dict(), 'Dict+Dict')
        scenario_bytes(Dict(), File(os.path.join(tmp, 's1')), 'Dict+File')
        scenario_duplicate_keys_in_request(Dict(), Dict(), 'Dict+Dict')
        scenario_duplicate_keys_in_request(Dict(), File(os.path.join(tmp, 's2')), 'Dict+File')
        scenario_partial_flush(Dict(), Dict(), 'Dict+Dict')
        scenario_partial_flush(Dict(), File(os.path.join(tmp, 's3')), 'Dict+File')
        scenario_pickle(tmp, 'Pickle(Dict)+Pickle(File)')
        scenario_encode(tmp, 'Encode(Dict)+Encode(File)')
        line("\nALL SCENARIOS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
