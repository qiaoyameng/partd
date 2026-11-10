"""Fault-injection scenarios for Buffer failure semantics.

Failures are injected only in the in-memory Dict fast store or a temporary
directory backing the slow File store.
"""
import os
import shutil
import tempfile

from partd import Buffer, Dict, File
from partd.buffer import BufferErrorState


class ExplodingDict(Dict):
    """Dict whose append raises, optionally after partially recording data."""
    def __init__(self, partial=False):
        super().__init__()
        self.partial = partial
        self.calls = 0

    def append(self, data, lock=True, **kwargs):
        self.calls += 1
        if self.partial:
            first_key = next(iter(data))
            super().append({first_key: data[first_key]}, lock=lock)
        raise RuntimeError('injected fast.append failure')


class ExplodingFile(File):
    """File whose append raises after writing some keys to temp dir."""
    def __init__(self, path, fail_after=1):
        super().__init__(path=path)
        self.fail_after = fail_after
        self.calls = 0

    def append(self, data, lock=True, **kwargs):
        self.calls += 1
        items = list(data.items())
        super().append(dict(items[:self.fail_after]), lock=lock)
        raise RuntimeError('injected slow.append partial-write failure')


def expect_buffer_error(label, fn):
    try:
        fn()
    except BufferErrorState as e:
        print(f"  [{label}] blocked retry with BufferErrorState: {e}")
        return
    raise AssertionError(f"{label}: expected BufferErrorState")


def case_fast_append_total_failure():
    print("\n=== fast.append fails outright: no phantom state, retries blocked ===")
    fast, slow = ExplodingDict(partial=False), Dict()
    with Buffer(fast, slow, available_memory=1000) as p:
        try:
            p.append({'k': b'AA', 'j': b'BB'})
        except RuntimeError as e:
            print(f"  append raised: {e}")
        assert dict(p.lengths) == {}, p.lengths
        assert p.memory_usage == 0, p.memory_usage
        assert fast.data == {}, fast.data
        assert slow.data == {}, slow.data
        print(f"  state after failure: lengths={dict(p.lengths)} mem={p.memory_usage} "
              f"fast={dict(fast.data)} slow={dict(slow.data)}")
        expect_buffer_error('retry append', lambda: p.append({'k': b'CC'}))
        expect_buffer_error('retry flush', lambda: p.flush())
        # reads stay consistent: nothing was written anywhere
        assert p.get(['k', 'j']) == [b'', b'']
        print("  get after poisoning: ['', ''] consistent with empty stores")


def case_slow_append_partial_write():
    print("\n=== slow.append partially writes then fails: no rollback, no blind retry ===")
    tmp = tempfile.mkdtemp(prefix='partd-fail-')
    try:
        fast, slow = Dict(), ExplodingFile(os.path.join(tmp, 'slow'), fail_after=1)
        with Buffer(fast, slow, available_memory=1000) as p:
            p.append({'a': b'11', 'b': b'22', 'c': b'33'})
            try:
                p.flush()
            except RuntimeError as e:
                print(f"  flush raised: {e}")
            # fast copies must still be there; bookkeeping unchanged
            assert fast.get(['a', 'b', 'c']) == [b'11', b'22', b'33']
            assert dict(p.lengths) == {'a': 2, 'b': 2, 'c': 2}
            assert p.memory_usage == 6
            print(f"  fast intact={fast.get(['a','b','c'])} lengths={dict(p.lengths)} "
                  f"mem={p.memory_usage}")
            print(f"  slow on disk (partial write, NOT rolled back)={slow.get(['a','b','c'])}")
            # blind retry would append b'11' a second time -> blocked
            expect_buffer_error('retry flush', lambda: p.flush(['a']))
            expect_buffer_error('more appends', lambda: p.append({'a': b'zz'}))
            # reads still reflect true global order: slow partial + fast full
            got = p.get(['a', 'b', 'c'])
            print(f"  get after poisoning: {got}")
            assert got == [b'11' + b'11', b'22', b'33'], got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case_fast_delete_failure():
    print("\n=== slow.append ok but fast.delete fails: data safe on slow, error marked ===")
    class FailDeleteDict(Dict):
        def _delete(self, keys, lock=True):
            raise RuntimeError('injected fast.delete failure')

    fast, slow = FailDeleteDict(), Dict()
    with Buffer(fast, slow, available_memory=1000) as p:
        p.append({'k': b'AA'})
        try:
            p.flush()
        except RuntimeError as e:
            print(f"  flush raised: {e}")
        assert slow.get(['k']) == [b'AA']
        print(f"  slow holds the data: {slow.get(['k'])}; error={p._error}")
        expect_buffer_error('retry append', lambda: p.append({'k': b'BB'}))


def case_auto_flush_failure_during_append():
    print("\n=== auto flush inside append hits partial slow failure ===")
    tmp = tempfile.mkdtemp(prefix='partd-fail-')
    try:
        fast, slow = Dict(), ExplodingFile(os.path.join(tmp, 'slow'), fail_after=1)
        with Buffer(fast, slow, available_memory=4) as p:
            p.append({'a': b'11'})  # fits
            try:
                p.append({'b': b'22', 'c': b'33'})  # mem 6 > 4 -> auto flush
            except RuntimeError as e:
                print(f"  append(auto-flush) raised: {e}")
            assert p._error is not None
            print(f"  poisoned={p._error} lengths={dict(p.lengths)} mem={p.memory_usage}")
            # all three fast copies remain described by lengths
            assert dict(p.lengths) == {'a': 2, 'b': 2, 'c': 2}
            assert p.memory_usage == 6
            expect_buffer_error('retry', lambda: p.flush())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    case_fast_append_total_failure()
    case_slow_append_partial_write()
    case_fast_delete_failure()
    case_auto_flush_failure_during_append()
    print("\nALL FAILURE SCENARIOS BEHAVE AS DESIGNED")
