"""Local, offline repartitioning of partd datasets.

This module re-buckets the records of an existing partitioned dataset
(e.g. moving an offline shuffle from 4 to 7 partitions) without
touching the source.  The caller explicitly provides the source
partition list, a record decoder, a key function and the target bucket
count.

Shuffle keys are currently restricted to ``str``, ``int`` and tuples of
those.  Buckets are assigned from a stable, type-tagged encoding of the
key (see ``encode_key``) hashed with blake2b -- never the builtin
``hash``, which varies per process for strings.
"""
import hashlib
import os
import shutil
import struct
import tempfile

from .buffer import Buffer
from .dict import Dict
from .encode import Encode
from .file import File
from .utils import frame


class RepartitionError(Exception):
    """A repartition run failed.

    Temporary output state has been cleaned up and any pre-existing
    target directory was left untouched.
    """


def _identity(x):
    return x


def bytes_partd(path=None):
    """A File-backed partd storing lists of opaque ``bytes`` records."""
    return Encode(_identity, _identity, _identity, File(path))


def encode_key(key):
    """Stable, type-tagged encoding of a shuffle key.

    Supports ``str``, ``int`` and (possibly nested) tuples of those.
    Type tags and length-prefixed tuple boundaries keep encodings
    unambiguous: ``("ab", "c")`` and ``("a", "bc")`` differ, as do
    ``"1"`` and ``1``.  The result is identical in every process,
    unlike the builtin ``hash``.
    """
    if isinstance(key, str):
        return b"s" + key.encode("utf-8")
    if isinstance(key, bool):
        raise TypeError("unsupported shuffle key type: 'bool'")
    if isinstance(key, int):
        return b"i" + str(key).encode("ascii")
    if isinstance(key, tuple):
        parts = [encode_key(part) for part in key]
        return (b"t" + struct.pack(">I", len(parts)) + b"".join(
            struct.pack(">I", len(part)) + part for part in parts))
    raise TypeError(
        "unsupported shuffle key type: %r" % type(key).__name__)


def bucket_of(key, nparts):
    """Deterministic target bucket for ``key`` in ``range(nparts)``."""
    digest = hashlib.blake2b(encode_key(key), digest_size=8).digest()
    return int.from_bytes(digest, "big") % nparts


def repartition(source, partitions, decode, key, nparts, path,
                available_memory=1e8):
    """Repartition records from ``source`` into ``nparts`` buckets.

    Parameters
    ----------
    source : partd.Interface
        Read-only source.  ``source.get(partition)`` must return a list
        of encoded records (``bytes``); ``bytes_partd`` produces and
        consumes exactly this shape.
    partitions : list
        Source partition keys.  Records are consumed in this declared
        order, which -- together with the source record order -- defines
        the record order within each target bucket.
    decode : callable
        Maps an encoded record to a Python object.  Used only so that
        ``key`` can extract a shuffle key; the original encoded record
        is what gets written to the output, so duplicates are preserved
        verbatim.
    key : callable
        Maps a decoded record to a shuffle key: ``str``, ``int`` or a
        tuple of those.
    nparts : int
        Number of target buckets.
    path : str
        Target directory.  Must not exist yet.  Output is written to a
        temporary sibling directory first and published (renamed) to
        ``path`` only after every bucket is complete.
    available_memory : int
        Memory budget in bytes shared by *all* target bucket buffers;
        excess spills to disk in arrival order.

    Returns
    -------
    partd.Interface
        Read interface over the published directory.  ``get(bucket)``
        returns the bucket's records as a list of encoded ``bytes`` in
        stable order (source partition order, then source record order).

    Raises
    ------
    RepartitionError
        If the target already exists, or decoding / key extraction
        fails.  Temporary state is removed and any pre-existing target
        is left untouched.
    """
    partitions = list(partitions)
    if not isinstance(nparts, int) or nparts < 1:
        raise ValueError("nparts must be a positive integer, got %r" % nparts)
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise RepartitionError("target already exists: %s" % path)
    parent = os.path.dirname(path)
    tmp_path = tempfile.mkdtemp(
        prefix=os.path.basename(path) + ".tmp-", dir=parent)
    buffer = Buffer(Dict(), File(tmp_path), available_memory=available_memory)
    try:
        for partition in partitions:
            for index, record in enumerate(source.get(partition)):
                try:
                    bucket = bucket_of(key(decode(record)), nparts)
                except Exception as exc:
                    raise RepartitionError(
                        "failed to decode or extract key for record %d of "
                        "source partition %r: %s" % (index, partition, exc)
                    ) from exc
                buffer.append({bucket: frame(record)})
        buffer.flush()
        os.rename(tmp_path, path)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    return bytes_partd(path)
