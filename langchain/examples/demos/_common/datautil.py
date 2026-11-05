"""Dataset helpers for the demos.

Datasets are streamed from the Hugging Face Hub (no full download) and the HF cache is kept under
``examples/demos/.data`` so re-runs are fast and nothing is committed. With ``DEMOS_FAKE=1`` the
demos read the small fixtures in ``_common/fixtures`` instead. ``*_LIMIT`` knobs live in each demo
and are read with ``env_int``.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .config import DEMOS_ROOT

DATA_DIR = DEMOS_ROOT / ".data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Keep all HF caching inside the demos folder.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATA_DIR / "hf" / "datasets"))


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def fixture(name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The records of ``_common/fixtures/<name>.jsonl``, at most ``limit`` of them."""
    lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records if limit is None else records[:limit]


def stream_hf(
    name: str,
    *,
    config: Optional[str] = None,
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield up to ``limit`` records from a HF dataset in streaming mode.

    Imported lazily so importing this module never pulls in the (heavy) datasets stack unless a
    demo actually needs it.

    A parquet dataset is read ahead on Arrow's thread pools, through Python file objects. Work
    of that kind still running when the interpreter exits deadlocks it on Python 3.13: a pool
    thread waits forever for the GIL while the exit waits for the pool (apache/arrow#50482). So
    when the stream stops, at ``limit`` or when the caller closes it, it is closed and the work
    it started is waited for, with the interpreter still running.
    """
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] streaming {name}" + (f"/{config}" if config else "") + f" (limit={limit}) ...")
    ds = load_dataset(name, config, split=split, streaming=True)
    records = iter(ds)
    try:
        for i, rec in enumerate(records):
            if limit is not None and i >= limit:
                break
            yield rec
    finally:
        records.close()
        del records, ds
        gc.collect()
        wait_for_arrow()


def wait_for_arrow() -> None:
    """Return when Arrow's IO and CPU thread pools have no work queued or running.

    The waits release the GIL, so pool threads in the middle of a Python call finish it. IO is
    waited for again after the CPU pool because a finished read hands its result to the CPU
    pool, which can pass work back.
    """
    pools = _arrow_pools()
    if pools is None:
        return
    io_pool, cpu_pool, wait_for_idle = pools
    for pool in (io_pool, cpu_pool, io_pool):
        wait_for_idle(pool)


@functools.cache
def _arrow_pools() -> Optional[tuple[int, int, Callable[[int], None]]]:
    """Arrow's IO and CPU thread pools and ``ThreadPool::WaitForIdle``, from ``libarrow``.

    pyarrow has no Python call for it. The symbols are Arrow's exported C++ API, under the
    Itanium names Linux and macOS use; elsewhere this warns once and returns None.
    """
    import pyarrow

    folder = Path(pyarrow.__file__).parent
    found = [p for p in folder.iterdir() if re.fullmatch(r"libarrow\.(\d+\.dylib|so\.\d+)", p.name)]
    try:
        lib = ctypes.CDLL(str(found[0]))  # a CDLL call releases the GIL while it runs
        io_pool = lib._ZN5arrow2io8internal15GetIOThreadPoolEv
        cpu_pool = lib._ZN5arrow8internal16GetCpuThreadPoolEv
        wait_for_idle = lib._ZN5arrow8internal10ThreadPool11WaitForIdleEv
    except (IndexError, OSError, AttributeError) as exc:
        warnings.warn(f"cannot wait for Arrow's thread pools ({exc}); an exit may hang")
        return None
    io_pool.restype = cpu_pool.restype = ctypes.c_void_p
    wait_for_idle.argtypes, wait_for_idle.restype = [ctypes.c_void_p], None
    return io_pool(), cpu_pool(), wait_for_idle


def batched(iterable, size: int):
    """Yield lists of up to ``size`` items from ``iterable``."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
