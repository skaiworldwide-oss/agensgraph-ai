"""Dataset helpers for the demos.

Datasets are streamed from the Hugging Face Hub (no full download) and the HF
cache is kept under ``examples/demos/.data`` so re-runs are fast and nothing is
committed. ``*_LIMIT`` knobs live in each demo and are read with ``env_int``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional

from .config import DATA_DIR

# Keep all HF caching inside the demos folder.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATA_DIR / "hf" / "datasets"))


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def stream_hf(
    name: str,
    *,
    config: Optional[str] = None,
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield up to ``limit`` records from a HF dataset in streaming mode."""
    from datasets import load_dataset

    print(f"[data] streaming {name}" + (f"/{config}" if config else "") + f" (limit={limit}) ...")
    ds = load_dataset(name, config, split=split, streaming=True)
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield rec


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
