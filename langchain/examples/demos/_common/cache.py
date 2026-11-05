"""Disk caches for the demos, on when ``DEMOS_CACHE_DIR`` names a folder.

Embeddings are kept by model, settings and text, and model answers by prompt and model settings,
in one SQLite file, so running the demos against a second server repeats no paid call the first
made. Vectors are kept as float32, the precision the vector store holds them at, and every caller
gets them back from the cache, so each run sees the same vectors.
"""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings


class _Table:
    """One key-value table in the cache file, safe to share between threads."""

    def __init__(self, folder: str, name: str) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(Path(folder) / "cache.sqlite", check_same_thread=False)
        self._lock = threading.Lock()
        self._name = name
        with self._lock:
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS {name} (k TEXT PRIMARY KEY, v BLOB NOT NULL)"
            )
            self._db.commit()

    def get(self, keys: Sequence[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._db.execute(
                    f"SELECT k, v FROM {self._name} WHERE k IN ({marks})", chunk
                ).fetchall()
                found.update(rows)
        return found

    def put(self, pairs: Iterable[tuple[str, bytes]]) -> None:
        with self._lock:
            self._db.executemany(
                f"INSERT OR REPLACE INTO {self._name} (k, v) VALUES (?, ?)", list(pairs)
            )
            self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._db.execute(f"DELETE FROM {self._name}")
            self._db.commit()


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


class CachedEmbeddings(Embeddings):
    """``inner``'s embeddings, each text embedded once per ``model``: the model's name and any
    settings that change its vectors. A query is kept apart from a document of the same text and
    embedded by ``embed_query``, since a model may embed the two differently."""

    def __init__(self, inner: Embeddings, folder: str, model: str) -> None:
        self.inner = inner
        self.model = model
        self._table = _Table(folder, "embeddings")

    def _split(self, texts: Sequence[str]) -> tuple[list[str], dict[str, bytes], list[int]]:
        keys = [_key(self.model, text) for text in texts]
        found = self._table.get(keys)
        return keys, found, [i for i, key in enumerate(keys) if key not in found]

    def _store(
        self, keys: list[str], found: dict[str, bytes], missing: list[int], vectors: Any
    ) -> list[list[float]]:
        made = [
            (keys[i], np.asarray(vector, dtype=np.float32).tobytes())
            for i, vector in zip(missing, vectors, strict=True)
        ]
        self._table.put(made)
        found.update(made)
        return [np.frombuffer(found[key], dtype=np.float32).tolist() for key in keys]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys, found, missing = self._split(texts)
        vectors = self.inner.embed_documents([texts[i] for i in missing]) if missing else []
        return self._store(keys, found, missing, vectors)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        keys, found, missing = self._split(texts)
        vectors = await self.inner.aembed_documents([texts[i] for i in missing]) if missing else []
        return self._store(keys, found, missing, vectors)

    def embed_query(self, text: str) -> list[float]:
        key = _key(self.model, "query", text)
        found = self._table.get([key])
        if key not in found:
            found = dict(self._query_kept(key, self.inner.embed_query(text)))
        return np.frombuffer(found[key], dtype=np.float32).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        key = _key(self.model, "query", text)
        found = self._table.get([key])
        if key not in found:
            found = dict(self._query_kept(key, await self.inner.aembed_query(text)))
        return np.frombuffer(found[key], dtype=np.float32).tolist()

    def _query_kept(self, key: str, vector: Any) -> list[tuple[str, bytes]]:
        made = [(key, np.asarray(vector, dtype=np.float32).tobytes())]
        self._table.put(made)
        return made


class AnswerCache(BaseCache):
    """A chat model's answers, by prompt and model settings.

    Answers are pickled, since a structured answer carries the parsed object, which LangChain's
    own serializer refuses; the file holds only what this machine's runs wrote. An answer that
    does not pickle is not kept, and one that no longer loads is asked for again.
    """

    def __init__(self, folder: str) -> None:
        self._table = _Table(folder, "answers")

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        key = _key(llm_string, prompt)
        found = self._table.get([key])
        if not found:
            return None
        try:
            return cast(RETURN_VAL_TYPE, pickle.loads(found[key]))
        except Exception:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        try:
            kept = pickle.dumps(list(return_val))
        except Exception:
            return
        self._table.put([(_key(llm_string, prompt), kept)])

    def clear(self, **kwargs: Any) -> None:
        self._table.clear()
