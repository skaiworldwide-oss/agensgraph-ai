"""What every component has: the engine it borrows connections from, and a close."""

from __future__ import annotations

from typing import Self

from ..errors import ClosedError
from .engine import AgensEngine, arelease, release


class Component:
    """The engine a component works through, and the share of it the component gives back.

    A component made with a conninfo string holds a share of the process's engine for that
    string, which :meth:`close` gives back; one handed an engine leaves it to its owner. After
    ``close`` every call raises :class:`~langchain_agensgraph.errors.ClosedError` naming
    ``_what``. A subclass sets ``_engine`` and ``_owns_engine`` in its constructor.
    """

    _what = "component"
    _engine: AgensEngine
    _owns_engine: bool
    _closed = False

    @property
    def engine(self) -> AgensEngine:
        return self._engine

    def _open(self) -> None:
        if self._closed:
            raise ClosedError(f"the {self._what} is closed")

    def close(self) -> None:
        """Give back this component's share of an engine it was handed as a conninfo string."""
        if not self._closed:
            self._closed = True
            if self._owns_engine:
                release(self._engine)

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_engine:
                await arelease(self._engine)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
