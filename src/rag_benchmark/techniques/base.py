from __future__ import annotations

from typing import Protocol

from ..shared.records import DocumentRecord, TechniqueResult


class Technique(Protocol):
    name: str

    def index(self, collection: str, documents: list[DocumentRecord]) -> int: ...

    def answer(self, collection: str, question: str) -> TechniqueResult: ...

