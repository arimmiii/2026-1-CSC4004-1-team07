from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchRequest:
    query: str
    domain: str
    top_k: int = 10


class SearchAdapter(Protocol):
    def search(self, request: SearchRequest) -> list[dict]:
        """Return raw search results as a list of dictionaries."""


class InMemorySearchAdapter:
    """Test adapter that serves predefined search results."""

    def __init__(self, results_by_query: dict[str, list[dict]] | None = None) -> None:
        self.results_by_query = results_by_query or {}

    def search(self, request: SearchRequest) -> list[dict]:
        return list(self.results_by_query.get(request.query, []))[: request.top_k]


class NullSearchAdapter:
    """Adapter placeholder for integration before a real provider is wired."""

    def search(self, request: SearchRequest) -> list[dict]:
        return []
