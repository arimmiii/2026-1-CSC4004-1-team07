from __future__ import annotations

import os

import requests

from .config import load_project_dotenv
from .search_adapter import SearchAdapter, SearchRequest


class TavilySearchAdapter(SearchAdapter):
    def __init__(self, api_key: str | None = None, topic: str = "general") -> None:
        load_project_dotenv()
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.topic = topic
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is not set.")

    def search(self, request: SearchRequest) -> list[dict]:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": request.query,
                "topic": self.topic,
                "max_results": request.top_k,
                "search_depth": "advanced",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": item.get("url", ""),
                "published_at": item.get("published_date", ""),
            }
            for item in payload.get("results", [])
        ]


class SerpApiSearchAdapter(SearchAdapter):
    def __init__(self, api_key: str | None = None, engine: str = "google") -> None:
        load_project_dotenv()
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        self.engine = engine
        if not self.api_key:
            raise RuntimeError("SERPAPI_API_KEY environment variable is not set.")

    def search(self, request: SearchRequest) -> list[dict]:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": self.engine,
                "q": request.query,
                "api_key": self.api_key,
                "num": request.top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "published_at": item.get("date", ""),
            }
            for item in payload.get("organic_results", [])
        ]
