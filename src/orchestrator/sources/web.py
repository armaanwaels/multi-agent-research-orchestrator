"""Source 3: web search.

The default backend is a fixture: a committed snapshot of 40 EIA "Today in
Energy" articles searched with BM25. It keeps the eval deterministic and free.
Set WEB_SEARCH_BACKEND=duckduckgo (and install the `live-web` extra) to search
the real web instead.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from .text_search import BM25, Passage


class FixtureWebSearch:
    backend = "fixture"

    def __init__(self, path: Path) -> None:
        self.articles = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.index = BM25([Passage(a["url"], a["title"], a["url"], f"{a['date']}. {a['text']}") for a in self.articles])

    async def search(self, query: str, k: int = 5) -> list[dict]:
        return [{"title": p.title, "url": p.url, "snippet": p.text[:1200]} for _, p in self.index.search(query, k)]

    async def read(self, url: str) -> dict:
        for a in self.articles:
            if a["url"] == url:
                return {"title": a["title"], "url": url, "text": f"{a['date']}. {a['text']}"}
        raise KeyError(f"{url} is not in the web fixture; only URLs returned by web_search can be read")


class DuckDuckGoWebSearch:
    backend = "duckduckgo"

    async def search(self, query: str, k: int = 5) -> list[dict]:
        from ddgs import DDGS

        hits = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=k)))
        return [{"title": h["title"], "url": h["href"], "snippet": h["body"]} for h in hits]

    async def read(self, url: str, max_chars: int = 6000) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            html = (await client.get(url)).text
        title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = " ".join(re.sub(r"<[^>]+>", " ", body).split())
        return {"title": title.group(1).strip() if title else url, "url": url, "text": text[:max_chars]}


def make_web_search(backend: str, data_dir: Path):
    if backend == "fixture":
        return FixtureWebSearch(data_dir / "web_fixture.jsonl")
    if backend == "duckduckgo":
        return DuckDuckGoWebSearch()
    raise ValueError(f"unknown WEB_SEARCH_BACKEND {backend!r}")
