"""Source 4: the live Wikipedia API."""

from __future__ import annotations

import re

import httpx

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "multi-agent-research-orchestrator/0.1 (github.com/armaanwaels)"}


class Wikipedia:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(headers=HEADERS, timeout=20)

    async def search(self, query: str, k: int = 5) -> list[dict]:
        r = await self.client.get(
            API,
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": k, "format": "json"},
        )
        r.raise_for_status()
        hits = r.json()["query"]["search"]
        return [{"title": h["title"], "snippet": re.sub(r"<[^>]+>", "", h["snippet"])} for h in hits]

    async def page(self, title: str, max_chars: int = 6000) -> dict:
        r = await self.client.get(
            API,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
                "format": "json",
            },
        )
        r.raise_for_status()
        page = next(iter(r.json()["query"]["pages"].values()))
        if "missing" in page:
            raise KeyError(f"no Wikipedia page titled {title!r}")
        return {
            "title": page["title"],
            "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
            "text": page.get("extract", "")[:max_chars],
        }
