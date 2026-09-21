"""Tools the specialist agents can call, and the evidence they produce.

Every search or read that returns content becomes an evidence item with a
stable id (E1, E2, ...). The synthesizer may only cite those ids, which is
what makes the citation check in the eval possible.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

import mcp

from .tracing import span


class Evidence(TypedDict):
    id: str
    source: str  # docs | sql | web | wikipedia | arxiv
    title: str
    url: str
    content: str


Handler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    # Which source label this tool's results carry; None means the result is not evidence.
    evidence_source: str | None = None

    def api_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _schema(**props: tuple[str, str]) -> dict[str, Any]:
    """Build a JSON schema from name=(type, description); every property is required."""
    return {
        "type": "object",
        "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": list(props),
    }


def local_tools(web, wikipedia, vectors) -> list[ToolSpec]:
    async def web_search(args):
        return json.dumps(await web.search(args["query"]))

    async def wikipedia_search(args):
        return json.dumps(await wikipedia.search(args["query"]))

    async def wikipedia_page(args):
        return json.dumps(await wikipedia.page(args["title"]))

    async def vector_search(args):
        return json.dumps(await vectors.search(args["query"]))

    return [
        ToolSpec(
            "web_search",
            f"Search the web for recent energy news and analysis (backend: {web.backend}).",
            _schema(query=("string", "Search query")),
            web_search,
            "web",
        ),
        ToolSpec(
            "wikipedia_search",
            "Find Wikipedia article titles matching a query. Returns titles and short snippets only.",
            _schema(query=("string", "Search query")),
            wikipedia_search,
        ),
        ToolSpec(
            "wikipedia_page",
            "Fetch the plain text of a Wikipedia article by exact title.",
            _schema(title=("string", "Exact article title, e.g. 'Capacity factor'")),
            wikipedia_page,
            "wikipedia",
        ),
        ToolSpec(
            "paper_search",
            "Semantic search over arXiv research abstracts on solar, wind, storage, hydrogen, nuclear and grids.",
            _schema(query=("string", "What the papers should be about")),
            vector_search,
            "arxiv",
        ),
    ]


MCP_EVIDENCE = {"search_docs": "docs", "read_doc": "docs", "run_sql": "sql"}


async def mcp_tools(client: mcp.Client) -> list[ToolSpec]:
    listed = await client.list_tools()
    specs = []
    for tool in listed.tools:

        async def call(args, _name=tool.name):
            result = await client.call_tool(_name, args)
            text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
            if result.is_error:
                raise RuntimeError(text)
            return text

        specs.append(ToolSpec(tool.name, tool.description or "", tool.input_schema, call, MCP_EVIDENCE.get(tool.name)))
    return specs


def _items(source: str, tool: str, args: dict[str, Any], raw: str) -> list[tuple[str, str, str]]:
    """Split a tool result into (title, url, content) evidence items."""
    data = json.loads(raw)
    if tool == "run_sql":
        return [(f"SQL: {args.get('sql_query', '')}", "owid-energy-data", json.dumps(data))]
    if tool == "read_doc":
        return [(data["doc_id"], "", data["text"])]
    if tool == "wikipedia_page":
        return [(data["title"], data["url"], data["text"])]
    if source == "docs":
        return [(h["section"], h["url"], h["text"]) for h in data]
    if source == "web":
        return [(h["title"], h["url"], h["snippet"]) for h in data]
    if source == "arxiv":
        return [(f"{h['title']} ({h['published']})", h["url"], h["abstract"]) for h in data]
    return [(tool, "", raw)]


class Toolbox:
    def __init__(self, tools: list[ToolSpec], max_chars: int = 2000) -> None:
        self.tools = {t.name: t for t in tools}
        self.max_chars = max_chars

    def subset(self, names: list[str]) -> list[ToolSpec]:
        return [self.tools[n] for n in names if n in self.tools]

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, list[Evidence] | None]:
        """Run a tool.

        Returns (text, None) for tools whose output is not evidence, or ("", items) for
        evidence tools; the caller numbers the items and renders them with `render`.
        """
        spec = self.tools.get(name)
        if spec is None:
            raise KeyError(f"unknown tool {name}")
        with span(f"tool.{name}", args=json.dumps(args)[:500]):
            raw = await spec.handler(args)
        if spec.evidence_source is None:
            return raw[: self.max_chars * 2], None
        return "", [
            Evidence(id="", source=spec.evidence_source, title=title, url=url, content=content[: self.max_chars])
            for title, url, content in _items(spec.evidence_source, name, args, raw)
        ]


def render(items: list[Evidence]) -> str:
    if not items:
        return "no results"
    return "\n\n".join(f"[{e['id']}] {e['title']}\n{e['url']}\n{e['content']}" for e in items)
