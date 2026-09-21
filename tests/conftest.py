from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orchestrator.llm import Ledger, Usage
from orchestrator.sources.sql import build_database

DATA = Path(__file__).resolve().parents[1] / "data"


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(name: str, args: dict, id: str):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id)


def response(*blocks, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
        model="scripted",
    )


class ScriptedLLM:
    """Stands in for the Anthropic API. `script(role, kwargs)` returns the next response."""

    def __init__(self, script):
        self.script = script
        self.ledger = Ledger()
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def create(self, *, role: str, model: str, **kwargs: Any):
        self.requests.append((role, kwargs))
        self.ledger.calls.append(Usage(role, "claude-opus-5", 100, 20))
        return self.script(role, kwargs)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("db") / "energy.db"
    build_database(DATA / "owid_energy.csv", path)
    return path


class FakeVectors:
    async def search(self, query: str, k: int = 5) -> list[dict]:
        return [
            {
                "title": "Levelized cost of green hydrogen",
                "url": "https://arxiv.org/abs/0000.00000",
                "published": "2024-01-01",
                "score": 0.9,
                "abstract": f"An abstract about {query}.",
            }
        ]


def plan_json(*steps: tuple[str, str]) -> str:
    return json.dumps({"steps": [{"agent": a, "instruction": i} for a, i in steps]})
