"""Settings, read from the environment (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[2]
# Nearest .env walking up from the working directory; real environment variables win.
load_dotenv(find_dotenv(usecwd=True))

# USD per million tokens: (input, output). Cache reads bill at 0.1x input, cache writes at 1.25x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class Settings:
    # One model per role so the eval can trade cost against quality per role.
    planner_model: str = field(default_factory=lambda: _env("ORCH_PLANNER_MODEL", "claude-opus-5"))
    worker_model: str = field(default_factory=lambda: _env("ORCH_WORKER_MODEL", "claude-opus-5"))
    synth_model: str = field(default_factory=lambda: _env("ORCH_SYNTH_MODEL", "claude-opus-5"))
    judge_model: str = field(default_factory=lambda: _env("ORCH_JUDGE_MODEL", "claude-opus-5"))

    data_dir: Path = field(default_factory=lambda: Path(_env("ORCH_DATA_DIR", str(ROOT / "data"))))
    sqlite_path: Path = field(default_factory=lambda: Path(_env("ORCH_SQLITE_PATH", str(ROOT / "data" / "energy.db"))))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = "arxiv_abstracts"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    postgres_url: str = field(
        default_factory=lambda: _env(
            "POSTGRES_URL", "postgresql://orchestrator:orchestrator@localhost:5433/orchestrator"
        )
    )
    # "fixture" searches the committed EIA snapshot; "duckduckgo" goes to the live web.
    web_search: str = field(default_factory=lambda: _env("WEB_SEARCH_BACKEND", "fixture"))
    # "otel" exports OpenTelemetry spans; "off" disables tracing.
    tracing: str = field(default_factory=lambda: _env("ORCH_TRACING", "off"))

    max_agent_turns: int = 5
    max_revisions: int = 2
    tool_result_chars: int = 2000


def price(model: str) -> tuple[float, float]:
    return PRICES.get(model, PRICES["claude-opus-5"])
