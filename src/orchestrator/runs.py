"""Run records in PostgreSQL: one row per research run, next to LangGraph's checkpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    thread_id   TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    status      TEXT NOT NULL,            -- running | awaiting_review | done | failed
    answer      TEXT,
    evidence    JSONB,
    cost_usd    DOUBLE PRECISION DEFAULT 0,   -- accumulated across processes
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
)
"""


class RunStore(Protocol):
    async def upsert(self, thread_id: str, question: str, **fields: Any) -> None: ...
    async def recent(self, limit: int = 20) -> list[dict]: ...


class MemoryRunStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def upsert(self, thread_id: str, question: str, **fields: Any) -> None:
        row = self.rows.setdefault(thread_id, {"thread_id": thread_id, "cost_usd": 0.0})
        row["cost_usd"] += fields.pop("cost_usd", 0.0)
        row.update(question=question, **fields)

    async def recent(self, limit: int = 20) -> list[dict]:
        return list(self.rows.values())[-limit:]


class PostgresRunStore:
    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self.conn = conn

    @classmethod
    @asynccontextmanager
    async def connect(cls, url: str) -> AsyncIterator[PostgresRunStore]:
        async with await psycopg.AsyncConnection.connect(url, autocommit=True, row_factory=dict_row) as conn:
            await conn.execute(SCHEMA)
            yield cls(conn)

    async def upsert(self, thread_id: str, question: str, **fields: Any) -> None:
        # question is required: Postgres checks NOT NULL on the insert row before resolving the conflict.
        fields = {"question": question, **fields}
        if "evidence" in fields:
            fields["evidence"] = json.dumps(fields["evidence"])
        cols = ["thread_id", *fields]
        # cost_usd is a delta: a run paused in one process and resumed in another adds up.
        updates = ", ".join(
            f"{c} = runs.cost_usd + EXCLUDED.cost_usd" if c == "cost_usd" else f"{c} = EXCLUDED.{c}" for c in fields
        )
        await self.conn.execute(
            f"INSERT INTO runs ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
            f"ON CONFLICT (thread_id) DO UPDATE SET {updates}, updated_at = now()",
            [thread_id, *fields.values()],
        )

    async def recent(self, limit: int = 20) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT thread_id, question, status, cost_usd, updated_at FROM runs ORDER BY updated_at DESC LIMIT %s",
            [limit],
        )
        return await cur.fetchall()
