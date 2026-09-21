"""Run records and checkpoints against a real PostgreSQL. Skipped when none is reachable."""

import os
import uuid

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from orchestrator.config import Settings
from orchestrator.runs import PostgresRunStore

from .test_graph import scripted_runtime

URL = os.environ.get("POSTGRES_URL", Settings().postgres_url)


def _reachable() -> bool:
    try:
        psycopg.connect(URL, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="PostgreSQL not reachable")


async def test_run_record_upserts_across_status_changes():
    tid = str(uuid.uuid4())
    async with PostgresRunStore.connect(URL) as store:
        await store.upsert(tid, "q", status="running")
        await store.upsert(tid, "q", status="awaiting_review", cost_usd=0.5)
        await store.upsert(tid, "q", status="done", answer="a", evidence=[{"id": "E1"}], cost_usd=1.0)
        row = next(r for r in await store.recent(100) if r["thread_id"] == tid)
        await store.conn.execute("DELETE FROM runs WHERE thread_id = %s", [tid])
    # Costs from the paused and resumed halves add up.
    assert row["status"] == "done" and row["cost_usd"] == 1.5


async def test_paused_run_resumes_from_a_fresh_checkpointer(db_path):
    from langgraph.types import Command

    from orchestrator.graph import build_graph, initial_state

    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    async with scripted_runtime(db_path) as rt:
        async with AsyncPostgresSaver.from_conn_string(URL) as saver:
            await saver.setup()
            paused = await build_graph(saver).ainvoke(initial_state("q"), config, context=rt.deps)
            assert "__interrupt__" in paused
        # A second connection stands in for a second process picking the run up later.
        async with AsyncPostgresSaver.from_conn_string(URL) as saver:
            done = await build_graph(saver).ainvoke(Command(resume={"action": "approve"}), config, context=rt.deps)
    assert done["answer"].startswith("Germany")
