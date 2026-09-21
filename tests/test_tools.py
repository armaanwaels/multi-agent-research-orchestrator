import json
import sqlite3

import mcp
import pytest

from orchestrator.agents import SPECIALISTS, run_specialist
from orchestrator.mcp_server import build_server
from orchestrator.sources.docs import DocStore
from orchestrator.sources.sql import SqlStore
from orchestrator.sources.text_search import BM25, Passage
from orchestrator.sources.web import FixtureWebSearch
from orchestrator.tools import Toolbox, local_tools, mcp_tools

from .conftest import DATA, FakeVectors, ScriptedLLM, response, text, tool_use


def test_bm25_ranks_matching_passage_first():
    index = BM25([Passage("a", "Wind", "", "turbines convert wind"), Passage("b", "Coal", "", "coal is burned")])
    assert index.search("wind turbines")[0][1].doc_id == "a"
    assert index.search("uranium") == []


def test_doc_search_returns_sections_with_source_url():
    hits = DocStore(DATA / "docs").search("capacity factor nuclear", k=2)
    assert hits and hits[0]["url"].startswith("https://www.eia.gov/")


def test_sql_is_read_only(db_path):
    sql = SqlStore(db_path)
    assert sql.query("SELECT COUNT(*) FROM energy")["rows"][0][0] > 7000
    with pytest.raises(ValueError):
        sql.query("DELETE FROM energy")
    with pytest.raises(sqlite3.OperationalError):  # the connection itself is read-only
        sql.query("WITH x AS (SELECT 1) INSERT INTO energy(country) SELECT 'x' FROM x")


async def test_web_fixture_search():
    hits = await FixtureWebSearch(DATA / "web_fixture.jsonl").search("wind farm", k=2)
    assert len(hits) == 2 and all(h["url"].startswith("https://www.eia.gov/todayinenergy/") for h in hits)


async def test_mcp_server_exposes_doc_and_sql_tools(db_path):
    async with mcp.Client(build_server(DATA / "docs", db_path)) as client:
        specs = {t.name: t for t in await mcp_tools(client)}
        assert set(specs) == {"list_docs", "search_docs", "read_doc", "list_tables", "describe_table", "run_sql"}
        out = json.loads(await specs["run_sql"].handler({"sql_query": "SELECT 1 AS one"}))
        assert out["rows"] == [[1]]
        with pytest.raises(RuntimeError):
            await specs["run_sql"].handler({"sql_query": "DROP TABLE energy"})


async def test_parallel_tool_calls_get_sequential_evidence_ids(db_path):
    async with mcp.Client(build_server(DATA / "docs", db_path)) as client:
        toolbox = Toolbox(
            await mcp_tools(client) + local_tools(FixtureWebSearch(DATA / "web_fixture.jsonl"), None, FakeVectors())
        )

        def script(role, kwargs):
            if len(kwargs["messages"]) == 1:
                return response(
                    tool_use("search_docs", {"query": "solar", "k": 2}, "t1"),
                    tool_use("paper_search", {"query": "perovskite"}, "t2"),
                    stop_reason="tool_use",
                )
            return response(text("Found things [E5] [E7]."))

        note, evidence = await run_specialist(
            ScriptedLLM(script),
            "m",
            toolbox,
            SPECIALISTS["documents"],
            "q",
            "find solar facts",
            first_id=5,
            max_turns=3,
        )
    assert [e["id"] for e in evidence] == ["E5", "E6", "E7"]
    assert [e["source"] for e in evidence] == ["docs", "docs", "arxiv"]
    assert "[E5]" in note


async def test_tool_errors_are_returned_to_the_model(db_path):
    async with mcp.Client(build_server(DATA / "docs", db_path)) as client:
        toolbox = Toolbox(await mcp_tools(client))
        seen = []

        def script(role, kwargs):
            if len(kwargs["messages"]) == 1:
                return response(tool_use("run_sql", {"sql_query": "DELETE FROM energy"}, "t1"), stop_reason="tool_use")
            seen.append(kwargs["messages"][-1]["content"][0])
            return response(text("The query failed."))

        _, evidence = await run_specialist(
            ScriptedLLM(script), "m", toolbox, SPECIALISTS["data"], "q", "i", first_id=1, max_turns=3
        )
    assert evidence == [] and seen[0]["is_error"] is True
