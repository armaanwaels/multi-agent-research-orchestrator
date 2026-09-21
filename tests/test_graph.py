"""Routing, human review, and one end-to-end run with the LLM stubbed out."""

from contextlib import asynccontextmanager

import mcp
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orchestrator.config import Settings
from orchestrator.graph import Deps, build_graph, initial_state, supervisor
from orchestrator.mcp_server import build_server
from orchestrator.research import research, resume
from orchestrator.runs import MemoryRunStore
from orchestrator.runtime import Runtime
from orchestrator.sources.web import FixtureWebSearch
from orchestrator.tools import Toolbox, local_tools, mcp_tools

from .conftest import DATA, FakeVectors, ScriptedLLM, plan_json, response, text, tool_use


def test_supervisor_routes_plan_steps_in_order_then_to_review():
    state = initial_state("q")
    state["plan"] = [{"agent": "data", "instruction": "a"}, {"agent": "search", "instruction": "b"}]
    assert supervisor(state).goto == "data"
    state["step_index"] = 1
    assert supervisor(state).goto == "search"
    state["step_index"] = 2
    assert supervisor(state).goto == "human_review"


def research_script(role, kwargs):
    """A planner that asks for data then documents; specialists make one tool call each."""
    if role == "planner":
        return response(text(plan_json(("data", "German solar share 2024"), ("documents", "how PV works"))))
    if role == "agent.data":
        if len(kwargs["messages"]) == 1:
            sql = "SELECT year, solar_share_elec FROM energy WHERE country = 'Germany' AND year = 2024"
            return response(tool_use("run_sql", {"sql_query": sql}, "d1"), stop_reason="tool_use")
        return response(text("Germany solar share 2024 [E1]."))
    if role == "agent.documents":
        if len(kwargs["messages"]) == 1:
            return response(
                tool_use("search_docs", {"query": "photovoltaic cells", "k": 2}, "c1"), stop_reason="tool_use"
            )
        return response(text("PV cells convert sunlight [E2]."))
    if role == "synthesizer":
        return response(
            text("Germany got about 15% of its electricity from solar in 2024 [E1]. PV cells convert light [E2].")
        )
    raise AssertionError(role)


@asynccontextmanager
async def scripted_runtime(db_path):
    # A context manager rather than a fixture: the MCP client must close in the task that opened it.
    async with mcp.Client(build_server(DATA / "docs", db_path)) as client:
        tools = await mcp_tools(client) + local_tools(FixtureWebSearch(DATA / "web_fixture.jsonl"), None, FakeVectors())
        deps = Deps(llm=ScriptedLLM(research_script), toolbox=Toolbox(tools), settings=Settings())
        yield Runtime(deps, InMemorySaver(), MemoryRunStore())


async def test_end_to_end_pauses_for_review_then_answers(db_path):
    async with scripted_runtime(db_path) as runtime:
        paused = await research(runtime, "How much solar did Germany use in 2024, and how does PV work?", "t1")
        assert paused["status"] == "awaiting_review"
        review = paused["review"]
        assert [s["agent"] for s in review["plan"]] == ["data", "documents"]
        assert review["evidence"][0]["id"] == "E1" and review["evidence"][0]["source"] == "sql"
        assert runtime.runs.rows["t1"]["status"] == "awaiting_review"

        async def approve(payload):
            return {"action": "approve"}

        done = await resume(runtime, "t1", approve)
        assert done["status"] == "done"
        assert "[E1]" in done["answer"]
        assert runtime.runs.rows["t1"]["status"] == "done"
        roles = [r for r, _ in runtime.deps.llm.requests]
        assert roles[0] == "planner" and roles[-1] == "synthesizer"


async def test_review_can_drop_evidence_and_instruct_the_writer(db_path):
    async with scripted_runtime(db_path) as runtime:
        graph = build_graph(runtime.checkpointer)
        config = {"configurable": {"thread_id": "t2"}}
        await graph.ainvoke(initial_state("q"), config, context=runtime.deps)
        result = await graph.ainvoke(
            Command(resume={"action": "edit", "drop": ["E2"], "note": "Be brief."}), config, context=runtime.deps
        )
        assert [e["id"] for e in result["evidence"]] == ["E1", "E3"]
        synth_prompt = runtime.deps.llm.requests[-1][1]["messages"][0]["content"]
        assert "[E2]" not in synth_prompt and "Be brief." in synth_prompt


async def test_reject_sends_the_run_back_to_the_planner_with_feedback(db_path):
    async with scripted_runtime(db_path) as runtime:
        graph = build_graph(runtime.checkpointer)
        config = {"configurable": {"thread_id": "t3"}}
        await graph.ainvoke(initial_state("q"), config, context=runtime.deps)
        result = await graph.ainvoke(
            Command(resume={"action": "reject", "note": "Need 2023 too."}), config, context=runtime.deps
        )
        assert "__interrupt__" in result  # replanned, gathered again, and paused for a second review
        planner_calls = [kw for role, kw in runtime.deps.llm.requests if role == "planner"]
        assert len(planner_calls) == 2 and "Need 2023 too." in planner_calls[1]["messages"][0]["content"]
        assert result["revisions"] == 1


async def test_auto_approve_runs_straight_through(db_path):
    async with scripted_runtime(db_path) as runtime:
        runtime.deps.auto_approve = True
        done = await research(runtime, "q", "t4")
        assert done["status"] == "done" and done["evidence"]


async def test_planner_prompt_carries_the_source_catalog(db_path):
    async with scripted_runtime(db_path) as runtime:
        runtime.deps.catalog = "What the sources contain: CATALOG-MARKER"
        runtime.deps.auto_approve = True
        await research(runtime, "q", "t5")
        planner_system = next(kw["system"] for role, kw in runtime.deps.llm.requests if role == "planner")
    assert "CATALOG-MARKER" in planner_system
