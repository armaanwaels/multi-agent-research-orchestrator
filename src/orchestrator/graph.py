"""The LangGraph supervisor graph.

    planner -> supervisor -> {search | documents | data} -> supervisor -> ... -> human_review -> synthesizer

The supervisor walks the planner's steps in order and hands control to one
specialist at a time. Before synthesis the graph interrupts so a person can
approve the gathered evidence, drop items, add an instruction, or reject the
research and send it back to the planner with feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from . import agents
from .agents import SPECIALISTS, Step
from .config import Settings
from .llm import LLM
from .tools import Evidence, Toolbox
from .tracing import span


class ResearchState(TypedDict, total=False):
    question: str
    plan: list[Step]
    step_index: int
    evidence: list[Evidence]
    notes: list[str]
    feedback: str
    reviewer_note: str
    revisions: int
    answer: str


@dataclass
class Deps:
    llm: LLM
    toolbox: Toolbox
    settings: Settings
    # Evals and scripted runs approve automatically; the CLI leaves this False.
    auto_approve: bool = False


def initial_state(question: str) -> ResearchState:
    return ResearchState(
        question=question, plan=[], step_index=0, evidence=[], notes=[], feedback="", reviewer_note="", revisions=0
    )


async def planner(state: ResearchState, runtime: Runtime[Deps]) -> dict[str, Any]:
    d = runtime.context
    with span("node.planner"):
        steps = await agents.plan(
            d.llm, d.settings.planner_model, state["question"], state.get("feedback", ""), state.get("notes")
        )
    return {"plan": steps, "step_index": 0}


def supervisor(state: ResearchState) -> Command:
    i = state["step_index"]
    if i < len(state["plan"]):
        return Command(goto=state["plan"][i]["agent"])
    return Command(goto="human_review")


def specialist_node(name: str):
    async def node(state: ResearchState, runtime: Runtime[Deps]) -> dict[str, Any]:
        d = runtime.context
        step = state["plan"][state["step_index"]]
        with span(f"node.{name}", instruction=step["instruction"]):
            note, found = await agents.run_specialist(
                d.llm,
                d.settings.worker_model,
                d.toolbox,
                SPECIALISTS[name],
                state["question"],
                step["instruction"],
                first_id=len(state["evidence"]) + 1,
                max_turns=d.settings.max_agent_turns,
            )
        return {
            "evidence": state["evidence"] + found,
            "notes": state["notes"] + [f"{name}: {note}"],
            "step_index": state["step_index"] + 1,
        }

    node.__name__ = name
    return node


def human_review(state: ResearchState, runtime: Runtime[Deps]) -> Command:
    if runtime.context.auto_approve:
        decision: dict[str, Any] = {"action": "approve"}
    else:
        # Execution stops here and the state is checkpointed. `Command(resume=decision)` continues it,
        # from this process or any other that shares the checkpointer.
        decision = interrupt(
            {
                "question": state["question"],
                "plan": state["plan"],
                "notes": state["notes"],
                "evidence": [
                    {"id": e["id"], "source": e["source"], "title": e["title"], "url": e["url"]}
                    for e in state["evidence"]
                ],
            }
        )
    action = decision.get("action", "approve")
    if action == "reject" and state["revisions"] < runtime.context.settings.max_revisions:
        return Command(
            goto="planner",
            update={"feedback": decision.get("note", ""), "revisions": state["revisions"] + 1},
        )
    drop = set(decision.get("drop", []))
    notes = state["notes"]
    for eid in drop:
        # Notes cite evidence too; unhook dropped ids so the writer cannot cite them.
        notes = [n.replace(f"[{eid}]", "[removed by reviewer]") for n in notes]
    return Command(
        goto="synthesizer",
        update={
            "evidence": [e for e in state["evidence"] if e["id"] not in drop],
            "notes": notes,
            "reviewer_note": decision.get("note", ""),
        },
    )


async def synthesizer(state: ResearchState, runtime: Runtime[Deps]) -> dict[str, Any]:
    d = runtime.context
    with span("node.synthesizer"):
        answer = await agents.synthesize(
            d.llm, d.settings.synth_model, state["question"], state["evidence"], state["notes"], state["reviewer_note"]
        )
    return {"answer": answer}


def build_graph(checkpointer=None):
    g = StateGraph(ResearchState, context_schema=Deps)
    g.add_node("planner", planner)
    g.add_node("supervisor", supervisor, destinations=("search", "documents", "data", "human_review"))
    for name in SPECIALISTS:
        g.add_node(name, specialist_node(name))
        g.add_edge(name, "supervisor")
    g.add_node("human_review", human_review, destinations=("planner", "synthesizer"))
    g.add_node("synthesizer", synthesizer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "supervisor")
    return g.compile(checkpointer=checkpointer)
