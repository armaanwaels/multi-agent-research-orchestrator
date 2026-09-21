"""Run one research question through the graph, pausing for review when asked."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import Command

from .graph import build_graph, initial_state
from .runtime import Runtime

# Returns the reviewer's decision, or None to leave the run paused.
ReviewFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


def _pending_review(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    return interrupts[0].value if interrupts else None


async def drive(
    rt: Runtime, thread_id: str, first_input: Any, review: ReviewFn | None, question: str
) -> dict[str, Any]:
    graph = build_graph(rt.checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    start = time.monotonic()
    ledger = rt.deps.llm.ledger
    reported = ledger.cost  # cost already recorded for this run by this process

    def new_cost() -> float:
        nonlocal reported
        delta, reported = ledger.cost - reported, ledger.cost
        return delta

    result = await graph.ainvoke(first_input, config, context=rt.deps)
    while (payload := _pending_review(result)) is not None:
        await rt.runs.upsert(thread_id, question, status="awaiting_review", cost_usd=new_cost())
        decision = await review(payload) if review else None
        if decision is None:
            return {"status": "awaiting_review", "thread_id": thread_id, "review": payload}
        result = await graph.ainvoke(Command(resume=decision), config, context=rt.deps)
    await rt.runs.upsert(
        thread_id,
        question,
        status="done",
        answer=result["answer"],
        evidence=result["evidence"],
        cost_usd=new_cost(),
    )
    return {
        "status": "done",
        "thread_id": thread_id,
        "answer": result["answer"],
        "evidence": result["evidence"],
        "plan": result["plan"],
        "notes": result["notes"],
        "seconds": time.monotonic() - start,
    }


async def research(rt: Runtime, question: str, thread_id: str, review: ReviewFn | None = None) -> dict[str, Any]:
    await rt.runs.upsert(thread_id, question, status="running")
    return await drive(rt, thread_id, initial_state(question), review, question)


async def resume(rt: Runtime, thread_id: str, review: ReviewFn) -> dict[str, Any]:
    """Continue a run that was left waiting for review, possibly in another process."""
    graph = build_graph(rt.checkpointer)
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.next:
        raise ValueError(f"run {thread_id} is not waiting for review")
    payload = snapshot.tasks[0].interrupts[0].value
    decision = await review(payload)
    if decision is None:
        return {"status": "awaiting_review", "thread_id": thread_id, "review": payload}
    return await drive(rt, thread_id, Command(resume=decision), review, snapshot.values["question"])
