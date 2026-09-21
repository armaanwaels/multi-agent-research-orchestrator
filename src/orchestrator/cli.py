"""Command line: seed the stores, ask a question, review evidence, resume runs."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any

from .config import Settings


def _review_in_terminal(payload: dict[str, Any]) -> dict[str, Any] | None:
    print("\n--- Review before synthesis ---")
    print(f"Question: {payload['question']}\n")
    print("Plan:")
    for i, step in enumerate(payload["plan"], 1):
        print(f"  {i}. [{step['agent']}] {step['instruction']}")
    print("\nAgent notes:")
    for note in payload["notes"]:
        print(f"  - {note}")
    print(f"\nEvidence ({len(payload['evidence'])} items):")
    for e in payload["evidence"]:
        print(f"  {e['id']:>4}  {e['source']:<9} {e['title'][:80]}")
    while True:
        choice = input("\n[a]pprove  [e]dit  [r]eject  [q]uit and resume later > ").strip().lower()
        if choice in ("a", "approve", ""):
            return {"action": "approve"}
        if choice in ("e", "edit"):
            raw = input("Evidence ids to drop (comma separated): ")
            drop = [x.strip().upper() for x in raw.split(",") if x.strip()]
            note = input("Instruction for the writer (optional): ").strip()
            return {"action": "edit", "drop": drop, "note": note}
        if choice in ("r", "reject"):
            return {"action": "reject", "note": input("What is missing or wrong? ").strip()}
        if choice in ("q", "quit"):
            return None


async def _terminal_review(payload: dict[str, Any]) -> dict[str, Any] | None:
    return await asyncio.to_thread(_review_in_terminal, payload)


def _print_result(result: dict[str, Any], cost: float) -> None:
    if result["status"] == "awaiting_review":
        print(f"\nRun paused. Resume with: orchestrator resume {result['thread_id']}")
        return
    print("\n" + result["answer"] + "\n")
    cited = {e["id"]: e for e in result["evidence"]}
    for eid in sorted(cited, key=lambda x: int(x[1:])):
        if f"[{eid}]" in result["answer"]:
            e = cited[eid]
            print(f"  [{eid}] {e['title'][:90]}  {e['url']}")
    print(f"\nthread {result['thread_id']}  cost ${cost:.4f}")


async def _ask(question: str, auto_approve: bool, persistent: bool) -> None:
    from .research import research
    from .runtime import open_runtime

    async with open_runtime(persistent=persistent, auto_approve=auto_approve) as rt:
        result = await research(rt, question, str(uuid.uuid4()), _terminal_review)
        _print_result(result, rt.deps.llm.ledger.cost)


async def _resume(thread_id: str) -> None:
    from .research import resume
    from .runtime import open_runtime

    async with open_runtime(persistent=True) as rt:
        result = await resume(rt, thread_id, _terminal_review)
        _print_result(result, rt.deps.llm.ledger.cost)


async def _runs() -> None:
    from .runs import PostgresRunStore

    async with PostgresRunStore.connect(Settings().postgres_url) as store:
        for r in await store.recent():
            when = f"{r['updated_at']:%Y-%m-%d %H:%M}"
            print(f"{when}  {r['status']:<16} ${r['cost_usd']:.3f}  {r['thread_id']}  {r['question'][:60]}")


async def _seed() -> None:
    from qdrant_client import AsyncQdrantClient

    from .sources.sql import build_database
    from .sources.vectors import VectorStore

    s = Settings()
    n = build_database(s.data_dir / "owid_energy.csv", s.sqlite_path)
    print(f"sqlite: {n} rows -> {s.sqlite_path}")
    qdrant = AsyncQdrantClient(url=s.qdrant_url)
    try:
        n = await VectorStore(qdrant, s.qdrant_collection, s.embedding_model).index_abstracts(
            s.data_dir / "arxiv_abstracts.jsonl"
        )
        print(f"qdrant: {n} abstracts -> {s.qdrant_url}/{s.qdrant_collection}")
    finally:
        await qdrant.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed", help="build the SQLite database and the Qdrant index")
    ask = sub.add_parser("ask", help="research a question")
    ask.add_argument("question")
    ask.add_argument("--auto-approve", action="store_true", help="skip the human review step")
    ask.add_argument("--no-db", action="store_true", help="keep state in memory instead of PostgreSQL")
    res = sub.add_parser("resume", help="continue a run that is waiting for review")
    res.add_argument("thread_id")
    sub.add_parser("runs", help="list recent runs")
    args = p.parse_args()

    if args.cmd == "seed":
        asyncio.run(_seed())
    elif args.cmd == "ask":
        asyncio.run(_ask(args.question, args.auto_approve, persistent=not args.no_db))
    elif args.cmd == "resume":
        asyncio.run(_resume(args.thread_id))
    elif args.cmd == "runs":
        asyncio.run(_runs())


if __name__ == "__main__":
    main()
