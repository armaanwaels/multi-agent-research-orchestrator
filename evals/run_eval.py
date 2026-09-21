"""Run the synthesis tasks through the full system and score them.

    uv run python evals/run_eval.py                 # all tasks
    uv run python evals/run_eval.py --limit 2       # quick pilot
    uv run python evals/run_eval.py --ids t04-caiso-solar t21-world-solar-wind

Needs the Qdrant and PostgreSQL containers running and `orchestrator seed` done.
Human review is auto-approved. Web search uses the committed fixture.
Writes results/eval_results.jsonl and results/eval_summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from judge import judge_citations, judge_relevance  # noqa: E402

from orchestrator.config import Settings  # noqa: E402
from orchestrator.llm import AnthropicLLM  # noqa: E402
from orchestrator.research import research  # noqa: E402
from orchestrator.runtime import open_runtime  # noqa: E402

RESULTS = ROOT / "results"


async def run_task(task: dict, settings: Settings, judge_llm: AnthropicLLM) -> dict:
    llm = AnthropicLLM()
    start = time.monotonic()
    async with open_runtime(settings, persistent=True, auto_approve=True, llm=llm) as rt:
        out = await research(rt, task["question"], f"eval-{task['id']}-{int(time.time())}")
    seconds = time.monotonic() - start
    before = judge_llm.ledger.cost
    relevance, citations = await asyncio.gather(
        judge_relevance(judge_llm, settings.judge_model, task["question"], task["reference"], out["answer"]),
        judge_citations(judge_llm, settings.judge_model, out["answer"], out["evidence"]),
    )
    return {
        "id": task["id"],
        "question": task["question"],
        "answer": out["answer"],
        "plan": out["plan"],
        "sources_used": sorted({e["source"] for e in out["evidence"]}),
        "sources_expected": task["sources"],
        "evidence_items": len(out["evidence"]),
        "relevance": relevance["score"],
        "relevance_reasoning": relevance["reasoning"],
        "citations": citations,
        "system_cost_usd": llm.ledger.cost,
        "system_cost_by_role": llm.ledger.by_role(),
        "judge_cost_usd": judge_llm.ledger.cost - before,
        "seconds": round(seconds, 1),
    }


def summarize(rows: list[dict], settings: Settings) -> str:
    n = len(rows)
    rel = [r["relevance"] for r in rows]
    cites = sum(r["citations"]["citations"] for r in rows)
    supported = sum(r["citations"]["supported"] for r in rows)
    invalid = sum(r["citations"]["invalid"] for r in rows)
    covered = sum(1 for r in rows if set(r["sources_expected"]) <= set(r["sources_used"]))
    system_cost = sum(r["system_cost_usd"] for r in rows)
    judge_cost = sum(r["judge_cost_usd"] for r in rows)
    lines = [
        f"Tasks: {n}. Models: planner {settings.planner_model}, agents {settings.worker_model}, "
        f"synthesizer {settings.synth_model}, judge {settings.judge_model}. Web search: {settings.web_search}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Relevance, mean judge score (1-5) | {statistics.mean(rel):.2f} |",
        f"| Relevance, share of tasks scoring 4 or 5 | {sum(s >= 4 for s in rel) / n:.0%} |",
        f"| Relevance, share of tasks scoring 5 | {sum(s == 5 for s in rel) / n:.0%} |",
        f"| Citation accuracy (supported / all citations) | {supported / cites:.0%} ({supported}/{cites}) |",
        f"| Citations to evidence ids that do not exist | {invalid} |",
        f"| Tasks where every expected source type was used | {covered}/{n} |",
        f"| Median time per task | {statistics.median(r['seconds'] for r in rows):.0f} s |",
        f"| System cost, total / per task | ${system_cost:.2f} / ${system_cost / n:.3f} |",
        f"| Judge cost, total | ${judge_cost:.2f} |",
        "",
        "| Task | Relevance | Citations supported | Sources used | Cost |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["citations"]
        lines.append(
            f"| {r['id']} | {r['relevance']} | {c['supported']}/{c['citations']} | "
            f"{', '.join(r['sources_used'])} | ${r['system_cost_usd']:.3f} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="eval")
    args = p.parse_args()

    tasks = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())
    if args.ids:
        tasks = [t for t in tasks if t["id"] in args.ids]
    if args.limit:
        tasks = tasks[: args.limit]
    # The eval always uses the fixture so results do not depend on live web search.
    settings = replace(Settings(), web_search="fixture")
    os.environ["WEB_SEARCH_BACKEND"] = "fixture"
    judge_llm = AnthropicLLM()
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(t):
        async with sem:
            try:
                row = await run_task(t, settings, judge_llm)
            except Exception as exc:  # record the failure and keep going
                print(f"{t['id']}: FAILED {exc!r}", flush=True)
                return None
            print(f"{t['id']}: relevance {row['relevance']}, cost ${row['system_cost_usd']:.3f}", flush=True)
            return row

    rows = [r for r in await asyncio.gather(*(guarded(t) for t in tasks)) if r]
    rows.sort(key=lambda r: r["id"])
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{args.out}_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows, settings)
    if len(rows) < len(tasks):
        summary = f"{len(tasks) - len(rows)} of {len(tasks)} tasks failed and are excluded.\n\n" + summary
    (RESULTS / f"{args.out}_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
