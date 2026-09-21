"""The five agents: planner, three retrieval specialists, and synthesizer.

The supervisor that routes between them is plain code in graph.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from .llm import LLM, text_of
from .tools import Evidence, Toolbox, render

AgentName = Literal["search", "documents", "data"]


class Step(TypedDict):
    agent: AgentName
    instruction: str


@dataclass(frozen=True)
class Specialist:
    name: AgentName
    tools: list[str]
    brief: str


SPECIALISTS: dict[str, Specialist] = {
    "search": Specialist(
        "search",
        ["web_search", "read_article", "wikipedia_search", "wikipedia_page"],
        "You search the web (recent energy news and analysis) and Wikipedia (background, definitions, history). "
        "Search results are short snippets; open an article with read_article when the detail you need is missing.",
    ),
    "documents": Specialist(
        "documents",
        ["list_docs", "search_docs", "read_doc", "paper_search"],
        "You retrieve from U.S. EIA reference documents (how energy sources work, U.S. facts) "
        "and from a vector index of arXiv research abstracts.",
    ),
    "data": Specialist(
        "data",
        ["list_tables", "describe_table", "run_sql"],
        "You query a SQLite database of Our World in Data energy statistics: electricity generation "
        "by source, shares, emissions and consumption by country and year. "
        "Always call describe_table before your first query: it gives column meanings and the exact year range. "
        "When a question asks about a recent year, query that year directly; do not assume data is missing "
        "until a query returns no rows.",
    ),
}


# ---------------------------------------------------------------- planner

PLANNER_SYSTEM = """You plan research for a team of specialist agents answering questions about energy.

Specialists:
- search: web search (recent EIA analysis articles) and Wikipedia.
- documents: U.S. EIA "Energy Explained" reference pages and a vector index of arXiv abstracts.
- data: SQL over Our World in Data energy statistics by country and year (2000 onward).

Write the smallest plan that covers every part of the question: one to four steps, each naming one
specialist and a specific instruction (what to find, and which numbers or years matter). Only include a
specialist whose sources can contribute. Steps run in order."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": ["search", "documents", "data"]},
                    "instruction": {"type": "string"},
                },
                "required": ["agent", "instruction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}


async def build_catalog(toolbox: Toolbox, web_backend: str) -> str:
    """Describe what each source actually contains, from the live tools, for the planner's prompt."""
    docs = json.loads((await toolbox.call("list_docs", {}))[0])
    cols = json.loads((await toolbox.call("describe_table", {"table": "energy"}))[0])
    years = next(c["meaning"] for c in cols if c["column"] == "year")
    names = [c["column"] for c in cols]
    absent = [x for x in ("geothermal", "biomass", "bioenergy") if not any(x in n for n in names)]
    web = (
        "40 EIA 'Today in Energy' articles from June to September 2026, mostly about U.S. energy markets"
        if web_backend == "fixture"
        else "live web search"
    )
    return (
        "What the sources contain:\n"
        f"- documents: EIA reference pages: {'; '.join(d['title'] for d in docs)}. "
        "Also 242 arXiv abstracts on solar, wind, storage, hydrogen, nuclear and electricity markets.\n"
        f"- data: table `energy`, {years.split(' (')[0].lower()}, one row per country or region per year. "
        f"Columns: {', '.join(names)}."
        + (f" No column for {', '.join(absent)}.\n" if absent else "\n")
        + f"- search: web search over {web}; Wikipedia for everything else.\n"
        "Use these ranges as given; do not tell a specialist that data may stop earlier."
    )


async def plan(
    llm: LLM,
    model: str,
    question: str,
    feedback: str = "",
    gathered: list[str] | None = None,
    catalog: str = "",
) -> list[Step]:
    prompt = f"Question: {question}"
    if gathered:
        prompt += "\n\nFindings so far:\n" + "\n".join(gathered)
    if feedback:
        prompt += (
            f"\n\nA human reviewer rejected the previous research with this feedback:\n{feedback}\n"
            "Plan only the additional steps needed."
        )
    resp = await llm.create(
        role="planner",
        model=model,
        system=f"{PLANNER_SYSTEM}\n\n{catalog}" if catalog else PLANNER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
    )
    steps = json.loads(text_of(resp))["steps"]
    return [Step(agent=s["agent"], instruction=s["instruction"]) for s in steps[:4]]


# ---------------------------------------------------------------- specialists

SPECIALIST_SYSTEM = """{brief}

You are one member of a research team. Use your tools to gather evidence for your instruction.
Every retrieved result is labeled with an evidence id like [E4]. Prefer primary numbers over commentary.
When you have enough, or after a few tool calls at most, reply with a short note (under 120 words) that
lists the findings relevant to the question, each followed by its evidence id. Report what you found and
what you could not find; do not write the final answer."""


async def run_specialist(
    llm: LLM,
    model: str,
    toolbox: Toolbox,
    spec: Specialist,
    question: str,
    instruction: str,
    first_id: int,
    max_turns: int,
) -> tuple[str, list[Evidence]]:
    tools = toolbox.subset(spec.tools)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Question being researched: {question}\n\nYour instruction: {instruction}"}
    ]
    evidence: list[Evidence] = []
    for turn in range(max_turns):
        last = turn == max_turns - 1
        resp = await llm.create(
            role=f"agent.{spec.name}",
            model=model,
            system=SPECIALIST_SYSTEM.format(brief=spec.brief),
            messages=messages,
            tools=[t.api_schema() for t in tools],
            # On the last turn, force a written note instead of another tool call.
            tool_choice={"type": "none"} if last else {"type": "auto"},
            output_config={"effort": "low"},
        )
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            return text_of(resp).strip(), evidence
        messages.append({"role": "assistant", "content": resp.content})

        async def run(block):
            try:
                return await toolbox.call(block.name, block.input)
            except Exception as exc:  # tool errors go back to the model, not up the stack
                return exc, None

        # Calls in one turn run concurrently; ids are assigned afterwards in call order.
        outcomes = await asyncio.gather(*(run(b) for b in calls))
        results = []
        for block, (text, items) in zip(calls, outcomes, strict=True):
            if isinstance(text, Exception):
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": f"error: {text}", "is_error": True}
                )
                continue
            if items is not None:
                for item in items:
                    item["id"] = f"E{first_id + len(evidence)}"
                    evidence.append(item)
                text = render(items)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
        messages.append({"role": "user", "content": results})
    return "(no summary)", evidence


# ---------------------------------------------------------------- synthesizer

SYNTH_SYSTEM = """You write the final answer for a research team, using only the evidence provided.

Rules:
- Put the evidence id in square brackets right after each claim it supports, e.g. "Solar produced 304 TWh in 2024 [E3]."
  Cite only ids that appear in the evidence list, and only for claims that evidence actually supports.
- Every number must come from the evidence. Give units and years.
- If the evidence is missing, thin or conflicting for part of the question, say so plainly.
- Answer the question directly in the first sentence. Keep it under 250 words. No headings."""


def format_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e['id']}] ({e['source']}) {e['title']} {e['url']}\n{e['content']}" for e in evidence)


async def synthesize(
    llm: LLM, model: str, question: str, evidence: list[Evidence], notes: list[str], reviewer_note: str = ""
) -> str:
    prompt = f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}\n\nAgent notes:\n" + "\n".join(notes)
    if reviewer_note:
        prompt += f"\n\nInstruction from the human reviewer: {reviewer_note}"
    resp = await llm.create(
        role="synthesizer",
        model=model,
        system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium"},
    )
    return text_of(resp).strip()
