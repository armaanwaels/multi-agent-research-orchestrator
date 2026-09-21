"""LLM-as-judge scoring: answer relevance against a reference, and citation accuracy.

Relevance is a 1-5 rubric score. Citation accuracy has two parts:
  valid     - the cited id exists in the evidence the answer was written from (checked in code)
  supported - the cited evidence actually supports the sentence that cites it (checked by the judge)
"""

from __future__ import annotations

import json
import re
from typing import Any

from orchestrator.llm import LLM, text_of
from orchestrator.tools import Evidence

RELEVANCE_RUBRIC = """You grade answers from a research assistant against a reference answer.

Score 1-5:
5 - Answers every part of the question; every key fact in the reference is present and correct.
4 - Answers every part; misses or is vague on a minor fact, and contradicts nothing in the reference.
3 - Answers most of the question, or gets one key fact wrong.
2 - Misses a main part of the question, or gets several key facts wrong.
1 - Does not answer the question.

Extra correct detail beyond the reference is fine. A different but equivalent rounding is correct.
Numbers that disagree with the reference by more than rounding count as wrong."""

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}

SUPPORT_PROMPT = """For each numbered claim below, decide whether the cited evidence supports it.
A claim is supported if the evidence states it or it follows directly from numbers in the evidence
(simple arithmetic such as differences, ratios or percentages counts). Judge only the cited evidence."""

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "integer"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

_CITE = re.compile(r"\[(E\d+(?:\s*[,–-]\s*E?\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def expand(group: str) -> list[str]:
    """'E3, E5' -> [E3, E5]; 'E3–E6' -> [E3, E4, E5, E6]."""
    ids: list[str] = []
    for part in re.split(r"\s*,\s*", group):
        m = re.fullmatch(r"E(\d+)\s*[–-]\s*E?(\d+)", part)
        if m:
            ids += [f"E{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        elif part:
            ids.append(part if part.startswith("E") else f"E{part}")
    return ids


def extract_citations(answer: str) -> list[tuple[str, list[str]]]:
    """Split an answer into sentences and return (sentence, cited ids) for each sentence that cites."""
    out = []
    for sentence in _SENTENCE.split(answer.replace("\n", " ")):
        ids = [i for g in _CITE.findall(sentence) for i in expand(g)]
        if ids:
            out.append((sentence.strip(), ids))
    return out


async def judge_relevance(llm: LLM, model: str, question: str, reference: str, answer: str) -> dict[str, Any]:
    resp = await llm.create(
        role="judge.relevance",
        model=model,
        system=RELEVANCE_RUBRIC,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade:\n{answer}",
            }
        ],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": RELEVANCE_SCHEMA}},
    )
    return json.loads(text_of(resp))


async def judge_citations(llm: LLM, model: str, answer: str, evidence: list[Evidence]) -> dict[str, Any]:
    by_id = {e["id"]: e for e in evidence}
    claims = extract_citations(answer)
    total = sum(len(ids) for _, ids in claims)
    invalid = sum(1 for _, ids in claims for i in ids if i not in by_id)
    checkable = [(s, [i for i in ids if i in by_id]) for s, ids in claims]
    checkable = [(s, ids) for s, ids in checkable if ids]
    supported = 0
    if checkable:
        blocks = []
        for n, (sentence, ids) in enumerate(checkable, 1):
            ev = "\n".join(f"  [{i}] {by_id[i]['title']}\n  {by_id[i]['content'][:1500]}" for i in ids)
            blocks.append(f"Claim {n}: {sentence}\nCited evidence:\n{ev}")
        resp = await llm.create(
            role="judge.citations",
            model=model,
            system=SUPPORT_PROMPT,
            messages=[{"role": "user", "content": "\n\n".join(blocks)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SUPPORT_SCHEMA}},
        )
        verdicts = {v["claim"]: v["supported"] for v in json.loads(text_of(resp))["verdicts"]}
        # A supported claim credits each valid id it cites.
        supported = sum(len(ids) for n, (_, ids) in enumerate(checkable, 1) if verdicts.get(n))
    return {
        "citations": total,
        "invalid": invalid,
        "supported": supported,
        "cited_sentences": len(claims),
        "accuracy": supported / total if total else None,
    }
