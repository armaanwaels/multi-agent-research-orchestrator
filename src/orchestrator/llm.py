"""Thin wrapper over the Anthropic SDK that records token usage and cost.

Every call goes through `LLM.create`, so the eval can report exact spend and the
tests can swap in `ScriptedLLM` without touching agent code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from .config import price
from .tracing import span

# Server-side refusal fallback; only the models listed here accept it.
FALLBACK_MODELS = {"claude-opus-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    pass


@dataclass
class Usage:
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0

    @property
    def cost(self) -> float:
        inp, out = price(self.model)
        return (
            self.input_tokens * inp
            + self.cache_read_tokens * inp * 0.1
            + self.cache_write_tokens * inp * 1.25
            + self.output_tokens * out
        ) / 1_000_000


@dataclass
class Ledger:
    calls: list[Usage] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(u.cost for u in self.calls)

    def by_role(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for u in self.calls:
            out[u.role] = out.get(u.role, 0.0) + u.cost
        return out


class LLM(Protocol):
    ledger: Ledger

    async def create(self, *, role: str, model: str, **kwargs: Any) -> Any: ...


class AnthropicLLM:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.ledger = Ledger()

    async def create(self, *, role: str, model: str, **kwargs: Any) -> Any:
        kwargs.setdefault("max_tokens", 16000)
        # Cache the stable prefix (tools, system, earlier turns) across agent-loop turns.
        kwargs.setdefault("cache_control", {"type": "ephemeral"})
        start = time.monotonic()
        with span(f"llm.{role}", model=model) as s:
            if model in FALLBACK_MODELS:
                resp = await self.client.beta.messages.create(
                    model=model, betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                resp = await self.client.messages.create(model=model, **kwargs)
            u = resp.usage
            usage = Usage(
                role=role,
                model=model,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.cache_read_input_tokens or 0,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
                seconds=time.monotonic() - start,
            )
            self.ledger.calls.append(usage)
            s.set_attributes(
                {
                    "llm.input_tokens": usage.input_tokens,
                    "llm.output_tokens": usage.output_tokens,
                    "llm.cost_usd": usage.cost,
                    "llm.stop_reason": resp.stop_reason or "",
                }
            )
        if resp.stop_reason == "refusal":
            raise RefusalError(f"{role} request was declined by {resp.model}")
        return resp


def text_of(resp: Any) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")
