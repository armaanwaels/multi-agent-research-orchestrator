"""Wires the sources, MCP client, checkpointer and run store together."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import mcp
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import AsyncQdrantClient

from .agents import build_catalog
from .config import Settings
from .graph import Deps
from .llm import LLM, AnthropicLLM
from .runs import MemoryRunStore, PostgresRunStore, RunStore
from .sources.vectors import VectorStore
from .sources.web import make_web_search
from .sources.wikipedia import Wikipedia
from .tools import Toolbox, local_tools, mcp_tools
from .tracing import setup_tracing


@dataclass
class Runtime:
    deps: Deps
    checkpointer: object
    runs: RunStore


def mcp_server_params() -> mcp.StdioServerParameters:
    return mcp.StdioServerParameters(
        command=sys.executable, args=["-m", "orchestrator.mcp_server"], env=dict(os.environ)
    )


@asynccontextmanager
async def open_runtime(
    settings: Settings | None = None,
    *,
    persistent: bool = True,
    auto_approve: bool = False,
    llm: LLM | None = None,
) -> AsyncIterator[Runtime]:
    s = settings or Settings()
    setup_tracing(s.tracing)
    async with AsyncExitStack() as stack:
        mcp_client = await stack.enter_async_context(mcp.Client(mcp_server_params()))
        qdrant = AsyncQdrantClient(url=s.qdrant_url)
        stack.push_async_callback(qdrant.close)
        tools = await mcp_tools(mcp_client) + local_tools(
            make_web_search(s.web_search, s.data_dir),
            Wikipedia(),
            VectorStore(qdrant, s.qdrant_collection, s.embedding_model),
        )
        if persistent:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(s.postgres_url))
            await checkpointer.setup()
            runs: RunStore = await stack.enter_async_context(PostgresRunStore.connect(s.postgres_url))
        else:
            checkpointer, runs = InMemorySaver(), MemoryRunStore()
        toolbox = Toolbox(tools, s.tool_result_chars)
        deps = Deps(
            llm=llm or AnthropicLLM(),
            toolbox=toolbox,
            settings=s,
            auto_approve=auto_approve,
            catalog=await build_catalog(toolbox, s.web_search),
        )
        yield Runtime(deps, checkpointer, runs)
