"""MCP server exposing the markdown docs and the SQLite database as tools.

Agents reach these through an MCP client over stdio (`python -m orchestrator.mcp_server`),
so the same server also works from any other MCP host.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import Settings
from .sources.docs import DocStore
from .sources.sql import SqlStore


def build_server(docs_dir: Path, db_path: Path) -> MCPServer:
    docs = DocStore(docs_dir)
    sql = SqlStore(db_path)
    server = MCPServer(
        "energy-sources",
        instructions="Energy reference documents (U.S. EIA) and an energy statistics database (Our World in Data).",
    )

    @server.tool()
    def list_docs() -> str:
        """List the available reference documents with their ids and titles."""
        return json.dumps(docs.list_docs())

    @server.tool()
    def search_docs(query: str, k: int = 4) -> str:
        """Keyword search over document sections. Returns the best matching sections with their text."""
        return json.dumps(docs.search(query, k))

    @server.tool()
    def read_doc(doc_id: str, start: int = 0, length: int = 4000) -> str:
        """Read part of one document by id, starting at a character offset."""
        try:
            return json.dumps(docs.read(doc_id, start, length))
        except KeyError as exc:
            raise ToolError(str(exc)) from None

    @server.tool()
    def list_tables() -> str:
        """List tables in the energy statistics database."""
        return json.dumps(sql.list_tables())

    @server.tool()
    def describe_table(table: str = "energy") -> str:
        """Show a table's columns, types and units."""
        try:
            return json.dumps(sql.describe(table))
        except KeyError as exc:
            raise ToolError(str(exc)) from None

    @server.tool()
    def run_sql(sql_query: str) -> str:
        """Run one read-only SQLite SELECT query and return up to 50 rows."""
        try:
            return json.dumps(sql.query(sql_query))
        except (ValueError, sqlite3.Error) as exc:
            # Bad SQL goes back to the agent as a readable error so it can fix the query.
            raise ToolError(f"{type(exc).__name__}: {exc}") from None

    return server


def main() -> None:
    s = Settings()
    docs_dir = Path(os.environ.get("ORCH_DOCS_DIR", s.data_dir / "docs"))
    build_server(docs_dir, s.sqlite_path).run("stdio")


if __name__ == "__main__":
    main()
