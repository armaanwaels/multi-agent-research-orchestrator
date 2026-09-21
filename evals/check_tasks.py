"""Rerun every SQL check in tasks.yaml against the seeded database."""

import sqlite3
import sys
import tempfile
from pathlib import Path

import yaml

from orchestrator.sources.sql import build_database

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tasks = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())
    # Build a fresh copy from the committed CSV so the check never depends on local state.
    db = Path(tempfile.mkdtemp()) / "energy.db"
    build_database(ROOT / "data" / "owid_energy.csv", db)
    con = sqlite3.connect(db)
    failures, n = 0, 0
    for t in tasks:
        for c in t.get("checks", []):
            n += 1
            got = con.execute(c["sql"]).fetchone()[0]
            if got is None or abs(got - c["expect"]) > 1e-6:
                failures += 1
                print(f"{t['id']}: {c['sql']} -> {got}, expected {c['expect']}")
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    print(f"{len(tasks)} tasks, {n} SQL checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
