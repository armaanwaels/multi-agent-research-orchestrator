"""Rerun every SQL check in tasks.yaml against the seeded database."""

import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tasks = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())
    con = sqlite3.connect(ROOT / "data" / "energy.db")
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
