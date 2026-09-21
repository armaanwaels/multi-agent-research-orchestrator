"""Source 2: SQLite database seeded from the Our World in Data energy CSV."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

TABLE = "energy"

COLUMN_NOTES = {
    "country": "Country or aggregate region name (e.g. 'World', 'Europe', 'High-income countries')",
    "year": "Calendar year, 2000 onward",
    "iso_code": "ISO 3166 alpha-3 code; empty for aggregate regions",
    "population": "People",
    "electricity_generation": "Total electricity generation, TWh",
    "electricity_demand": "Electricity demand, TWh",
    "per_capita_electricity": "Electricity generation per person, kWh",
    "primary_energy_consumption": "Primary energy consumption, TWh",
    "energy_per_capita": "Primary energy consumption per person, kWh",
    "carbon_intensity_elec": "Carbon intensity of electricity, gCO2 per kWh",
    "greenhouse_gas_emissions": "Emissions from electricity generation, million tonnes CO2-equivalent",
}
for src in ("coal", "gas", "oil", "nuclear", "hydro", "solar", "wind", "renewables", "fossil", "low_carbon"):
    COLUMN_NOTES[f"{src}_electricity"] = f"Electricity generated from {src.replace('_', '-')}, TWh"
for src in ("renewables", "solar", "wind", "coal", "nuclear", "fossil"):
    COLUMN_NOTES[f"{src}_share_elec"] = f"Share of electricity generated from {src}, percent"


def build_database(csv_path: Path, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = [[(r[c] if r[c] != "" else None) for c in cols] for r in reader]
    types = {c: "TEXT" if c in ("country", "iso_code") else ("INTEGER" if c == "year" else "REAL") for c in cols}
    con = sqlite3.connect(db_path)
    con.execute(f"CREATE TABLE {TABLE} ({', '.join(f'{c} {types[c]}' for c in cols)})")
    con.executemany(f"INSERT INTO {TABLE} VALUES ({', '.join('?' * len(cols))})", rows)
    con.execute(f"CREATE INDEX idx_country_year ON {TABLE}(country, year)")
    con.commit()
    con.close()
    return len(rows)


class SqlStore:
    def __init__(self, db_path: Path, max_rows: int = 50) -> None:
        # Read-only connection: the agent writes SQL, so the database refuses writes itself.
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.max_rows = max_rows

    def list_tables(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    def describe(self, table: str = TABLE) -> list[dict]:
        cols = self.con.execute(f"PRAGMA table_info({table})").fetchall()
        if not cols:
            raise KeyError(f"unknown table {table!r}")
        return [{"column": c[1], "type": c[2], "meaning": COLUMN_NOTES.get(c[1], "")} for c in cols]

    def query(self, sql: str) -> dict:
        statement = sql.strip().rstrip(";")
        if not statement.lower().startswith(("select", "with")):
            raise ValueError("only SELECT queries are allowed")
        cur = self.con.execute(statement)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(self.max_rows + 1)
        return {
            "columns": columns,
            "rows": [list(r) for r in rows[: self.max_rows]],
            "truncated": len(rows) > self.max_rows,
        }
