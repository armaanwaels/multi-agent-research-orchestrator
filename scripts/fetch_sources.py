"""Download the public source snapshots into data/.

Run once; the outputs are committed so the eval does not drift when the
upstream pages change. Wikipedia is the one source queried live.

  data/docs/*.md             EIA "Energy Explained" pages (US government work, public domain)
  data/web_fixture.jsonl     EIA "Today in Energy" articles, the offline stand-in for web search
  data/arxiv_abstracts.jsonl arXiv abstracts on energy topics (arXiv metadata is CC0)
  data/owid_energy.csv       Our World in Data energy dataset, 2000 onward (CC BY 4.0)
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {
    "User-Agent": "multi-agent-research-orchestrator/0.1 (github.com/armaanwaels)",
}

EIA_PAGES = [
    "solar/",
    "solar/photovoltaics-and-electricity.php",
    "solar/where-solar-is-found.php",
    "wind/",
    "wind/electricity-generation-from-wind.php",
    "wind/where-wind-power-is-harnessed.php",
    "hydropower/",
    "hydropower/where-hydropower-is-generated.php",
    "nuclear/",
    "nuclear/nuclear-power-plants.php",
    "nuclear/us-nuclear-industry.php",
    "coal/",
    "coal/use-of-coal.php",
    "natural-gas/",
    "natural-gas/use-of-natural-gas.php",
    "oil-and-petroleum-products/",
    "geothermal/",
    "geothermal/use-of-geothermal-energy.php",
    "biomass/",
    "biofuels/",
    "hydrogen/",
    "hydrogen/production-of-hydrogen.php",
    "electricity/",
    "electricity/electricity-in-the-us.php",
    "electricity/energy-storage-for-electricity-generation.php",
    "electricity/how-electricity-is-generated.php",
    "electricity/prices-and-factors-affecting-prices.php",
    "renewable-sources/",
    "us-energy-facts/",
    "energy-and-the-environment/",
]

ARXIV_QUERIES = [
    "perovskite solar cell efficiency",
    "offshore wind power forecasting",
    "grid-scale battery energy storage",
    "hydrogen electrolysis cost",
    "small modular nuclear reactor",
    "electricity market renewable integration",
]

OWID_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
OWID_COLUMNS = [
    "country",
    "year",
    "iso_code",
    "population",
    "electricity_generation",
    "electricity_demand",
    "coal_electricity",
    "gas_electricity",
    "oil_electricity",
    "nuclear_electricity",
    "hydro_electricity",
    "solar_electricity",
    "wind_electricity",
    "renewables_electricity",
    "fossil_electricity",
    "low_carbon_electricity",
    "renewables_share_elec",
    "solar_share_elec",
    "wind_share_elec",
    "coal_share_elec",
    "nuclear_share_elec",
    "fossil_share_elec",
    "carbon_intensity_elec",
    "primary_energy_consumption",
    "energy_per_capita",
    "per_capita_electricity",
    "greenhouse_gas_emissions",
]


def slug(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.removesuffix(".php").lower()).strip("-")


def fetch_eia_docs(client: httpx.Client) -> None:
    out = DATA / "docs"
    out.mkdir(parents=True, exist_ok=True)
    for page in EIA_PAGES:
        url = f"https://www.eia.gov/energyexplained/{page}"
        resp = client.get(url)
        if resp.status_code != 200:
            print(f"skip {url} ({resp.status_code})")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        body = soup.select_one("div.article-content")
        if body is None:
            print(f"skip {url} (no article body)")
            continue
        for tag in body.select("img, script, style, figure, .share, .related"):
            tag.decompose()
        title = soup.title.get_text(strip=True).split(" - ")[0] if soup.title else page
        md = markdownify(str(body), heading_style="ATX", strip=["a"])
        md = re.sub(r"\n{3,}", "\n\n", md).strip()
        (out / f"{slug(page)}.md").write_text(
            f"# {title}\n\nSource: {url} (U.S. Energy Information Administration, public domain)\n\n{md}\n"
        )
        print(f"doc {slug(page)} {len(md)} chars")
        time.sleep(0.5)


def fetch_today_in_energy(client: httpx.Client, limit: int = 40) -> None:
    # Unknown IDs return a fallback article with HTTP 200, so only follow real links.
    ids: set[int] = set()
    for listing in ("archive.php", "?tg=electricity", "?tg=renewables"):
        html = client.get(f"https://www.eia.gov/todayinenergy/{listing}").text
        ids.update(int(m) for m in re.findall(r"detail\.php\?id=(\d+)", html))
    rows, titles = [], set()
    for article_id in sorted(ids, reverse=True):
        if len(rows) >= limit:
            break
        url = f"https://www.eia.gov/todayinenergy/detail.php?id={article_id}"
        resp = client.get(url)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.select_one("div.tie-article h1") or soup.find("h1")
        body = soup.select_one("div.tie-article") or soup.select_one("div.article-content")
        if body is None or h1 is None:
            continue
        date = soup.select_one("span.date")
        text = " ".join(p.get_text(" ", strip=True) for p in body.find_all("p"))
        title = h1.get_text(strip=True)
        if title in titles:
            continue
        titles.add(title)
        rows.append(
            {
                "title": title,
                "url": url,
                "date": date.get_text(strip=True) if date else "",
                "text": text[:4000],
            }
        )
        time.sleep(0.5)
    with open(DATA / "web_fixture.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"web fixture: {len(rows)} articles")


def fetch_arxiv(client: httpx.Client, per_query: int = 50) -> None:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    seen, rows = set(), []
    for q in ARXIV_QUERIES:
        # The arXiv API answers 406 to httpx requests here while urllib and curl work,
        # so this one source uses urllib. Colons must stay literal in the query.
        query = "+AND+".join(f"all:{w}" for w in q.split())
        req = urllib.request.Request(
            f"https://export.arxiv.org/api/query?search_query={query}&max_results={per_query}",
            headers={"User-Agent": UA["User-Agent"]},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
        root = ET.fromstring(text)
        for e in root.findall("a:entry", ns):
            arxiv_id = e.findtext("a:id", namespaces=ns).rsplit("/", 1)[-1]
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            rows.append(
                {
                    "id": arxiv_id,
                    "title": " ".join(e.findtext("a:title", namespaces=ns).split()),
                    "summary": " ".join(e.findtext("a:summary", namespaces=ns).split()),
                    "published": e.findtext("a:published", namespaces=ns)[:10],
                    "authors": [a.findtext("a:name", namespaces=ns) for a in e.findall("a:author", ns)][:5],
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "query": q,
                }
            )
        time.sleep(3)  # arXiv asks for one request every three seconds
    with open(DATA / "arxiv_abstracts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"arxiv: {len(rows)} abstracts")


def fetch_owid(client: httpx.Client) -> None:
    reader = csv.DictReader(io.StringIO(client.get(OWID_URL).text))
    with open(DATA / "owid_energy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OWID_COLUMNS)
        writer.writeheader()
        n = 0
        for row in reader:
            if int(row["year"]) >= 2000:
                writer.writerow({c: row.get(c, "") for c in OWID_COLUMNS})
                n += 1
    print(f"owid: {n} rows")


STEPS = {"docs": fetch_eia_docs, "web": fetch_today_in_energy, "arxiv": fetch_arxiv, "owid": fetch_owid}


def main() -> None:
    steps = sys.argv[1:] or list(STEPS)
    DATA.mkdir(exist_ok=True)
    with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as client:
        for name in steps:
            STEPS[name](client)


if __name__ == "__main__":
    main()
