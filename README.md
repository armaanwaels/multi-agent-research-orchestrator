# multi-agent-research-orchestrator

Five LangGraph agents that plan, search, retrieve and synthesize answers across five data sources, with a human review step before the answer is written and an eval harness that scores relevance and citation accuracy.

## Problem

A research question usually needs more than one source: a number from a database, an explanation from reference docs, a recent development from the news, background from an encyclopedia, a pointer into the literature. One RAG call over one index does not cover that, and an answer that hides where each claim came from is hard to check. I split the work across specialist agents with narrow tool sets, turned every retrieved item into a citable piece of evidence, put a person in front of the evidence before the answer is written, and measured the result.

The domain is energy, because it has public sources of every kind: EIA reference pages and articles (U.S. government work, public domain), Our World in Data statistics (CC BY 4.0), arXiv abstracts (CC0 metadata) and Wikipedia.

## Architecture

```mermaid
flowchart TD
    Q([question]) --> P[planner]
    P --> S{supervisor}
    S --> A1[search agent<br/>web search, Wikipedia]
    S --> A2[documents agent<br/>EIA docs via MCP, arXiv via Qdrant]
    S --> A3[data agent<br/>SQLite via MCP]
    A1 --> S
    A2 --> S
    A3 --> S
    S --> R[[human review<br/>LangGraph interrupt]]
    R -->|approve or edit| W[synthesizer]
    R -->|reject with feedback| P
    W --> ANS([answer with evidence citations])
    PG[(PostgreSQL<br/>checkpoints, runs)] -.- S
    MCP[MCP server<br/>docs + SQL tools] -.- A2
    MCP -.- A3
    QD[(Qdrant)] -.- A2
```

The planner writes a short plan (one to four steps, each naming a specialist). The supervisor walks it. Each specialist runs a small tool loop and returns a note plus evidence items with ids like `E4`. Before synthesis the graph interrupts: a person approves, drops evidence items, adds an instruction for the writer, or rejects the research with feedback, which sends it back to the planner. The synthesizer may only cite evidence ids. State is checkpointed to PostgreSQL after every step, so a run paused for review can be resumed later from another process.

| Source | Contents | Reached through |
|---|---|---|
| Markdown docs | 30 EIA "Energy Explained" pages | MCP server (`search_docs`, `read_doc`) |
| SQLite | Our World in Data energy table, 7,636 country-year rows from 2000 | MCP server (`run_sql`, read-only) |
| Web search | 40 EIA "Today in Energy" articles (fixture), or live DuckDuckGo | local tool; `WEB_SEARCH_BACKEND` |
| Wikipedia | live API | local tool |
| Vector store | 242 arXiv abstracts, `bge-small-en-v1.5` embeddings | Qdrant |

ARCHITECTURE.md has each agent's contract and the reasoning behind the design.

## Quickstart

Needs Docker and an `ANTHROPIC_API_KEY`.

```bash
git clone https://github.com/armaanwaels/multi-agent-research-orchestrator.git
cd multi-agent-research-orchestrator
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
docker compose up -d        # Qdrant, PostgreSQL, and a one-shot seed job
docker compose run --rm app ask "How did Germany's coal share of electricity change between 2015 and 2025?"
```

From a fresh clone on an Apple M3 Pro, `docker compose up -d` had the stores seeded in 26 seconds with Docker's build cache warm; a cold build also pulls the base images, which I did not time. The example question then took 56 seconds and cost $0.20 on the default models. Without `--auto-approve`, `ask` stops at the review step and shows the plan, the agents' notes and every evidence item:

```
[a]pprove  [e]dit  [r]eject  [q]uit and resume later >
```

Quitting leaves the run in PostgreSQL. `docker compose run --rm app resume <thread-id>` continues it, and `docker compose run --rm app runs` lists recent runs with their cost.

Without Docker, with Python 3.11 and uv:

```bash
docker compose up -d qdrant postgres
uv sync
uv run orchestrator seed
uv run orchestrator ask "..."        # add --auto-approve to skip review
uv run pytest
```

Models are set per role with `ORCH_PLANNER_MODEL`, `ORCH_WORKER_MODEL`, `ORCH_SYNTH_MODEL` and `ORCH_JUDGE_MODEL`, all defaulting to `claude-opus-5`. Set `ORCH_TRACING=otel` to emit OpenTelemetry spans for every node, model call and tool call (with `OTEL_EXPORTER_OTLP_ENDPOINT` to send them to LangSmith, Langfuse, Jaeger or any OTLP collector).

## Evaluation

Produced by this command on September 21, 2026, at the commit tagged `eval-v2-2026-09-21`:

```bash
ORCH_PLANNER_MODEL=claude-sonnet-5 ORCH_WORKER_MODEL=claude-sonnet-5 \
ORCH_SYNTH_MODEL=claude-sonnet-5 ORCH_JUDGE_MODEL=claude-sonnet-5 \
uv run python evals/run_eval.py --budget 5.5
```

32 tasks, all completed. Human review was auto-approved and web search used the fixture. Full output is in `results/eval_summary.md` (table) and `results/eval_results.jsonl` (every answer, plan, judge rationale and cost).

| Metric | Run 2 (current) | Run 1 |
|---|---|---|
| Relevance: tasks the judge scored 4 or 5 out of 5 | 94% (30/32) | 78% (25/32) |
| Relevance: mean judge score | 4.59 / 5 | 4.28 / 5 |
| Citation accuracy: citations whose evidence supports the sentence | 96% (291/303) | 92% (258/281) |
| Citations to evidence ids that do not exist | 0 | 0 |
| Tasks where the agents used every source type the task needs | 28/32 | 28/32 |
| Median time per task | 22 s | 27 s |
| Cost: system / judge / total | $1.90 / $0.82 / $2.72 | $2.00 / $0.83 / $2.83 |

**What changed between runs.** Run 1 (commit `eval-2026-09-21`, results in `results/eval-v1_*`) exposed three weak spots, fixed before run 2:

1. The data agent assumed the database ended in 2023 and never queried 2024 or 2025. `describe_table` now reports the real year range, and the agent is told to query recent years instead of assuming they are missing.
2. The planner did not know what the documents contained, so it skipped the EIA page that answers the geothermal question. It now gets a catalog of every source (document titles, table columns and year range, web fixture scope) built from the live tools at startup.
3. Web search returned snippets only. A `read_article` tool now opens the full article.

I also corrected one reference answer between runs (t19, see below). On the 31 tasks whose reference did not change, tasks scoring 4 or 5 went from 25 to 29.

**How the tasks were made.** The 32 questions and reference answers in `evals/tasks.yaml` were written from the committed source snapshots, not from an external benchmark. 22 need two or more source types. Every database number in a reference is listed with the SQL that produces it, and `uv run python evals/check_tasks.py` reruns all 43 checks (CI runs it too). The references were not reviewed line by line. One was wrong in run 1: t19 took EIA's section heading ("electricity generation and space heating") as the main uses of natural gas, while the same page's numbers make industry the second-largest use. It was corrected before run 2.

**How they are scored.** An LLM judge grades each answer against its reference on a 1-5 rubric (`evals/judge.py`). Citations are checked in two steps: code confirms the cited id exists, then the judge decides whether the cited evidence supports the sentence. The judge is the same model family as the system, which is a known source of bias.

**Tasks that still score 3 or lower in run 2:**

| Task | Run 1 | Run 2 | Cause |
|---|---|---|---|
| t05 largest U.S. wind farm | 5 | 2 | Regression. The planner told the search agent to use Wikipedia, which names an older record holder. The June 2026 EIA article on SunZia was in the web fixture but never searched. The catalog makes the planner name sources, and here it named the wrong one. |
| t22 Puerto Rico outages | 3 | 3 | The search agent had `read_article` available but did not open the article, so the fact past the snippet cut-off was missed again. |

With the default models (Opus 5 for every role), a two-task pilot scored 5 on both tasks at about $0.42 per task including the judge (`uv run python evals/run_eval.py --ids t05-sunzia t10-hydrogen --out pilot`, results in `results/pilot_summary.md`, run before the fixes). Two tasks say little about quality; both full runs used Sonnet 5 to stay inside a $5 budget per run.

## Decisions and tradeoffs

- **A deterministic supervisor walks the plan.** Agents never hand off to each other, so the route is a data structure I can read back and assert on in tests. It is bounded at four steps, five turns per agent and two replans. The cost is that nothing reacts mid-run when one agent's findings call for another lookup.
- **Review happens once per run, just before synthesis.** The question a person can usefully answer is "is this the right evidence". Interrupts plus PostgreSQL checkpoints mean the reviewer does not need to be in the same process.
- **Evidence ids instead of free-form citations.** The writer cannot cite anything that was not retrieved, which makes citation validity checkable in code and leaves only "does this evidence support this claim" to the judge.
- **MCP for docs and SQL.** Those tools run as a separate MCP server over stdio, so they also work from any MCP host, and the agents do not import the data layer.
- **A web fixture by default.** The eval must not change when the web does. The live backend is one environment variable away.
- **Qdrant over pgvector.** It keeps the vector index apart from run state. At 242 abstracts pgvector would be enough, and it would remove a service.

## What went wrong / limitations

- **The first eval found three bugs.** The data agent assumed the database ended in 2023, the planner did not know what the documents contained, and web search could not read past a snippet. All three are fixed (see Evaluation), and the table shows both runs.
- **The planner can steer a specialist to the wrong source.** In run 2 it sent a "largest wind farm" question to Wikipedia instead of the newer EIA article, and the score for that task fell from 5 to 2. The planner should say what to find, not where, and leave source choice to the specialist.
- **Agents do not always use the tools they have.** `read_article` fixed nothing for t22 because the search agent never called it. Tool availability is not tool use; the brief or a check on thin evidence would have to push for it.
- **The web fixture is U.S.-centric.** Every article is from EIA, so questions about other countries get irrelevant web results.
- **Run-to-run variance was not measured.** Each configuration ran once. Some score changes between runs (t24 and t29 each dropped from 5 to 4 with no related change) are noise, so a few points of the 78% to 94% gain may be too.
- **The references were not reviewed line by line, and one was wrong.** t19 is documented above. Read the relevance number with that in mind.
- **Steps run one after another.** Independent steps could run in parallel with LangGraph's `Send`.
- **Human review is bypassed in the eval.** The interrupt, edit and resume paths are covered by tests and were run by hand against the real API, but the eval measures the system without a reviewer.
- **The same model family writes and grades.** A judge from another provider, or a hand-graded sample, would make the relevance number more trustworthy.

## Stack

Python 3.11, uv, LangGraph, Anthropic API, MCP (Python SDK 2.x), PostgreSQL, Qdrant, fastembed, SQLite, OpenTelemetry, Docker Compose, GitHub Actions.
