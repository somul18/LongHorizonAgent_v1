# LongHorizonAgent: state, not history

Long-running agents usually fail the same way. Every observation and action gets appended to the
prompt, so each step costs more than the last. Stale facts pile up next to current ones, and the
agent's picture of the world gets less reliable the longer it runs.

**LHA drops the transcript entirely.** Each step the model sees only a small, typed,
**explicit working state**. It edits that state itself through a handful of operations, and
everything else goes to cold storage, where the agent can `recall` it when needed.

```
               +----------------------------- every step -----------------------------+
               |                                                                      |
  WorkingState |  render() ─► LLM (Bedrock) ─► {state_ops, action} ─► apply_ops ─► tool |
  (bounded,    |      ▲                                                  │         │   |
   checkpointed|      └──────── Compactor keeps it under budget ◄────────┘         │   |
   each step)  |                (cheap model, e.g. Liquid LFM, digests evictions)  │   |
               +───────────────────────────────────────────────────────────────────┼───+
                                                                                   ▼
                     Archive (append-only JSONL, searchable) ──► Tinybird (live analytics)
                     superseded facts · evicted facts · every observation · notes
```

## What persists and what gets discarded

| Section | Lifetime | How it leaves the prompt |
|---|---|---|
| `goal` | whole run | never |
| `facts` (key → value, source, step, confidence, pin) | until overwritten or dropped | `set_fact` **overwrites**, and the stale value moves to the archive. Unpinned, least-recently-used facts get evicted when over budget. |
| `tasks` (the plan) | until closed | done/dropped tasks are collapsed into a one-line digest |
| `questions` | until resolved | `resolve_question` |
| `focus` | one line | overwritten |
| `notes` | volatile | a bounded ring buffer; the oldest roll off first |
| `observation` | **one step** | only the latest tool result is shown; all results are archived |

Eviction order is from cheapest to lose to most valuable: notes, then the observation, then
closed tasks, then unpinned facts. Pinned facts are never evicted. The archive leaves breadcrumbs
(`# ARCHIVED: key1, key2…`), so the agent knows what it can `recall`.

Two more properties matter for runs that last hours or days:

- **Crash-safe.** State is atomically checkpointed every step. Kill the process and resume
  with the same `--run-id` days later, and it picks up at the exact step.
- **Self-correcting.** Malformed JSON or invalid ops never crash a run. They come back to the
  model as `FEEDBACK FROM LAST STEP`.

## Results: IncidentDesk benchmark

`lha/bench/incident_desk.py` is a synthetic long-horizon task where the ground truth keeps changing.
The agent drains an inbox of N messages. Most are noisy chatter. Some reassign a service's owner,
region or version, and they do so repeatedly, so earlier information goes stale. At the end, the
agent must report the *current* values, including values last seen at the very start of the run.

`python -m lha.bench.run` (offline, deterministic, runs in seconds; working-state budget of 450 tokens):

| messages | agent | score | peak prompt tok | total input tok | cost vs naive |
|---:|---|---:|---:|---:|---:|
| 50 | **stateful** | 1.00 | 817 | 39,240 | **50.3%** |
| 50 | naive | 1.00 | 2,878 | 78,000 | 100% |
| 200 | **stateful** | 1.00 | 817 | 161,963 | **14.2%** |
| 200 | naive | 1.00 | 11,224 | 1,136,749 | 100% |
| 1000 | **stateful** | 1.00 | 817 | 807,440 | **2.9%** |
| 1000 | naive | 1.00 | 56,087 | 28,079,688 | 100% |
| 3000 | **stateful** | 1.00 | 881 | 2,423,130 | **1.0%** |
| 3000 | naive | 1.00 | 169,525 | 253,700,413 | 100% |

- **The prompt stays flat.** The stateful agent's prompt is about 820 tokens at step 50 and at step 3,000.
  The naive agent's prompt grows linearly, so its total cost grows quadratically. At 3,000 steps it
  would no longer fit in most context windows.
- **Nothing is lost.** With this budget the compactor evicts facts. At answer time, the agent
  `recall`s the evicted facts it needs from the archive and still scores 1.00.
- **How to read the offline numbers:** both arms here are *deterministic reader policies* that see
  exactly the prompt a model would see. That makes the token numbers exact and reproducible, but
  the naive reader is a perfect reader, so accuracy ties at 1.00 in offline mode. To measure how a
  real model's accuracy drops as it reads 50k–170k tokens of conflicting history, run
  `--live` (below).

## Quick start

```bash
pip install -e ".[dev,aws]"
pytest -q                                # 7 tests: ops, compaction, pins, parsing, flat-vs-growing, crash/resume
python -m lha.bench.run                  # offline benchmark -> results/report.md

cp .env.example .env && $EDITOR .env     # then export the vars
python -m lha.bench.run --live --sizes 50,200 --skip-naive-above 200
python -m lha.cli run "Track this week's major open-weight model releases: name, org, params, license" --run-id rel1
python -m lha.cli run --run-id rel1      # resume (after Ctrl-C, a crash, or tomorrow)
python -m lha.cli state rel1             # what the agent currently believes
python -m lha.cli recall rel1 "license"  # dig into cold storage
```

## Sponsor stack

| Sponsor | Role in LHA | Where |
|---|---|---|
| **AWS** | Main reasoning model through Bedrock's Converse API (`LHA_BEDROCK_MODEL_ID`). Checkpoints and archives are plain files, ready to move to S3 or DynamoDB. | `lha/llm.py::BedrockLLM` |
| **Liquid AI** | A small, efficient LFM model as the **memory model**: it writes the digests of evicted and completed work, so compaction doesn't burn frontier-model tokens. It can also run the whole agent through any OpenAI-compatible endpoint. | `OpenAICompatLLM`, `summarizer_from_env` |
| **Nimble** | `web_search` / `web_fetch` tools that bring in fresh web data. Pages land in `observation` for one step only, and the agent copies what matters into facts with a `source`. | `lha/tools.py::nimble_tools` |
| **Tinybird** | Every step and every archive item streams to the Events API. Endpoints show `context_growth` (is the prompt flat?), `run_costs`, and `fact_history` (an audit trail of every value a key ever had). | `lha/sinks.py`, `tinybird/` |
| Black Forest Labs | Not used yet. A natural next step is a `generate_image` tool whose outputs are archived by reference, never inlined. | — |

Deploy the Tinybird project with the `tb` CLI (`tb deploy` from `tinybird/`), then set `TINYBIRD_TOKEN`.
The Nimble endpoint defaults to its realtime API and can be overridden with `LHA_NIMBLE_BASE_URL`.

## Code map

```
lha/state.py        WorkingState, edit ops, render()        <- the core idea
lha/compactor.py    budget enforcement + eviction policy
lha/archive.py      append-only log + BM25-ish recall
lha/agent.py        StatefulAgent (checkpoint/resume) and NaiveAgent baseline
lha/llm.py          Bedrock, OpenAI-compatible (Liquid), Scripted
lha/tools.py        tool box, recall, Nimble web tools
lha/sinks.py        JSONL + Tinybird telemetry
lha/bench/          IncidentDesk env + benchmark runner
tinybird/           datasources + API endpoints
```

## 3-minute demo script

1. **The problem (30s):** run `python -m lha.bench.run --sizes 50,1000,3000` and point at the naive
   agent's peak prompt: 170k tokens and 254M total input tokens.
2. **The fix (60s):** run `python -m lha.cli state <run>` and show that this is *all* the agent sees:
   goal, plan, facts with provenance, and archive breadcrumbs.
3. **Nothing lost (30s):** show `recall` bringing back an evicted fact, and Tinybird's `fact_history`
   listing every value a key ever held.
4. **Days-long (30s):** start a live run, `Ctrl-C` it, resume it with the same `--run-id`, and show the step counter continuing.
5. **Dashboard (30s):** the Tinybird `context_growth` endpoint, where the naive curve rises and the stateful curve stays flat.

## Roadmap

- Embedding recall (Bedrock Titan or Cohere embeddings), alongside lexical recall
- Learned eviction: let the Liquid model score fact value instead of LRU × confidence
- Multi-agent: shared fact store with per-key ownership and conflict resolution
- Live-model accuracy sweep across horizons (`--live --seeds 0,1,2`)
