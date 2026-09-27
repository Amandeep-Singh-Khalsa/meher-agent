# Meher Sweets & Namkeen — grounded customer-service agent

A customer-service chat agent for a fictional family sweet shop in Rajouri Garden, New Delhi.
It answers questions about products, prices, delivery, discounts, returns and bulk orders in
English, Hindi and Hinglish, saves sales leads and escalates complaints to the shop team.
The design rule: a local 7B model writes the prose, and deterministic code owns every fact,
every rupee figure, every tool call and every refusal.

## Architecture

```
                customer
                   |
                   v
   POST /chat  +  GET /leads  +  GET /health        (FastAPI, api/app.py)
                   |
                   v
        run_turn()  (agent/loop.py)  <- the only caller of the model
                   |
       +-----------+------------+------------------+
       |           |            |                  |
       v           v            v                  v
  retrieval/   grounding/    tools/registry    llm/client.py
  pipeline.py  guard.py      (save_lead,       -> Ollama
  + retriever  billing.py     escalate)         qwen2.5:7b-instruct
  + resolver   amounts.py    + validation.py     (localhost:11434)
       |           |            |                  ^
       v           v            v                  |
  data/ (business.md, prices.csv, policies.md) ----+
                                            (hand-written OpenAI-compatible
                                             chat-completions client, temperature 0.0)

  Where determinism replaces the model:
    retrieval   decides what is true (BM25 + alias windows over data/)
    billing     decides what the money is (integer arithmetic, prices.csv only)
    guard       decides what may be said (every rupee amount re-checked against
                the computed set; AI disclosure, length cap, script match, PII)
    registry    decides what a tool call may do (validation, action log, handoff)
    loop        decides how many times to ask the model (hard budget of 4 calls)
```

## Quickstart (Windows PowerShell)

Run everything from the repository root:

```
C:\Users\Amandeep\Desktop\yashitask\meher-agent
```

The repo already contains a Python 3.13 virtual environment. To recreate it:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .          # add .[dev] to also get pytest
```

Start Ollama and pull the model (already pulled on this machine):

```powershell
ollama serve
ollama pull qwen2.5:7b-instruct
```

Model access is configured by three environment variables. A `.env` file with the local
Ollama defaults is already in the repo (`LLM_BASE_URL=http://localhost:11434/v1`,
`LLM_API_KEY=ollama`, `LLM_MODEL=qwen2.5:7b-instruct`). Copy `.env.example` to `.env`
to change them; any OpenAI-compatible chat-completions endpoint works.

Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn meher_agent.api.app:app --host 127.0.0.1 --port 8000
```

Call it from another window:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"conversation_id":"demo-1","message":"How much is 500 g of sugar-free kaju katli?"}'
```

## Endpoints

Four routes: `GET /`, `POST /chat`, `GET /leads`, `GET /health`.
Every chat response carries exactly four keys — `reply`, `sources`, `actions`, `handoff` —
plus the diagnostics headers `X-Model-Calls` and `X-Tool-Errors`.

### GET /

Response:

```json
{"service":"meher-agent","version":"1.0.0","endpoints":{"chat":"POST /chat  {conversation_id, message}","leads":"GET /leads","health":"GET /health"}}
```

### POST /chat

Request:

```json
{"conversation_id":"demo-1","message":"How much is 500 g of sugar-free kaju katli?"}
```

Response (200, `X-Model-Calls: 1`, `X-Tool-Errors: 0`):

```json
{
  "reply": "500 g Sugar-free Kaju Katli costs Rs 780. If you need more, I can help with the total. prices.csv#KKSF-500\n(AI assistant)",
  "sources": ["prices.csv#KKSF-500", "policies.md#prices-and-gst", "policies.md#ingredients-and-allergens", "policies.md#storage"],
  "actions": [],
  "handoff": false
}
```

A lead turn — the model answers and the `save_lead` tool call is echoed back:

Request:

```json
{"conversation_id":"demo-2","message":"We need 30 large gift boxes for our office Diwali party on 3 November. I'm Ritu Malhotra, ritu.m@example.com"}
```

Response:

```json
{
  "reply": "Ritu, we can help with that. The total for 30 large Diwali gift boxes is Rs 43,500. Since you're ordering less than 50 boxes, no discount applies. Delivery is free as the total is above Rs 999. Please confirm by paying 30% advance of Rs 13,050.\n(AI assistant)",
  "sources": ["prices.csv#GBL", "policies.md#diwali-2026-gift-boxes-and-discounts", "policies.md#bulk-orders", "policies.md#ingredients-and-allergens"],
  "actions": [{"type": "save_lead", "args": {"name": "Ritu Malhotra", "need": "30 large Diwali gift boxes for the office Diwali party on 3 November", "email": "ritu.m@example.com"}}],
  "handoff": false
}
```

A complaint — `escalate` fires and the turn is handed to the shop team:

Request:

```json
{"conversation_id":"demo-3","message":"The gift box you delivered 30 minutes ago is completely crushed. Very disappointed."}
```

Response:

```json
{
  "reply": "I am sorry about that, and I understand how disappointing it is. Please send a photo of the box as it arrived, with your name and the delivery time.\n\nI can pass your question to the shop team and have them reply to you by email.\n\nOur team will follow up with you by email within one working day.\n\nSource: prices.csv#GBS, prices.csv#GBL, policies.md#returns-and-damaged-deliveries, policies.md#complaints\n\nI am the AI assistant for Meher Sweets & Namkeen.",
  "sources": ["prices.csv#GBS", "prices.csv#GBL", "policies.md#returns-and-damaged-deliveries", "policies.md#complaints"],
  "actions": [{"type": "escalate", "args": {"reason": "customer reports a crushed gift box delivered 30 minutes ago and is very disappointed"}}],
  "handoff": true
}
```

A blank field is rejected with a 422:

Request `{"conversation_id":"","message":"hello"}` ->

```json
{"detail":"conversation_id: String should have at least 1 character"}
```

### GET /leads

Every saved lead, oldest first, with contact details masked (`r*****@example.com`,
`******3210`; an absent field is JSON `null`):

```json
[
  {
    "name": "Ritu Malhotra",
    "email": "r*****@example.com",
    "phone": null,
    "need": "30 large Diwali gift boxes for the office Diwali party on 3 November",
    "conversation_id": "readme-demo-2",
    "created_at": "2026-09-27T14:15:49Z"
  }
]
```

### GET /health

```json
{"status":"ok","model":"qwen2.5:7b-instruct","llm_reachable":true,"max_steps":4,"leads":4}
```

Process liveness is reported independently of the model: `llm_reachable` is cached for
30 s and probed on a worker thread with a 5 s deadline, so a cold or broken endpoint does
not hang a liveness poll.

## Tests and evaluation

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

732 tests, all passing, no network: the LLM is stubbed with scripted fakes and the
evaluation harness is exercised offline against hand-built summaries.

The evaluation replays a JSONL case file against the running service over HTTP:

```powershell
.\.venv\Scripts\python.exe -m evals.runner evals\seed_cases.jsonl --repeats 3 --base-url http://127.0.0.1:8000
```

Real flags (verified with `--help`): `--cases` (or a positional path), `--base-url`
(default `$EVAL_BASE_URL` or `http://127.0.0.1:8000`), `--repeats` (default 3),
`--out-dir` (default `reports`), `--only PREFIX`, `--category NAME`,
`--concurrency N` (default 1, sequential — the reproducible default), `--timeout S`
(default 180), `--wait-s S` (how long to wait for `/health`, default 60).

Each run writes `reports\run-<timestamp>.json` (every case, run and check, with numerators
and denominators) and rewrites `reports\summary.md` (archiving the previous one).
`evals\seed_cases.jsonl` holds the 13 public seed cases; `evals\cases.jsonl` holds the
full 87-case set. The expected totals in the case file come from
`scripts\compute_expected_totals.py`, an independent re-implementation of the pricing
rules that reads only `data\prices.csv` and imports nothing from the agent package.

## Configuration

Everything tunable lives in `config.toml`; model access and a few overrides are
environment variables. The real ones, read by `meher_agent.config` and `evals.runner`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | chat-completions endpoint |
| `LLM_API_KEY` | `ollama` | bearer token; placeholder values are not sent |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | model id |
| `MEHER_CONFIG` | `<repo>\config.toml` | config file path |
| `MEHER_TEMPERATURE` | `0.0` | sampling temperature (0.0 for reproducibility) |
| `MEHER_MAX_STEPS` | `4` | hard cap on model calls per customer message |
| `MEHER_NUM_CTX` | `8192` | context window requested from the endpoint |
| `MEHER_REQUEST_TIMEOUT_S` | `120.0` | per-request timeout |
| `MEHER_MAX_RETRIES` | `2` | retries on 429/5xx, with backoff |
| `MEHER_RETRY_BACKOFF_S` | `1.5` | base backoff |
| `MEHER_MAX_REPLY_CHARS` | `1200` | reply length cap, enforced in code |
| `MEHER_ALLOW_REPLY_REPAIR` | `true` | one repair model call when the guard rejects a draft |
| `MEHER_ALLOW_INTENT_REPAIR` | `true` | one "call the tool now" nudge inside the same budget |
| `MEHER_INR_PER_USD` | `87.0` | for the cost-per-100-conversations metric |
| `MEHER_HOST` / `MEHER_PORT` | `127.0.0.1` / `8000` | service bind address |
| `MEHER_LOG_LEVEL` | `INFO` | log level |
| `MEHER_DATA_DIR` | `<repo>\data` | shop data directory |
| `EVAL_BASE_URL` | `http://127.0.0.1:8000` | where the eval harness finds the service |

## The model is local, and costs nothing per call

The default configuration runs `qwen2.5:7b-instruct` on a local Ollama server. There is no
per-call charge: `config.toml` prices input and output tokens at `$0.0/Mtok`, and the
evaluation report reads the cost line as Rs 0.00 per 100 conversations. The token counts
themselves are real measurements, not estimates — but see the limitation below about this
particular service not exposing usage headers.

## Known limitations

- **Token usage is not measured by this service.** `/chat` does not return token counts,
  so the eval harness records `usage_source: "unknown"`, zero tokens and a Rs 0.00 cost
  line that must be read as "not measured", not "free". The report says so explicitly.
- **A lead can still be lost.** With a 7B model behind a ~900-token system prompt, the
  model sometimes attempts a `save_lead` call that validation rejects (bad name, missing
  fields). A rejected call is surfaced to the model as a tool error and is never silently
  replaced, so that turn ends without a saved lead. In the newest report on disk
  (`reports\run-20260927-182635.json`) the only failing case is `lead-01` for exactly this
  reason; in a verification run against the current code it passed.
- **Latency.** A local 7B model answers a turn in p50 ~7.6s, p95 ~21s (14 messages,
  `reports\run-20260927-182635.json`). Not a UI-grade response time.
- **Conversations and leads are in-memory only.** The lead store and the conversation
  store (LRU, 5000 conversations, 12 turns of history) are process-local and lost on
  restart.
- **The model writes prose, not facts.** Rephrasing, ordering and tone vary between runs
  even at temperature 0; the guard rejects invented facts, but it cannot make a sloppy
  sentence elegant.
- **No streaming, no chat UI.** Responses are plain JSON; SSE and a front-end are not
  built.
- **The 87-case eval on disk was run one repeat at a time.** The runner defaults to
  `--repeats 3`, but each `reports\run-*.json` on disk contains a single run, so
  mean-of-3 / worst-of-3 aggregation across runs has not been exercised end to end.
