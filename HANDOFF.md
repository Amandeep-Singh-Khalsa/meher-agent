# HANDOFF.md — Meher Sweets Grounded Agent

**Date:** 2026-09-28
**Repo:** https://github.com/Amandeep-Singh-Khalsa/meher-agent
**Status:** Shipped (v1). Eval running in background. Known 7B reasoning gaps documented.

---

## 1. What this project is

A **grounded customer-service agent** for Meher Sweets & Namkeen (a sweet shop). It answers
customer questions about prices, policies, opening hours, and takes orders — but with a hard
rule: **it may never invent a fact**. Every rupee figure, policy statement, and source citation
must be verifiable from the shop's data files.

**Constraints from the PRD:**
- Local 7B model (Ollama `qwen2.5:7b-instruct`), zero API cost
- Max 4 model calls per turn (hard budget)
- Deterministic code owns all facts; the model only writes prose
- Must handle Hinglish/Hindi, prompt injection, PII masking, lead capture, complaints
- Deliverables: working service, eval harness, 50+ test cases, docs, submission zip

**Candidate:** Yashi Gupta → submission zip must be `dhanur-task-yashi-gupta.zip`

---

## 2. Architecture

```
Customer → POST /chat → AgentLoop → Retrieval → Guard → Billing → Tools → LLM → Reply
                                ↓
                    Deterministic lead capture
                    Deterministic complaint escalation
                    Post-response guard (G1-G4)
```

**The core design decision:** the model writes the sentence, deterministic code owns every fact,
every rupee figure, every tool call, and every refusal. This is why a 7B model can be sloppy in
prose without being able to lie about a price.

### Layers

| Layer | File | What it does |
|---|---|---|
| **Config** | `config.toml` | Model, temperature, max_steps, server, runtime |
| **Data** | `data/business.md`, `data/prices.csv`, `data/policies.md` | Authoritative facts (READ-ONLY) |
| **Corpus** | `src/meher_agent/data/corpus.py` | Loads data files, builds search index |
| **Retrieval** | `src/meher_agent/retrieval/pipeline.py` | BM25 + bilingual lexicon, builds briefing |
| **Resolver** | `src/meher_agent/retrieval/resolver.py` | Cross-turn resolution ("30 minutes ago" → 30 gift boxes) |
| **Billing** | `src/meher_agent/grounding/billing.py` | Deterministic rupee totals |
| **Amounts** | `src/meher_agent/grounding/amounts.py` | G2 extractor, Indian formatting, truncation guard |
| **Guard** | `src/meher_agent/grounding/guard.py` | G1-G4 checks, intent detection, templates |
| **LLM** | `src/meher_agent/llm/client.py` | OpenAI-compatible Ollama client |
| **Prompts** | `src/meher_agent/llm/prompts.py` | 900-token grounded system prompt |
| **Tools** | `src/meher_agent/tools/registry.py` | `save_lead` / `escalate` validation + execution |
| **Loop** | `src/meher_agent/agent/loop.py` | 4-call budget, deterministic fallbacks |
| **API** | `src/meher_agent/api/app.py` | FastAPI: `/chat`, `/chat/stream`, `/eval/stream`, `/leads`, `/health`, `/` (chat UI), `/eval` (dashboard) |
| **Safety** | `src/meher_agent/safety/pii.py` | PII masking in logs and API output |
| **Eval** | `evals/runner.py`, `evals/checks.py`, `evals/report.py` | 87 cases, 3-repeat methodology, p50/p95 |
| **Oracle** | `scripts/compute_expected_totals.py` | Independent price calculator (stdlib only) |

---

## 3. What's been built

### Core (all working, all tested)
- Corpus/config/types
- Devanagari transliteration, bilingual lexicon, BM25 retrieval, deterministic order resolver
- Deterministic billing and post-response guard/templates
- Hand-written OpenAI-compatible Ollama client and 900-token grounded prompt
- `save_lead`/`escalate` validation, lead store, conversation store
- Four-call-cap agent loop
- PII masking and masked logging
- FastAPI `POST /chat`, `GET /leads`, `GET /health`
- Evaluation checks, runner, three-repeat reporting, p50/p95, token/cost metrics

### Key fixes made during development
1. **Deterministic lead capture** — the 7B model frequently answered correctly but *omitted*
   the `save_lead` tool call. The loop now reads name/contact from the message with regexes and
   calls the validated tool itself when the model made no tool attempt at all.
2. **Deterministic complaint escalation** — same pattern. If the guard detects a complaint and
   the model didn't escalate, the loop escalates on its behalf.
3. **Escalation hijacking fix** — the model was calling `escalate` for non-complaint messages
   (price/delivery questions), burning the budget. Now rejected at the tool boundary.
4. **Contact grounding** — the model fills a missing phone field by inventing one. The loop drops
   any contact detail not present in the customer's message.
5. **Eval isolation** — run-scoped conversation ids keep repeats independent.

### Deliverables
- `README.md` — setup, architecture, endpoints, config
- `TECHNICAL_REPORT.md` — full design doc with real numbers
- `Makefile` — install, serve, test, eval, package
- `scripts/package_submission.py` — deterministic zip builder
- `scripts/compute_expected_totals.py` — independent oracle
- `evals/cases.jsonl` — 87 cases (13 seed + 74 authored)
- `tests/test_cases.py` — offline case validation
- SSE streaming for chat + eval progress
- Chat UI (`GET /`) and eval dashboard (`GET /eval`)

---

## 4. Current state

### Working
- 733 tests pass
- Service runs on `http://127.0.0.1:8000`
- Chat UI at `GET /`, eval dashboard at `GET /eval`
- SSE streaming for chat (`POST /chat/stream`) and eval (`POST /eval/stream`)
- Lead capture: 4/4 deterministic, no fabricated phone numbers
- Complaint escalation: 3/3 deterministic
- Submission zip built and pushed

### Known limitations (7B model ceiling)
- **Arithmetic:** 3-item totals, 40×1450 multiplication — the model gets these wrong
- **Policy keywords:** sometimes omits "2 hours", "4:00 pm" from replies
- **Case-authoring errors:** a few test expectations were wrong (fixed: lead-04, inject-12, unknown-03)

### Git log
```
a0c7ca0 Fix case-authoring errors: lead-04 allowed amounts, inject-12 substring, unknown-03 alternatives
7f26e3b Fix escalation hijacking: reject non-complaint escalate calls
68a7f4e Add deterministic complaint escalation, SSE streaming, and chat UI
e1225a2 Add evaluation harness, docs, packaging, and deterministic lead capture
ed14daf Add grounding, retrieval, LLM client, prompt and tool layers
3d448e8 Scaffold repo, externalised config, and frozen data contracts
```

---

## 5. How to run everything

### Setup
```powershell
cd C:\Users\Amandeep\Desktop\yashitask\meher-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

### Start the service
```powershell
# Terminal 1: Ollama
ollama pull qwen2.5:7b-instruct
ollama serve

# Terminal 2: Service
.\.venv\Scripts\python.exe -m uvicorn meher_agent.api.app:app --port 8000
```

### Test
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

### Evaluate
```powershell
.\.venv\Scripts\python.exe -m evals.runner evals\cases.jsonl --repeats 3 --out-dir reports
```

### Showcase
- Chat UI: http://127.0.0.1:8000
- Eval dashboard: http://127.0.0.1:8000/eval
- curl: `curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{\"conversation_id\":\"demo\",\"message\":\"How much is 2 kg Kaju Katli?\"}'`

### Package
```powershell
.\.venv\Scripts\python.exe scripts\package_submission.py
```

---

## 6. What to do next (priority order)

1. **Check eval results** — the 3-repeat eval on 87 cases should be finished. Read `reports/run-*.json`
   for final numbers. Fix any remaining case-authoring errors.
2. **Improve 7B reasoning gaps** — the arithmetic and policy keyword failures are model limitations.
   Options: better prompt engineering, few-shot examples, or a calculator tool (scope creep).
3. **Add auth + rate limiting** — currently none. CORS is not configured.
4. **Add persistence** — leads are in JSONL on disk (good), but conversations could be improved.
5. **Improve the chat UI** — add eval report tab, better styling, conversation history.
6. **Add token usage tracking** — currently `tok=n/a` because Ollama doesn't expose usage headers.

---

## 7. Key files reference

| File | Lines | Purpose |
|---|---|---|
| `src/meher_agent/agent/loop.py` | ~800 | The turn loop: budget, deterministic fallbacks, tool routing |
| `src/meher_agent/grounding/guard.py` | ~1075 | G1-G4 checks, intent detection, templates |
| `src/meher_agent/grounding/billing.py` | ~200 | Deterministic rupee totals |
| `src/meher_agent/grounding/amounts.py` | ~150 | G2 extractor, Indian formatting |
| `src/meher_agent/retrieval/pipeline.py` | ~300 | BM25 + bilingual retrieval |
| `src/meher_agent/api/app.py` | ~420 | FastAPI routes, SSE, chat UI, eval dashboard |
| `evals/runner.py` | ~540 | Eval harness with streaming callback |
| `evals/checks.py` | ~400 | G1-G4 + case assertions |
| `scripts/compute_expected_totals.py` | ~590 | Independent oracle (stdlib only) |
| `tests/test_loop.py` | ~970 | Loop tests (most important) |
| `tests/test_cases.py` | ~900 | Case validation tests |

---

## 8. Competitor comparison

We were compared against https://github.com/jaymittal611/-Customer-Query-Agent (llama3.1:8b).
**Ours is better.** Key differences:

| Dimension | Ours | Competitor |
|---|---|---|
| G2 (no invented amounts) | Actually works | Fake — `invented_amounts` never populated |
| Billing | Deterministic engine | LLM does all arithmetic |
| Lead capture | Deterministic, no fabrication | Regex fallback + strips rupee amounts |
| Complaint escalation | Deterministic | Keyword + substring match (false positives) |
| PII masking | API output + logs | Logs only |
| Tests | 733 | ~3 unit test files |
| Readme honesty | Accurate | Inflated scores (94% vs actual 87%) |
| Persistence | JSONL on disk | In-memory only |

The only things the competitor has that we don't: a better model (llama3.1:8b vs qwen2.5:7b)
and a prettier eval UI. We added SSE eval streaming and an eval dashboard to close the UI gap.

---

## 9. Environment

- **OS:** Windows 11
- **Python:** 3.13 (venv at `.venv`)
- **Model:** Ollama `qwen2.5:7b-instruct` at `http://localhost:11434/v1`
- **Service:** `http://127.0.0.1:8000`
- **Git:** `main` branch, remote `https://github.com/Amandeep-Singh-Khalsa/meher-agent.git`
- **Submission zip:** `C:\Users\Amandeep\Desktop\yashitask\dhanur-task-yashi-gupta.zip`
