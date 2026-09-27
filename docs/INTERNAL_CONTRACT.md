# Internal contract

Every module below must expose exactly these public names. The point of this file
is that six pieces of work were developed in parallel without blocking each
other: each one codes against the signatures below and nothing else.

Repo root: `C:\Users\Amandeep\Desktop\yashitask\meher-agent`
Python: `.venv\Scripts\python.exe` (3.13). Run with `$env:PYTHONPATH="<root>\src"`.
Never modify `data/` or `evals/seed_cases.jsonl`.

## Already written (do not rewrite)

| Module | Exports |
|---|---|
| `meher_agent.types` | `SKU`, `Section`, `RetrievedSection`, `RetrievedSKU`, `ResolvedOrderItem`, `Resolution`, `Briefing`, `LineItem`, `Quote`, `Action`, `Lead`, `ToolResult`, `AgentOutcome`, `ChatRequest`, `ChatResponse` |
| `meher_agent.config` | `Config`, `load_config`, `get_config`, and the `POLICY_*` constants |
| `meher_agent.data.corpus` | `Corpus`, `load_corpus`, `slugify_section` |
| `meher_agent.grounding.amounts` | `extract_rupee_amounts`, `has_rupee_amount_outside`, `normalise_for_match`, `contains_any/all/none`, `ascii_digits`, `format_inr`, `to_int`, `strip_thousands` |
| `meher_agent.tools.validation` | `ValidationError`, `normalise_phone/email/date/name`, `is_valid_*`, `clean_optional_text` |
| `meher_agent.agent.services` | `Services`, `build_services`, `get_services` |

## To be written, by owner

### A — retrieval (`meher_agent.retrieval`)
```python
# text/translit.py
fold_devanagari(text: str) -> str        # Devanagari -> Latin transliteration, lossless enough to match the CSV
devanagari_ratio(text: str) -> float     # 0.0-1.0 share of Devanagari code points
detect_language(text: str) -> str        # "hindi" | "hinglish" | "english"

# text/lexicon.py
SYNONYMS: dict[str, tuple[str, ...]]     # canonical key -> matching surface forms, incl. Hindi/Hinglish
expand(token: str) -> set[str]           # token + its synonym cluster

# retrieval/retriever.py
class Retriever:
    def __init__(self, corpus: Corpus, config: RetrievalConfig) -> None
    def sections(self, query: str, top_k: int | None = None) -> list[RetrievedSection]
    def skus(self, query: str, top_k: int | None = None) -> list[RetrievedSKU]

# retrieval/resolver.py
class OrderResolver:
    def __init__(self, corpus: Corpus, config: RetrievalConfig) -> None
    def resolve(self, message: str) -> Resolution     # items + distance_km + unresolved notes

# retrieval/pipeline.py
class RetrievalPipeline:
    def __init__(self, corpus: Corpus, config: Config) -> None
    def brief(self, message: str, history: list[TurnRecord]) -> Briefing
```
`RetrievalPipeline.brief` is the only entry point the agent uses. It must set
`Briefing.language`, `Briefing.resolved`, and `Briefing.allowed_amounts`
(= sorted catalog prices + `[60, 999, 5000]` + any computed quote values).

### B — grounding (`meher_agent.grounding`)
```python
class BillingEngine:
    def __init__(self, corpus: Corpus, config: Config) -> None
    def quote(self, resolution: Resolution, *, today: date | None = None) -> Quote

class ReplyGuard:
    def __init__(self, corpus: Corpus, config: Config) -> None
    def allowed_amounts(self, quote: Quote | None) -> set[int]
    def inspect(self, reply: str, *, briefing: Briefing, quote: Quote | None,
                is_first_turn: bool) -> GuardVerdict
    def repair(self, reply: str, verdict: GuardVerdict, *, briefing: Briefing,
               quote: Quote | None) -> str    # deterministic, guaranteed-safe rewrite
```
`GuardVerdict` (new model, put in `grounding/guard.py`): `ok: bool`,
`violations: list[str]`, `invented_amounts: list[int]`.

Billing rules, straight from `data/policies.md`:
- line totals = `qty * unit_price`, integers only
- discount: 5% **of the gift-box subtotal only**, and only when gift-box qty >= 50
- delivery: free when `subtotal - discount >= 999`, else 60; impossible beyond 8 km
- advance: 30% when total weight > 10 kg **or** gift boxes > 25
- cash on delivery limit 5000; GST is already included in every price

### C — llm / tools / agent (`meher_agent.llm`, `.tools`, `.agent`)
```python
# llm/client.py
class LLMClient:
    def __init__(self, config: Config) -> None
    def complete(self, messages: list[dict], *, tools: list[dict] | None = None,
                 tool_choice: str | dict | None = None) -> LLMResponse
# LLMResponse: .content: str, .tool_calls: list[ToolCallRequest], .usage: Usage,
#              .finish_reason: str, .raw: dict
# ToolCallRequest: .id: str, .name: str, .arguments: dict   (arguments parsed, never raising)

# llm/prompts.py
SYSTEM_PROMPT: str
build_messages(briefing: Briefing, history: list[TurnRecord]) -> list[dict]
tool_schemas() -> list[dict]        # OpenAI function-calling schema for the two tools
NUDGE_TOOL: str                    # text for the intent-repair model call

# tools/registry.py
class ToolRegistry:
    def __init__(self, corpus: Corpus, config: Config) -> None
    def names(self) -> list[str]
    def run(self, name: str, args: dict, *, conversation_id: str) -> ToolResult
# A ToolResult(ok=False) is returned to the model as a `tool` message. Never raise.

# tools/store.py
class LeadStore:
    def add(self, lead: Lead) -> Lead
    def all(self) -> list[Lead]
    def clear(self) -> None

# agent/conversation.py
class ConversationStore:
    def __init__(self, *, max_conversations: int, max_history_turns: int) -> None
    def get(self, conversation_id: str) -> Conversation
# Conversation: .conversation_id, .turns: list[TurnRecord], .handoff: bool,
#              .reply_count: int, .last_language: str

# agent/loop.py
def run_turn(message: str, conversation_id: str, services: Services) -> AgentOutcome
```
`run_turn` is the single entry point for the whole system. It must:
1. load the conversation, resolve/retrieve a `Briefing`, compute a `Quote`
2. run at most `config.llm.max_steps` model calls, handling tool calls
3. if the turn looks like a lead/complaint and no tool fired, spend one call on
   `NUDGE_TOOL` (counts against the same 4)
4. guard the final reply, one repair call if `allow_reply_repair`
5. on step exhaustion, call `escalate` implicitly and set `handoff=True`
6. populate every diagnostic field on `AgentOutcome`

### D — safety / logging / api
```python
# safety/pii.py
def mask_email(value: str | None) -> str | None      # ritu.m@example.com -> r*****@example.com
def mask_phone(value: str | None) -> str | None      # 9876543210 -> ******3210
def mask_text(text: str) -> str                       # scrubs every email+phone in free text
def mask_value(value: str | None) -> str | None      # dispatch on shape
# logging_utils.py
class PiiMaskingFilter(logging.Filter)                # masks record.msg and record.args
def setup_logging(level: str) -> None

# api/app.py
app: FastAPI
POST /chat   -> ChatResponse   {reply, sources, actions, handoff}
GET  /leads  -> list[dict]     masked, each has name/email/phone/need/conversation_id/created_at
GET  /health -> {status, ...}
```
`Action.to_public()` returns `{"type": ..., "args": ...}`; that is the exact
shape placed in the response's `actions` array.

### E — evaluation harness (`evals/`, importable as a top-level package)
```python
# evals/checks.py
def normalise(text: str) -> str                # re-use grounding.amounts
def check_case(case: dict, result: CaseResult) -> list[CheckOutcome]
# CheckOutcome: .name, .passed, .detail
# G1 first reply contains "AI"; G2 rupee amounts subset of allowed;
# G3 1 <= len(reply) <= 1200; G4 sources non-empty and valid, for
#    categories fact/price/arithmetic/policy/hindi/hinglish
# evals/runner.py
def run_cases(cases_path: Path, *, base_url: str, repeats: int, out_dir: Path) -> RunSummary
# evals/report.py
def write_reports(runs: list[RunSummary], out_dir: Path) -> tuple[Path, Path]
```
The harness must be a **standalone package** runnable as
`python -m evals.runner --cases evals/cases.jsonl` from the repo root with
`src` importable, and must not import anything from `api/`.

### F — scripts + cases
```python
# scripts/compute_expected_totals.py
def compute(lines: list[tuple[str, int]], distance_km: int | None) -> dict
# scripts/build_report.py -> TECHNICAL_REPORT.pdf via reportlab
# scripts/package_submission.py -> dhanur-task-yashi-gupta.zip
```
`compute_expected_totals.py` must be an **independent** re-implementation of
the pricing rules reading only `data/prices.csv` — it must NOT import
`meher_agent.grounding.billing`, so that the expected totals in `evals/cases.jsonl`
are an independent oracle rather than a restatement of the agent's own logic.

## Response contract (fixed by the task spec)

```json
{"reply": "...", "sources": ["prices.csv#KK-1000"],
 "actions": [{"type": "save_lead", "args": {"name": "Ritu Malhotra", "email": "ritu.m@example.com"}}],
 "handoff": false}
```

## Rules for everyone

- No comments explaining *what* the code does; only *why* where non-obvious.
- No network calls in unit tests. Stub the LLM.
- Every module gets type hints on public functions.
- Run `.venv\Scripts\python.exe -m pytest tests -q` before you report done.
- `git add`/`commit` is handled centrally. Do not run git.
