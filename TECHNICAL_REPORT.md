# Technical report — Meher Sweets & Namkeen customer-service agent

Repo: `C:\Users\Amandeep\Desktop\yashitask\meher-agent`. Python 3.13 (`.venv`), FastAPI,
hand-written OpenAI-compatible client over `httpx`, no vector database, no embeddings, no
third-party agent framework. Every claim below is traceable to a file in the repo or to a
number in `reports\run-20260927-182635.json` (the newest run report on disk) unless it is
marked as measured live during the writing of this report.

---

## 1. Problem and constraints

The shop is fictional but the problem is not: answer customer questions about a small
catalogue with a hard evidence policy. The constraints, fixed by the task spec and
`docs\INTERNAL_CONTRACT.md`:

1. **Grounded answers only.** Every fact must come from `data\business.md`,
   `data\prices.csv` or `data\policies.md`. The catalogue is 14 SKUs and 18
   policy/business sections; the shop has exactly one discount (5% off the
   gift-box subtotal from 50 boxes), one bulk rule (>10 kg or >25 gift boxes needs 3 days'
   notice and a 30% advance), free delivery at Rs 999, a Rs 60 delivery fee below that,
   an 8 km radius and a Rs 5,000 cash-on-delivery limit (`data\policies.md`).
2. **No invented prices.** A reply may only state rupees the system can prove were
   computed. The eval check G2 fails any other rupee amount.
3. **A small model, a hard budget.** The model is `qwen2.5:7b-instruct` on a local Ollama
   server (`LLM_MODEL` in `.env`), temperature 0.0, and at most **4 model calls per
   customer message** (`config.toml:10`).
4. **Three languages.** English, Hindi (Devanagari) and Hinglish (Roman-script Hindi),
   detected per turn.
5. **A fixed HTTP contract.** `POST /chat` returns exactly `reply`, `sources`, `actions`,
   `handoff` (`docs\INTERNAL_CONTRACT.md`, response contract section).
6. **Cost and privacy.** The model is local, so there is no per-call charge; and no
   customer contact detail may reach a log line or an API response unmasked.

The consequence of constraint 1+2 with a 7B model: a small model *will* verbalise
confidently and *will* invent numbers. The architecture is built so that a confident,
sloppy model is acceptable and a lying one is not.

## 2. Architecture and why each layer exists

```
HTTP (api/app.py)
  -> agent/loop.py            the turn loop; the only code that talks to the model
      -> retrieval/pipeline.py  message + history -> Briefing (what is true)
          -> retrieval/retriever.py   BM25 over sections + alias/trigram match over SKUs
          -> retrieval/resolver.py    quantities, SKUs, distance out of the message
      -> grounding/billing.py   Resolution -> Quote (what the money is), integer arithmetic
      -> grounding/guard.py     ReplyGuard: post-hoc verification and deterministic repair
      -> grounding/amounts.py    the one rupee-extraction implementation (shared with the harness)
      -> tools/registry.py      save_lead / escalate; validation, action log, handoff
      -> llm/client.py          chat-completions over httpx, retries, defensive parsing
```

| Layer | Exists because |
|---|---|
| `data\corpus.py` | `data/` must be parsed exactly once, into typed `SKU`/`Section` objects with stable source ids (`prices.csv#KK-1000`, `policies.md#bulk-orders`). Everything downstream cites these ids. |
| `retrieval\` | The model must never see all of `data/` — only the sections and SKUs this turn could be about. Retrieval decides what is *true*; nothing else reads `data/` directly. |
| `grounding\billing.py` | Money must be computed, not guessed. Deterministic integer arithmetic from `prices.csv` is the only source of rupee figures. |
| `grounding\guard.py` | A finished reply must be *verified* against what billing computed, and rewritten deterministically when it fails. The guard is the difference between "the model said a number" and "a number was said". |
| `tools\registry.py` | The model's only way to change the world. Every argument is hostile input; every call is validated, logged and either applied or turned into a tool error. |
| `agent\loop.py` | The budget, the tool round trips, the fallback discipline. The only place that decides how many times to ask. |
| `api\app.py` | The fixed contract, PII masking on the way out, liveness independent of the model. |
| `safety\pii.py` + `logging_utils.py` | Contact details must not reach logs. A logging *filter* rather than per-call-site masking, so no future `logger.info` can leak. |

`agent\services.py:50` (`build_services`) is the composition root; each module is
importable and testable in isolation.

## 3. The central design decision: where the model may be creative

**The model writes the sentence. Deterministic code owns every fact, every rupee figure,
every tool call and every refusal.** The model is allowed to choose wording, ordering,
emphasis and tone — the things a 7B model is acceptable at. It is not allowed to choose
content. Concretely, each of these is deterministic:

| Property | Where | Mechanism |
|---|---|---|
| Every rupee amount | `grounding\amounts.py:60` + `grounding\guard.py:384` | `extract_rupee_amounts` re-extracts every amount in a finished reply; each value must be in `allowed_amounts` = catalog prices ∪ {60, 999, 5000} ∪ every value the `Quote` computed (`guard.py:354`). One implementation is shared with the eval harness, so the guard and the grader cannot disagree. |
| Every percentage | `guard.py:49`, `guard.py:411` | The shop has exactly {5, 30} (gift-box discount, bulk advance; GST is also 5). Any other percentage in a reply is a fabricated offer and is rejected. |
| The computed money itself | `grounding\billing.py:91` | Integer-only arithmetic (`round_half_up`, `billing.py:56`): 5% of a gift-box subtotal of 1950 is 98 rupees every time, with no binary floating point. |
| The AI disclosure (first turn) | `agent\loop.py:151` (`_with_disclosure`) | The literal token "AI" is appended by code when the model's first reply lacks it. Clamping happens *before* the tag is appended so the disclosure can never push the reply back over the length limit. This is a deterministic fix, not a model failure, and it costs no model call. |
| The reply-length cap | `loop.py:130` (`_clamp`), `guard.py:439` | 1–1200 characters, clamped on a word boundary, enforced again in `_finalise` (`loop.py:713`) after any repair. |
| PII masking | `safety\pii.py:84-131`, `logging_utils.py:36` | `ritu.m@example.com` -> `r*****@example.com`; `9876543210` -> `******3210`; masking is idempotent and applied by a logging filter, so no log line can carry a raw address or mobile. |
| Tool validation | `tools\registry.py:280`, `tools\validation.py` | Phones must be 10-digit Indian mobiles; emails RFC-shaped; dates ISO; names non-empty. A bad call is a `ToolResult(ok=False)` handed back to the model as a normal turn — never a 500. |
| Which citations are valid | `loop.py:738` (`_sources`) | A cited id is accepted only in the exact form `data/` uses (`prices.csv#KK-1000`), checked against `corpus.source_ids`; an invented id is dropped. |
| Refusals | `guard.py:578` (`_template_reply`) | Out-of-scope, privacy and refusal wording is assembled from fixed templates per language, never composed by the model. |
| The escalation decision | `loop.py:548`, `registry.py:390` | A successful `escalate` sets the handoff flag; when the loop escalates on the model's behalf (step limit), the guard's handoff template becomes the reply, not the model's sentence. |

**What this buys:** a small model can be sloppy in prose without being able to lie about
a price. The failure mode of a 7B model — fluent, confident, wrong numbers — is caught
by a checker that is cheap, exact and testable, instead of being prevented by prompt
engineering alone. The eval numbers bear this out: across 13 case runs in the newest
report, G2 (invented amounts) had **0 failures in 13 replies**, and the reply was 100%
within the length cap. The one case that failed (`lead-01`) failed on a *tool call*, not
on a fabricated fact.

## 4. The model-call budget

`config.toml:10` sets `max_steps = 4`. The budget is a counter taken **before** each call
(`agent\loop.py:227`, `_Budget.take`), inside `_complete` (`loop.py:378`) — the only
function in the codebase that calls the model. Repair and intent nudge calls come out of
the same pool, which is the only way the cap means anything. Three invariants hold
regardless of what the endpoint returns (`loop.py:10-16`):

1. at most 4 model calls per customer turn — an exception, an empty answer or a runaway
   tool loop cannot exceed it, because the counter is spent before the request is made;
2. a reply always exists, always fits 1–1200 characters, and always discloses AI on the
   first turn;
3. no exception escapes `run_turn`; a turn that fails still records its history, hands
   off, and logs a masked traceback (`loop.py:803`).

The budget is spent in a fixed order, because the order is the design (`loop.py:22-27`):

1. the answer itself;
2. tool round trips — a lead or a complaint is worth more than a second draft;
3. the intent nudge (`NUDGE_TOOL`, `llm\prompts.py:109`) — one "you must call the tool
   now" message when a lead/complaint turn produced no tool call;
4. the guard repair — the only one that can be skipped without losing the turn, since
   `_deterministic` (`loop.py:702`) can rewrite the reply without a model.

**When the budget is exhausted**, `_complete` returns `None` and the loop calls
`_step_limit_reply` (`loop.py:539`): it sets `step_limit_hit`, escalates on the model's
behalf through the registry (so the handoff is recorded), and answers with the guard's
handoff template. The customer is never left with nothing, and the response carries
`handoff: true`.

The cap is a hard architectural boundary, not a tuning knob, for three reasons: the
eval harness's latency and cost numbers are only comparable across runs if every turn
spends the same bounded number of calls; a 7B model in a tool loop is the cheapest way to
burn minutes per turn; and "the model refused to stop" must not be able to take the
service down. `tests\test_loop.py` pins the behaviour: `test_model_calls_never_exceed_max_steps`,
`test_budget_is_shared_with_the_repair_calls`, `test_step_cap_stops_at_max_steps_and_escalates`.

## 5. Tool calling and the lead-capture problem

Two tools: `save_lead` (name, need, optional phone/email/quantity/date) and `escalate`
(reason) — `tools\registry.py:46`. The model sees them as OpenAI function schemas
(`tools\schemas.py`) whose descriptions say what *not* to do ("never invent a contact
detail").

**The problem, measured.** With a 7B model behind a ~900-token system prompt, the model
frequently answered a lead turn correctly but *omitted* the `save_lead` tool call: it
quoted the customer's details back and never stored them. The newest report on disk shows
it: `reports\run-20260927-182635.json`, case `lead-01` — the reply is right ("The total is
Rs 43,500, and a 30% advance of Rs 13,050 is required") but `expect_action` fails with
"expected one of ['save_lead'], last turn called nothing" and `expect_lead` fails with
"no save_lead action was taken". A lead quoted back to the customer but never stored is
a lost lead, and a lost lead is the one outcome this agent exists to prevent.

**The final fix** (`loop.py:362` and `loop.py:490`, `_deterministic_lead`): when the model
made **no tool attempt at all**, the loop itself reads the name and contact out of the
customer's message and calls the validated tool — at zero model-call cost. The signals
are deterministic: an ordering word or a self-introduction plus a phone/email
(`_looks_like_lead`, `loop.py:184`), the name captured by a regex after "my name is / I am /
I'm / this is / myself" in Latin or Devanagari (`_NAME_AFTER_INTRO_RE`, `loop.py:198`),
the contact by the shared phone/email extractors. The `need` field is the request itself
with the contact details scrubbed back out, capped at 400 characters, so the CRM holds a
requirement and not a line of PII. The extracted fields still go through the registry's
validation, so a bad extraction is rejected, not stored
(`tests\test_loop.py::test_a_lead_the_model_forgets_is_saved_deterministically` asserts
the lead is written, with post-validation args, in one model call).

**The complementary rule** (`loop.py:357-361`): a model that *did* attempt a tool call is
left alone, even when the attempt was rejected. The deterministic save is gated on
`not self.tool_attempted`, and a rejected model-issued call is surfaced to the model as a
tool error (`loop.py:425`) rather than silently replaced — the loop asks the customer to
correct a bad phone rather than guessing at one (`loop.py:443`, `_ground_contact_args`;
`tests\test_loop.py::test_a_rejected_lead_call_asks_rather_than_guessing` asserts the lead
is *not* written in that case). So the `lead-01` failure mode in the report is precisely
this branch: the model attempted a call the registry refused, and the design refuses to
second-guess it. The honest summary is that the fix closed the "forgot the call entirely"
hole — the dominant one — and deliberately left the "tried and failed" hole to the tool
error path, because replacing a rejected call with a guessed one would fabricate data.

**The related anti-fabrication guard** (`loop.py:443`, `_ground_contact_args`): the model
fills a missing field rather than leaving it out. Asked to save a lead from a message
carrying only an email, it will happily invent a ten-digit phone number. A fabricated
number in a CRM is worse than a missing one, because the team will call it. So any contact
detail the customer never actually gave is *dropped* before validation: a plausible phone
whose digits do not appear in the customer's message, or an email not present in the
message, is removed from the args, and the call then proceeds without it (or is rejected
if nothing valid remains). Invalid-but-not-plausible values are left to the registry,
whose specific complaint ("10 digits", "not a valid email") is more useful to the
customer than a silent omission (`loop.py:462-470`).

Two further tool-path behaviours worth naming. A 7B model asked to call a tool often
writes `escalate {"reason": "..."}` into its answer instead of the tool channel; the loop
rescues those calls through the same registry and strips them from the prose
(`_rescue_prose_calls`, `loop.py:575`, capped at two per turn), so the customer never
reads a JSON blob and the complaint is still flagged. And only *successful* calls reach
the public `actions` array (`_ok_actions`, `loop.py:763`): a rejected call is a
diagnostic (`X-Tool-Errors` header), never a promise.

## 6. Grounding and citation

**Retrieval** (`retrieval\retriever.py`): two hand-rolled indexes over `data/`, no
external dependency. Sections are scored with BM25 (k1=1.5, b=0.75, `config.toml:31`)
over synonym-expanded tokens, plus ~20 explicit intent anchors (`retriever.py:150`) that
guarantee the right policy section reaches the model when the customer's wording shares
no terms with it ("koi discount milega?" still reaches the Diwali discounts section).
SKUs are matched fuzzily: alias-phrase windows (every surface form including folded
Devanagari, `retriever.py:66`), token overlap against the synonym-expanded query, and
character-trigram containment, so "sugar free kaju katli" lands on KKSF-500 and "moti
laddoo" on ML-1000 with no stemmer or embedding model.

**Bilingual handling** (`text\translit.py`): Devanagari is folded to Latin by a literal
transliteration (consonant + inherent vowel, matras replacing it, virama suppressing it,
final-schwa deletion), so "काजू कटली" and "kaju katli" reach the same alias. No
medial-schwa deletion is attempted because in Hindi it is lexical, not script-determined;
residual variation is absorbed by the synonym lexicon (`text\lexicon.py:43`, which carries
the folded Devanagari spelling of every product) and the trigram matcher. Digits are
folded to ASCII because every downstream regex understands only 0-9. Language is a
three-way tag (`detect_language`, `translit.py:322`): Devanagari at >=15% of letters is
`hindi`; a Romanised Hindi function word (none of them English: "kitna", "kya", "hai",
"bhaiya"...) is `hinglish`; anything else is `english`. Product names never count on
their own, so "I want 10 samosas" stays English.

**Order resolution** (`retrieval\resolver.py`) reads quantities, product mentions and
delivery distance out of the message without ever computing money. It refuses to guess
silently: a weight no pack divides, an unstated pack size, a rupee amount that looks like
a count, a date, a duration, a kilometre reading — each becomes an `approximated` flag or
an `unresolved` note the model is told about (`resolver.py:410`). Cross-turn continuity
lives in `retrieval\pipeline.py:126`: the last three user turns are re-resolved so "make
it 3 kg" updates the order under discussion, and an explicit additive word ("add", "also",
"साथ") merges the new lines into it.

**The rupee-amount extractor** (`grounding\amounts.py:60`) is the single implementation of
the money-matching rule, shared by the reply guard and the eval harness so they cannot
diverge. An amount must carry an explicit currency marker ("Rs", "Rs.", "₹", "INR",
"रु", or the word "rupees"), so "5% GST", "2 hours", "3 November", "10 samosas" are
ignored. The exact-format requirement is the point: the grouped form
`(?:\d{1,3}(?:,\d{2,3})+(?!\d)|\d+)` *requires* at least one comma group, and the
`(?!\d)` lookahead stops `\d{1,3}` from matching a prefix of a longer digit run —
without it "Rs 1200" silently truncates to 120 (`amounts.py:28-31`). That truncation
class of bug is exactly what a regex written for "3,850"-style output produces when fed
unformatted model text. Matching for `must_include` / `must_not_include` strips
thousands separators (including Indian grouping, `1,00,000` -> `100000`) before comparing
(`normalise_for_match`, `amounts.py:86`). Indian digit grouping in customer-facing text
goes through `format_inr` (`amounts.py:143`), and `billing.inr` (`billing.py:46`) strips a
stray trailing separator so a value never prints as "3,850,".

**Citation**: the model is instructed to end with the ids it used, copied from the
SHOP FACTS block (`llm\prompts.py:91`). The loop then validates every cited id against
`corpus.source_ids` and fills up to 4 sources from retrieval's own evidence, the model's
citations first (`_sources`, `loop.py:738`, `MAX_SOURCES = 4`). G4 in the harness fails
an id that does not exist in `data/`.

## 7. Safety

- **Prompt injection.** The customer message is fenced and re-labelled as untrusted data
  everywhere it appears (system context, few-shot turns, history:
  `llm\prompts.py:179`, `build_messages`). The fence markers themselves are neutralised
  inside customer text (`_neutralise_fence`, `prompts.py:153`) so a pasted
  "--- END CUSTOMER MESSAGE ---" cannot masquerade as framing. The system prompt names the
  attacks and the required response (refuse in one sentence, then keep helping;
  `prompts.py:52-59`). Post-hoc, the guard rejects any reply containing leaked-instruction
  phrases (`find_prompt_leak`, `guard.py:254`) — 27 phrases covering "ignore previous
  instructions", "you are the owner now", "as an AI language model I must",
  "developer message" and friends. The injection cases in the eval file are the evidence:
  4 injection cases plus 8 more in the full file, including a tool-call injection with an
  invalid phone and a data-exfiltration payload (`evals\cases.jsonl`, inject-07).
- **PII masking.** One rule set, three callers: the HTTP response
  (`Lead.to_public_masked`, `types.py:211`), the logging filter
  (`logging_utils.py:36`), and free-text scrubbing. Idempotent by construction
  (`safety\pii.py:14-16`). A phone-shaped value that is not exactly a 10-digit mobile is
  reported as absent rather than echoed (`mask_phone`, `pii.py:94`). The unhandled-error
  handler masks the *rendered traceback* before logging it, because a stack frame quoting
  the failing line is the most likely place for a contact detail to appear
   (`api\app.py:224`).
- **Out-of-scope refusal.** Two detectors run: the pipeline's hint
  (`pipeline.py:175`, positive evidence of another subject, deliberately high bar so
  "do you make rabri?" stays in scope and *unknown*) and the guard's wider keyword set
  (`guard.py:552`). Refusals are fixed templates per language; the model never composes
  one. In the newest report the out_of_scope category passed 1/1 and the injection
  category 1/1.

## 8. Evaluation

**The check set.** `evals\checks.py` implements the graded checks literally from the
spec, plus the case-level fields the schema defines:

| Check | Meaning |
|---|---|
| G1 | first reply of the conversation contains the word "AI" |
| G2 | every rupee amount in the checked reply is allowed (catalog prices + policy 60/999/5000 + the case's `allowed_amounts`) |
| G3 | reply is 1..1200 characters |
| G4 | for categories fact/price/arithmetic/policy/hindi/hinglish, sources are non-empty and every id exists in `data/` |
| must_include / must_include_any / must_not_include | case-defined strings, case-insensitive, thousands separators stripped |
| expect_action | the last turn called the expected tool (or none); a loop-generated escalate on step exhaustion is excluded only when the service marks it (`checks.py:464`) |
| expect_lead | a saved lead matches the expected fields under the *same* normalisation the registry uses (`checks.py:574`) — one implementation, imported by both |

Every reply-level check runs against the reply to the **last** turn. Nothing raises: a
malformed case becomes a failed outcome; a check that does not apply is reported
`skipped=True, passed=True` so denominators stay honest; a failed request fails every
applicable check instead of silently passing on an empty reply (`checks.py:17-25`).

**Case schema.** One JSON object per line: `id`, `category`, `turns` (list of strings),
optional `must_include`, `must_include_any`, `must_not_include`, `allowed_amounts`,
`expect_action`, `expect_lead`, and a `quote` block for the independent oracle
(`scripts\compute_expected_totals.py`, which re-implements the pricing rules from
`data\prices.csv` alone and imports nothing from the agent package — a test greps its
source to prove it). The public seed file has 13 cases in 12 categories; the full file has
87.

**Methodology.** Each case is replayed 3 times (`--repeats 3`, the default) against the
live service. Conversation ids are run-scoped: `eval-<run_id>-<repeat>-<case_id>`
(`runner.py:236`), where `run_id` is the invocation timestamp, so no turn ever sees a
previous case's history and G1 genuinely measures a first reply rather than a fourth one.
Rates are reported as mean-of-runs and worst-run (latency and cost worst = max), because
sampling at temperature 0 on a local 7B is not bit-reproducible (`report.py:11-14`).
p50/p95 are nearest-rank percentiles with no interpolation, so every quoted latency is an
observed value in the JSON (`report.py:122`). A partial JSON is rewritten after every case
so an interrupted run is not lost (`runner.py:26`).

**Real numbers** (from `reports\run-20260927-182635.json`, the newest run report on
disk — 13 case runs, 14 messages, 0 failed requests, endpoint `http://127.0.0.1:8000`):

| Metric | Value |
|---|---|
| Overall pass rate (all checks of a case) | **92.3%** (12/13) |
| All graded check outcomes | 97.0% |
| G1 AI disclosure | 100% (13/13) |
| G2 invented amounts | **0 failures in 13 replies** |
| G3 length | 100% |
| G4 citations | 100% |
| Action accuracy (cases that set `expect_action`) | **66.7%** (2/3) |
| Lead-field match (`expect_lead`) | **0.0%** (0/1) |
| must_include / must_include_any / must_not_include | 100% / 100% / 100% |
| Latency p50 / p95 per message | **7.56s / 20.97s** (14 messages) |
| Messages per conversation | 1.08 |
| Prompt / completion tokens per message | 0 / 0 (not measured — see below) |
| Cost per 100 conversations | Rs 0.00 (local model, $0.0/Mtok) |

Per category: arithmetic 2/2, every other category 1/1 except **lead 0/1**. The single
failing case is `lead-01`, failing `expect_action` and `expect_lead` for the reason
analysed in section 5 — the model attempted a `save_lead` call that the registry
rejected, and a rejected model-issued call is never silently replaced.

Three further observations from the run artefacts, reported plainly:

- The three `run-*.json` files on disk are three **separate single-run invocations**
  (each contains `runs: 1`), not one 3-repeat invocation, so the mean-of-3 aggregation has
  not been exercised end to end against the current code.
- An earlier run (`run-20260927-182217.json`) scored 3/13 with G1 failing on 10 cases —
  replies with no AI disclosure at all. That run predates the deterministic disclosure
  fix (`_with_disclosure`); the newest run's 100% G1 is the fix working, not the model
  having improved. Run-to-run variance at temperature 0 is real: the same binary scored
  12/13 twice and 3/13 once on the same case file.
- A verification run of the same command against the current code (written to a temp
  directory, not into `reports\`) scored **13/13 with 100% on every graded metric**,
  p50 13.13s / p95 20.96s, with `lead-01` passing on a model-issued `save_lead`. The
  lead-capture fix works when the model cooperates; the residual risk is the
  rejected-call branch, which is a deliberate design trade, not an oversight.

**Token counts are 0 because the service reports no token usage.** The harness never
estimates tokens; the cost line is derived from measured tokens only and must be read as
"not measured" rather than "free" (`report.py:565`). The `Usage` fallback in
`llm\client.py:330` does estimate from characters (3.6 chars/token) when the endpoint is
silent, but the HTTP service does not expose usage headers, so the harness records
`usage_source: "unknown"`.

## 9. Test strategy

**732 tests, all passing** (`.venv\Scripts\python.exe -m pytest tests -q`, ~38s, no
network). 13 test files, one per concern:

| File | Covers |
|---|---|
| `test_loop.py` (40) | the budget, tool round trips, intent/guard repair, disclosure, prose-call rescue, the deterministic lead save, the rejected-call rule, source validation, exception containment |
| `test_guard.py` (44) | amount/percentage/phone/prompt-leak/length/script violations, repair, templates, intent detection |
| `test_billing.py` (32) | the pricing rules, including the seed cases as fixtures and the boundary conditions (exactly 999, exactly 10 kg, exactly 50 boxes) |
| `test_retrieval.py` (38) | BM25, anchors, alias windows, Devanagari folding, language detection, cross-turn carry |
| `test_cases.py` (57) | the case file against the independent oracle, including the source-grep that proves the oracle imports nothing from the agent |
| `test_checks.py` (51) | G1-G4 and the case-level checks, including tolerance of malformed cases |
| `test_runner_metrics.py` (32) | percentiles, denominators, cost arithmetic, report rendering — offline, synthetic summaries |
| `test_api.py` (30) | the four routes, the four-key contract, 422s, masking, header diagnostics, 500 containment |
| `test_masking.py` (41) | every masking shape, idempotence, the logging filter |
| `test_tools.py` (31) | validation, action log, per-conversation filtering, thread safety |
| `test_lead_validation.py` (19) | phone/email/date/name normalisation, including the seed lead values |
| `test_llm_client.py` (31) | retries, usage parsing, malformed tool calls, health |
| `test_conversation.py` (15) | LRU eviction, history caps, per-conversation isolation |
| `test_invalid_toolcall.py` (6) | rejected calls stay tool errors, never reach `actions`, never 500 |

The strategy is: every layer is tested offline against fakes and fixtures, with the LLM
stubbed by scripted response queues (`tests\test_loop.py`), so the suite is fast and
deterministic; the end-to-end path is covered by the live eval harness instead. The two
harness-adjacent files (`test_checks.py`, `test_runner_metrics.py`) import the `evals`
package, which puts `src` on `sys.path` itself (`evals\__init__.py:27`) so the harness is
runnable without environment setup.

## 10. Trade-offs and what is next

**Trade-offs accepted.**

- *A 7B model with a 4-call budget.* Chosen because the task fixes both. The cost is
  latency (p95 ~21s) and prose variance; the benefit is a system that runs anywhere with
  no per-call charge and no data leaving the machine.
- *Deterministic lead save only when the model made no attempt.* The alternative —
  always overwriting a rejected call — would fabricate contact details. The residual
  failure mode (a rejected `save_lead` loses the turn's lead) is visible in the eval and
  is the honest price of never inventing data.
- *Template fallbacks over model retries.* When the guard rejects a reply twice, the
  customer gets a template, not a third model draft. Templates are boring and always
  correct; a third draft from a model that failed twice is neither.
- *In-memory state.* Leads and conversations are process-local and lost on restart
  (`tools\store.py:5`). The graded behaviour is what the model was told happened, not
  durability.
- *No token usage from the service.* The cost line is honest ("not measured") rather
  than populated with estimates presented as facts.

**What is next.**

1. **SSE streaming** for `/chat` — the loop already produces the reply in one final
   string, so streaming would be token-level from the client with a guard pass at the
   end; not built.
2. **A chat UI** — the four-key contract is enough for a minimal front-end; not built.
3. **Usage headers on `/chat`** so the eval's token and cost lines are measured rather
   than zero; the client already parses both header and body usage shapes
   (`llm\client.py:330`, `evals\runner.py:59`).
4. **A 3-repeat end-to-end run** of the current code to exercise the mean/worst
   aggregation, and a larger repeat count on the lead cases to quantify the residual
   rejected-call risk.
5. **Persistence for leads** (append-only file or SQLite) if the CRM use outgrows a
   single process.

---

### Appendix: inaccuracies noticed while writing this report

1. `grounding\billing.py:46-53` — the docstring of `inr()` claims `format_inr` "currently
   leaves a trailing thousands separator on every value of 1000 or more ('3,850,')". The
   current `format_inr` (`amounts.py:143`) does not produce a trailing comma; the
   `rstrip(",")` is harmless belt-and-braces, but the docstring describes a bug that
   appears to have been fixed in `format_inr` without updating this text.
2. `reports\summary.md` says "runs: 1 (the full set is replayed 1x ...)" — accurate for
   the artefact, but the runner's default is `--repeats 3`, and the three-run aggregation
   the report code is built for has not been exercised end to end on the current code.
3. The guard's first-turn AI check (`guard.py:450`) is a substring test
   (`"ai" not in text.casefold()`), while the harness G1 (`evals\checks.py:111`) and the
   loop's disclosure (`loop.py:101`) are word-boundary tests. The substring test is
   stricter, so no reply can pass the guard while failing G1, but the two are not the
   same predicate and could drift.
