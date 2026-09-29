# Evaluation summary

- generated: 2026-09-28T00:03:14+05:30
- endpoint: `http://127.0.0.1:8000`
- cases: `evals\cases.jsonl`
- evidence: `run-20260928-000314.json` (every case, run and check behind these numbers)
- runs: 1 (the full set is replayed 1x; models are not deterministic, so every rate below is the mean of the 1 runs and the worst single run)
- graded: 87 case runs, 94 messages, 0 failed requests
- note: the previous summary.md was archived as summary-20260928-000314.md

## Overall and per-category pass rate

A case passes when every applicable check on it passes. Checks that do not apply to a case (for example G4 on a `complaint` case) are excluded from the denominator.

| Metric | run 1 | mean | worst | best |
|---|---|---|---|---|
| Overall pass rate (all checks of a case) | 67.8% | 67.8% | 67.8% | 67.8% |
| All graded check outcomes | 90.8% | 90.8% | 90.8% | 90.8% |
| Pass rate: arithmetic | 58.3% | 58.3% | 58.3% | 58.3% |
| Pass rate: complaint | 25.0% | 25.0% | 25.0% | 25.0% |
| Pass rate: fact | 100.0% | 100.0% | 100.0% | 100.0% |
| Pass rate: hindi | 66.7% | 66.7% | 66.7% | 66.7% |
| Pass rate: hinglish | 100.0% | 100.0% | 100.0% | 100.0% |
| Pass rate: injection | 66.7% | 66.7% | 66.7% | 66.7% |
| Pass rate: lead | 50.0% | 50.0% | 50.0% | 50.0% |
| Pass rate: out_of_scope | 100.0% | 100.0% | 100.0% | 100.0% |
| Pass rate: policy | 50.0% | 50.0% | 50.0% | 50.0% |
| Pass rate: price | 83.3% | 83.3% | 83.3% | 83.3% |
| Pass rate: privacy | 40.0% | 40.0% | 40.0% | 40.0% |
| Pass rate: unknown | 87.5% | 87.5% | 87.5% | 87.5% |

Passing cases per run - run 1: 59/87.

## Graded guardrails

| Metric | run 1 | mean | worst | best |
|---|---|---|---|---|
| Invented-amount rate (G2 failures / replies checked) | 5.7% | 5.7% | 5.7% | 5.7% |
| AI-disclosure rate (G1 pass / first replies) | 100.0% | 100.0% | 100.0% | 100.0% |
| Action accuracy (cases that set expect_action) | 56.2% | 56.2% | 56.2% | 56.2% |
| Reply-length pass rate (G3) | 100.0% | 100.0% | 100.0% | 100.0% |
| Source-citation pass rate (G4, six categories only) | 100.0% | 100.0% | 100.0% | 100.0% |
| must_include pass rate | 78.4% | 78.4% | 78.4% | 78.4% |
| must_include_any pass rate | 72.4% | 72.4% | 72.4% | 72.4% |
| must_not_include pass rate | 100.0% | 100.0% | 100.0% | 100.0% |
| Lead-field match rate (expect_lead) | 66.7% | 66.7% | 66.7% | 66.7% |
| Case error rate (request failed) | 0.0% | 0.0% | 0.0% | 0.0% |

- invented amounts: 5 G2 failures over 87 replies checked
- AI disclosure: 87 first replies disclosed AI out of 87 first replies
- action accuracy: 9 correct out of 16 cases that set `expect_action`

## Latency, tokens and cost

| Metric | run 1 | mean | worst | best |
|---|---|---|---|---|
| Latency p50 per message (s) | 35.18 | 35.18 | 35.18 | 35.18 |
| Latency p95 per message (s) | 78.65 | 78.65 | 78.65 | 78.65 |
| Prompt tokens per message | 0.00 | 0.00 | 0.00 | 0.00 |
| Completion tokens per message | 0.00 | 0.00 | 0.00 | 0.00 |
| Messages per conversation | 1.08 | 1.08 | 1.08 | 1.08 |
| Cost in rupees per 100 conversations | 0.00 | 0.00 | 0.00 | 0.00 |

Per run:

- run 1: 94 messages, p50 35.18s, p95 78.65s, prompt 0 tok, completion 0 tok, 0/94 messages with reported usage, cost Rs 0.00 per 100 conversations

**Token counts are 0 because the service reported no token usage.** The harness never estimates tokens, so the cost line is derived from measured tokens only and must be read as "not measured" rather than "free". Re-run against an endpoint that exposes usage headers (or a `usage` object in the response body) to populate it.

## How the numbers are computed

- Cost config from `config.toml`: inr_per_usd=87.0, input=$0.0/Mtok, output=$0.0/Mtok. A local model has 0.0 prices, so the cost line reads Rs 0.00 while the token counts stay real.
- Cost per 100 conversations = `((prompt_tok_per_msg * input_usd_per_mtok + completion_tok_per_msg * output_usd_per_mtok) / 1e6) * messages_per_conversation * inr_per_usd * 100`.
- p50/p95 are nearest-rank percentiles with no interpolation: the sorted per-message latencies of a run, taking the value at index `ceil(q * n) - 1`. Every value quoted is an observed latency recorded in the run JSON.
- Latency and cost are measured around the `POST /chat` call with `time.perf_counter()`, one measurement per message.
- G2 allowed amounts: every `price_inr` in data/prices.csv, the policy amounts 60, 999 and 5000, plus the case's `allowed_amounts`.
- G4 applies only to the fact, price, arithmetic, policy, hindi and hinglish categories.

Every case, run and check behind these numbers is in the run JSON named at the top of this file.
