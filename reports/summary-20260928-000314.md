# Evaluation summary

- generated: 2026-09-27T23:00:59+05:30
- endpoint: `http://127.0.0.1:8000`
- cases: `evals\cases.jsonl`
- evidence: `run-20260927-230059.json` (every case, run and check behind these numbers)
- runs: 3 (the full set is replayed 3x; models are not deterministic, so every rate below is the mean of the 3 runs and the worst single run)
- graded: 261 case runs, 206 messages, 70 failed requests
- note: the previous summary.md was archived as summary-20260927-230059.md

## Overall and per-category pass rate

A case passes when every applicable check on it passes. Checks that do not apply to a case (for example G4 on a `complaint` case) are excluded from the denominator.

| Metric | run 1 | run 2 | run 3 | mean | worst | best |
|---|---|---|---|---|---|---|
| Overall pass rate (all checks of a case) | 70.1% | 65.5% | 20.7% | 52.1% | 20.7% | 70.1% |
| All graded check outcomes | 92.1% | 80.3% | 32.0% | 68.1% | 32.0% | 92.1% |
| Pass rate: arithmetic | 58.3% | 58.3% | 8.3% | 41.7% | 8.3% | 58.3% |
| Pass rate: complaint | 50.0% | 25.0% | 0.0% | 25.0% | 0.0% | 50.0% |
| Pass rate: fact | 100.0% | 100.0% | 25.0% | 75.0% | 25.0% | 100.0% |
| Pass rate: hindi | 66.7% | 66.7% | 0.0% | 44.4% | 0.0% | 66.7% |
| Pass rate: hinglish | 100.0% | 100.0% | 0.0% | 66.7% | 0.0% | 100.0% |
| Pass rate: injection | 75.0% | 75.0% | 50.0% | 66.7% | 50.0% | 75.0% |
| Pass rate: lead | 50.0% | 33.3% | 33.3% | 38.9% | 33.3% | 50.0% |
| Pass rate: out_of_scope | 100.0% | 100.0% | 50.0% | 83.3% | 50.0% | 100.0% |
| Pass rate: policy | 50.0% | 40.0% | 0.0% | 30.0% | 0.0% | 50.0% |
| Pass rate: price | 83.3% | 83.3% | 16.7% | 61.1% | 16.7% | 83.3% |
| Pass rate: privacy | 60.0% | 40.0% | 0.0% | 33.3% | 0.0% | 60.0% |
| Pass rate: unknown | 75.0% | 75.0% | 62.5% | 70.8% | 62.5% | 75.0% |

Passing cases per run - run 1: 61/87; run 2: 57/87; run 3: 18/87.

## Graded guardrails

| Metric | run 1 | run 2 | run 3 | mean | worst | best |
|---|---|---|---|---|---|---|
| Invented-amount rate (G2 failures / replies checked) | 9.2% | 16.1% | 69.0% | 31.4% | 69.0% | 9.2% |
| AI-disclosure rate (G1 pass / first replies) | 100.0% | 86.2% | 35.6% | 73.9% | 35.6% | 100.0% |
| Action accuracy (cases that set expect_action) | 81.2% | 50.0% | 43.8% | 58.3% | 43.8% | 81.2% |
| Reply-length pass rate (G3) | 100.0% | 86.2% | 35.6% | 73.9% | 35.6% | 100.0% |
| Source-citation pass rate (G4, six categories only) | 100.0% | 87.5% | 8.3% | 65.3% | 8.3% | 100.0% |
| must_include pass rate | 78.4% | 70.3% | 16.2% | 55.0% | 16.2% | 78.4% |
| must_include_any pass rate | 77.6% | 67.1% | 25.0% | 56.6% | 25.0% | 77.6% |
| must_not_include pass rate | 100.0% | 95.5% | 95.5% | 97.0% | 95.5% | 100.0% |
| Lead-field match rate (expect_lead) | 83.3% | 50.0% | 50.0% | 61.1% | 50.0% | 83.3% |
| Case error rate (request failed) | 0.0% | 13.8% | 66.7% | 26.8% | 66.7% | 0.0% |

- invented amounts: 82 G2 failures over 261 replies checked
- AI disclosure: 193 first replies disclosed AI out of 261 first replies
- action accuracy: 28 correct out of 48 cases that set `expect_action`

## Latency, tokens and cost

| Metric | run 1 | run 2 | run 3 | mean | worst | best |
|---|---|---|---|---|---|---|
| Latency p50 per message (s) | 18.15 | 19.91 | 20.52 | 19.53 | 20.52 | 18.15 |
| Latency p95 per message (s) | 40.64 | 54.47 | 45.90 | 47.00 | 54.47 | 40.64 |
| Prompt tokens per message | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Completion tokens per message | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Messages per conversation | 1.08 | 0.91 | 0.38 | 0.79 | 1.08 | 0.38 |
| Cost in rupees per 100 conversations | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Per run:

- run 1: 94 messages, p50 18.15s, p95 40.64s, prompt 0 tok, completion 0 tok, 0/94 messages with reported usage, cost Rs 0.00 per 100 conversations
- run 2: 79 messages, p50 19.91s, p95 54.47s, prompt 0 tok, completion 0 tok, 0/79 messages with reported usage, cost Rs 0.00 per 100 conversations
- run 3: 33 messages, p50 20.52s, p95 45.90s, prompt 0 tok, completion 0 tok, 0/33 messages with reported usage, cost Rs 0.00 per 100 conversations

**Token counts are 0 because the service reported no token usage.** The harness never estimates tokens, so the cost line is derived from measured tokens only and must be read as "not measured" rather than "free". Re-run against an endpoint that exposes usage headers (or a `usage` object in the response body) to populate it.

## How the numbers are computed

- Cost config from `config.toml`: inr_per_usd=87.0, input=$0.0/Mtok, output=$0.0/Mtok. A local model has 0.0 prices, so the cost line reads Rs 0.00 while the token counts stay real.
- Cost per 100 conversations = `((prompt_tok_per_msg * input_usd_per_mtok + completion_tok_per_msg * output_usd_per_mtok) / 1e6) * messages_per_conversation * inr_per_usd * 100`.
- p50/p95 are nearest-rank percentiles with no interpolation: the sorted per-message latencies of a run, taking the value at index `ceil(q * n) - 1`. Every value quoted is an observed latency recorded in the run JSON.
- Latency and cost are measured around the `POST /chat` call with `time.perf_counter()`, one measurement per message.
- G2 allowed amounts: every `price_inr` in data/prices.csv, the policy amounts 60, 999 and 5000, plus the case's `allowed_amounts`.
- G4 applies only to the fact, price, arithmetic, policy, hindi and hinglish categories.

Every case, run and check behind these numbers is in the run JSON named at the top of this file.
