# Open-JEV local benchmark — 2026-09-26

Local replay of Vita's 20-prompt battery plus 18 variants (38 cases) against Open-JEV **v0.3.6** (`bdf810e`, VitaDAO/open-jev-tinfoil main), with the enclave's runtime settings. The results are compared with the hosted Tinfoil enclave run from vita-agent (`probe_battery.out`).

## Headline

- **Correctness parity: 38/38.** Local status, full `reason_codes` and planned queries match the hosted enclave exactly. Each case was also stable across 6 runs.
- **Local selector latency (loopback, M5 Max, 4 threads):** p50 **230 ms**, p95 **429 ms**, max 625 ms (n = 190 timed calls).
  - Planned: p50 237 / p95 429 ms.
  - Handoff: p50 226 / p95 389 ms.
- **Hosted, same cases:** p50 **1518 ms**, p95 **2455 ms**, single runs.
- **Hosted ≈ 505 ms + 4.46 × local** (linear fit over the 38 cases, r = 0.95):
  - **~0.5 s fixed per call.** This is client bootstrap/attestation plus network. It is clearest on B16, a regex-only rejection that takes 1.4 ms locally and 590 ms hosted, and on V18, a research fast-path that takes 15 ms locally and 647 ms hosted.
  - **The rest is model compute, about 4.5× slower in the enclave** (4 vCPUs) than on 4 M5 Max threads. For a typical full plan (~400 ms locally) that is ~1.8 s of the ~2.5 s hosted.
  - So on long turns the dominant cost is CPU in the enclave, not the network. On short turns (profile fast-paths, regex rejections) it is the fixed attestation/network overhead.
- **Cold start:**
  - Process start to `/health` ready: 9.2 s. Startup already runs one warm-up inference.
  - First `/v1/select` after that: 254 ms, versus a 212 ms median for the same question. So there is no meaningful cold penalty once the process is ready.
- **Memory:** RSS 2.16 GB idle, 2.20 GB peak, well inside the enclave's 8 GB.

## Setup

| Item | Value |
|---|---|
| Source | VitaDAO/open-jev-tinfoil @ `bdf810e0` (tag v0.3.6), equal to origin/main |
| Model | `com-kotobalabs/open-jev-deberta-v3-large` @ `19bf9a64…`, packed with `scripts/pack_weights.py` (868 MB `model.safetensors`) |
| Env | `ENABLE_EXPERIMENTAL_SELECTOR=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH=vendor MODEL_DIR=model HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, locally generated `OPEN_JEV_API_KEY` (not recorded) |
| Server | `uvicorn server:create_app --factory --host 127.0.0.1 --port 18080 --workers 1 --no-access-log` |
| Host | Apple M5 Max, 18 cores (capped to 4 threads), 128 GB RAM, macOS 26.4, Python 3.12, CPU torch |
| Enclave | 4 vCPU / 8 GB (tinfoil-config.yml) |
| Request | Last `vita-selector/v2` request in `jev_capture.jsonl`, with 123 metrics, `Europe/Bucharest` and literature on. `state.current_request` was swapped per case and `recent_user_requests=[]`. |
| Method | For each case: 1 untimed warm-up call, then 5 timed calls. Client wall-clock uses `httpx`, keep-alive, over loopback. |

### Identity check against Vita PINS (`GET /health`)

| Field | Local | Pin | |
|---|---|---|---|
| `selector_sha256` | `48e38e4fd9e0a0f4dd729f7432e6a0f44ff12ef84245ee88aa5c468f9ca345d4` | same | ✅ |
| `selector_adapter_sha256` | `79057d0e2673aa2813a8ea64193d74182a33392b1221c7c06d4afc7afe18dc36` | same | ✅ |
| `model_revision` | `19bf9a64815add579fbf6c907bef584d9277a8e4` | same | ✅ |

**Server-reported timing:** `/v1/select` returns no timing field; only `/decide` returns `inference_ms`. The response carries `schema_version`, `advisory`, `time_zone`, `selector_sha256`, `adapter_sha256`, `model_revision`, `request_sha256` and `diagnostics.clause_decisions`. Over loopback, client wall-clock ≈ server time.

## Per-question results

Latency columns are in ms:
- **first** is the warm-up call.
- **med** and **p95** are over the 5 timed calls.
- **hosted** is the single hosted run from vita-agent.
- **Δ** is hosted minus local median.
- **match** means status, reason codes and queries are all identical to the hosted run.

| # | Question | Status | Reason codes | Planned queries | first | med | p95 | hosted | Δ | match |
|---|---|---|---|---|---|---|---|---|---|---|
| B01 | How many workouts did I do this month? | handoff | unsupported_operation | `[]` | 254 | 211 | 219 | 1564 | 1353 | ✅ |
| B02 | How many workouts did I do between May 17 and August 17 2026? | handoff | unsupported_operation | `[]` | 231 | 232 | 245 | 1808 | 1576 | ✅ |
| B03 | How many workouts have I ever recorded? | handoff | unsupported_operation | `[]` | 202 | 212 | 214 | 1634 | 1422 | ✅ |
| B04 | What was my sleep score last night? | handoff | query_contract_unrepresentable | `[]` | 393 | 360 | 381 | 2509 | 2149 | ✅ |
| B05 | How has my sleep been over the last 6 months? | planned | — | `[{"kind": "health", "metrics": ["resilience_sleep_recovery", "sleep_consistency", "sleep_deep", "sleep_effi…` | 400 | 416 | 428 | 2754 | 2338 | ✅ |
| B06 | What was my HRV over the last 7 days? | planned | — | `[{"kind": "health", "metrics": ["heart_rate_variability"], "operation": "trend", "period": {"kind": "relati…` | 379 | 398 | 425 | 2268 | 1870 | ✅ |
| B07 | Compare my Whoop and Oura recovery this week. | handoff | source_filter | `[]` | 218 | 225 | 227 | 1468 | 1243 | ✅ |
| B08 | When was my last lab test? | handoff | unresolved_temporal_phrase | `[]` | 202 | 210 | 279 | 1398 | 1188 | ✅ |
| B09 | Show my ApoB trend. | planned | — | `[{"kind": "health", "metrics": ["apob"], "operation": "trend", "period": {"kind": "all_history"}}]` | 272 | 214 | 273 | 1422 | 1208 | ✅ |
| B10 | What are my fasting glucose, HbA1c, and hsCRP? | handoff | requested_metric_unavailable | `[]` | 273 | 301 | 324 | 1534 | 1233 | ✅ |
| B11 | What are my allergies and current medications? | handoff | unbound_profile_qualifier | `[]` | 114 | 99 | 100 | 877 | 778 | ✅ |
| B12 | What are my health goals? | planned | — | `[{"kind": "health", "records": ["profile"], "operation": "latest", "period": {"kind": "all_history"}, "prof…` | 99 | 87 | 98 | 849 | 762 | ✅ |
| B13 | What are my chronic conditions? | planned | — | `[{"kind": "health", "records": ["profile"], "operation": "latest", "period": {"kind": "all_history"}, "prof…` | 89 | 88 | 89 | 819 | 731 | ✅ |
| B14 | Did last night's sleep line up with a workout that day? | handoff | query_contract_unrepresentable | `[]` | 382 | 520 | 610 | 2415 | 1895 | ✅ |
| B15 | Given my ApoB and my goals, what should I focus on next? | handoff | profile_history_unavailable | `[]` | 129 | 109 | 119 | 873 | 764 | ✅ |
| B16 | What does the literature say about an ApoB of 78 mg/dL? | handoff | clock_or_timezone_qualifier_unavailable | `[]` | 5 | 1 | 2 | 590 | 589 | ✅ |
| B17 | Is a last-night Oura sleep score of 76 good compared to published data? | handoff | unresolved_temporal_phrase | `[]` | 217 | 225 | 233 | 1819 | 1594 | ✅ |
| B18 | Any papers relevant to my medications and longevity goals? | handoff | profile_read_intent_unconfirmed | `[]` | 102 | 101 | 110 | 843 | 742 | ✅ |
| B19 | Analyze me. | planned | — | `[{"kind": "health", "metrics": ["1_25_dihydroxy_vitamin_d", "activity_hr_average", "activity_hr_max", "acti…` | 210 | 203 | 208 | 1495 | 1292 | ✅ |
| B20 | What should I prioritize, using both my data and the literature? | handoff | unresolved_metric_or_record | `[]` | 249 | 242 | 256 | 1775 | 1533 | ✅ |
| V01 | What is my fasting glucose? | planned | — | `[{"kind": "health", "metrics": ["fasting_glucose"], "operation": "trend", "period": {"kind": "all_history"}}]` | 255 | 233 | 253 | 1496 | 1263 | ✅ |
| V02 | What is my HbA1c? | planned | — | `[{"kind": "health", "metrics": ["hba1c"], "operation": "trend", "period": {"kind": "all_history"}}]` | 235 | 228 | 244 | 1470 | 1242 | ✅ |
| V03 | What is my hsCRP? | handoff | requested_metric_unavailable | `[]` | 230 | 226 | 240 | 1632 | 1406 | ✅ |
| V04 | What are my fasting glucose and HbA1c? | planned | — | `[{"kind": "health", "metrics": ["fasting_glucose", "hba1c"], "operation": "trend", "period": {"kind": "all_…` | 269 | 263 | 273 | 1647 | 1384 | ✅ |
| V05 | Show my latest HbA1c. | planned | — | `[{"kind": "health", "metrics": ["hba1c"], "operation": "latest", "period": {"kind": "all_history"}}]` | 248 | 248 | 249 | 1502 | 1254 | ✅ |
| V06 | Show my HbA1c trend. | planned | — | `[{"kind": "health", "metrics": ["hba1c"], "operation": "trend", "period": {"kind": "all_history"}}]` | 251 | 263 | 271 | 1509 | 1246 | ✅ |
| V07 | How many workouts did I do in September 2026? | handoff | unsupported_operation | `[]` | 246 | 275 | 300 | 1541 | 1266 | ✅ |
| V08 | Show my workouts this month. | planned | — | `[{"kind": "health", "records": ["workouts"], "operation": "latest", "period": {"kind": "calendar", "period"…` | 296 | 272 | 296 | 1540 | 1268 | ✅ |
| V09 | Show my latest sleep score. | planned | — | `[{"kind": "health", "metrics": ["sleep_score"], "operation": "latest", "period": {"kind": "all_history"}}]` | 238 | 230 | 236 | 1576 | 1346 | ✅ |
| V10 | What was my sleep score on September 21, 2026? | planned | — | `[{"kind": "health", "metrics": ["sleep_score"], "operation": "trend", "period": {"kind": "between", "start_…` | 478 | 446 | 457 | 2446 | 2000 | ✅ |
| V11 | Show my Oura readiness this week. | handoff | source_filter | `[]` | 230 | 242 | 250 | 1527 | 1285 | ✅ |
| V12 | Show my Whoop recovery this week. | handoff | source_filter | `[]` | 230 | 241 | 265 | 1686 | 1445 | ✅ |
| V13 | Show my latest lab results. | handoff | unresolved_metric_or_record | `[]` | 223 | 231 | 234 | 1436 | 1205 | ✅ |
| V14 | What are my allergies? | planned | — | `[{"kind": "health", "records": ["profile"], "operation": "latest", "period": {"kind": "all_history"}, "prof…` | 91 | 89 | 95 | 855 | 766 | ✅ |
| V15 | What are my current medications? | handoff | unbound_profile_qualifier | `[]` | 89 | 90 | 97 | 831 | 741 | ✅ |
| V16 | What are my health goals? | planned | — | `[{"kind": "health", "records": ["profile"], "operation": "latest", "period": {"kind": "all_history"}, "prof…` | 88 | 88 | 89 | 832 | 744 | ✅ |
| V17 | What does the literature say about ApoB? | planned | — | `[{"kind": "health", "metrics": ["apob"], "operation": "trend", "period": {"kind": "all_history"}}]` | 367 | 385 | 404 | 2291 | 1906 | ✅ |
| V18 | What does research say about aspirin and longevity? | planned | — | `[{"kind": "research", "topic": "aspirin and longevity"}]` | 17 | 15 | 16 | 647 | 632 | ✅ |
## Summary stats (timed calls)

| Slice | Cases | Local p50 | Local p95 | Hosted p50 | Hosted p95 |
|---|---|---|---|---|---|
| All | 38 | 230 | 429 | 1518 | 2455 |
| Planned | 18 | 237 | 429 | 1499 | 2492 |
| Handoff | 20 | 226 | 389 | 1538 | 2420 |
| Battery (B) | 20 | 216 | 432 | — | — |
| Variants (V) | 18 | 239 | 420 | — | — |

Handoffs cost about as much as plans. Most handoffs are decided *after* the model runs (clause classification, then binding fails). The exceptions are:
- B16 (regex pre-check, ~1 ms).
- Profile fast-paths such as B11/B12/B13/B15/B18/V14–V16 (~90–110 ms).
- The research fast-path V18 (~15 ms).

## Hosted vs local

- **Upstream reference points:** a 1 s optional-selector budget, CI loopback median 723 ms / p95 1.9 s, and model-only ~120 ms locally. The local numbers here (230 / 429 ms) are faster than CI loopback because the M5 Max is fast per core. The ~90 ms profile path is close to the "model-only ~120 ms" figure.
- **Fixed per-call overhead ≈ 0.5–0.6 s.** This is the intercept of the fit and the hosted time of near-zero-compute cases (B16 590 ms, V18 647 ms, profile cases ~820–880 ms against ~90 ms locally). It matches vita-agent's ~0.6 s client bootstrap/attestation measurement. `probe_battery.py` builds a new `VitaClient` per question, so every hosted number includes one bootstrap. **Reusing a client/attested session across turns should save most of this.**
- **Compute scaling ≈ 4.5×** between the enclave's 4 vCPUs and 4 M5 Max threads. Full plans (B04/B05/B06/B14/V10/V17) take ~360–520 ms locally and 2.3–2.8 s hosted. **Within the enclave, the 1 s budget is exceeded by compute alone** for any multi-clause or long-context turn. Moving attestation off the hot path helps short turns, but long turns need more/faster vCPUs, fewer model passes, or earlier regex-level handoffs.
- **Caveat:** hosted figures are single runs (n = 1 per case), so treat per-case Δ as ±~200 ms noise. The fit (r = 0.95) is still a strong signal.

## Why the handoffs happen (findings only; no code changed)

The citations were verified against v0.3.6. Behaviour was confirmed by calling the pure-Python parsing helpers (`enhance`, `entities`, `temporal`, the clock regex) without the model.

**Bugs:**

1. **hsCRP → `requested_metric_unavailable`** (B10, V03), even though `hscrp` is in `available_metrics`.
   - `query_selector.py:21` `SPELLINGS={…,'hscrp':'hs crp'}` is applied in `enhance()` at `query_selector.py:130`, so the inventory ID can no longer match literally.
   - `metadata/selector-index.v1.json` `inventory_to_catalog` has no `hscrp → hs_crp` entry. The catalog fallback (`proposal_binding.py:77-79`) therefore labels it unavailable, `proposal_selector.py:115-116` raises `requested_metric_unavailable`, and `proposal_selector.py:161` adds `unresolved_metric_or_record`.
   - Fix candidate: add the mapping, or drop the spelling rewrite.
2. **"current medications" → `unbound_profile_qualifier`** (B11, V15).
   - `query_selector.py:133` rewrites `current` → `latest`.
   - The profile filler strip list at `query_selector.py:319` removes `current` but not `latest`, so `query_selector.py:320` sees "latest" left over and raises.
3. **"ApoB of 78 mg/dL" → `clock_or_timezone_qualifier_unavailable`** (B16).
   - The IANA-timezone alternative `\b[a-z_]+/[a-z_]+\b` in the clock regex at `query_selector.py:255` matches `mg/dl`.
   - After that is fixed, "78" would still raise `unbound_quantity` (`proposal_binding.py:329`), because the query is on the health path (see 4).
4. **"literature" / "papers" / "published" are not research triggers** (V17 misroute, plus B15, B16, B17, B18).
   - The clause router at `query_selector.py:216` is `\b(?:research|studies|trials|evidence)\b`. `research_requested` at `proposal_binding.py:124` uses the same list.
   - So "What does the literature say about ApoB?" goes to `health()`. A bound metric with no research word forces `task='health'` (`proposal_selector.py:137-138`), and the default operation is trend.
   - `RESEARCH_QUESTION` (around `query_selector.py:90`) *does* accept "literature", so the two lists are inconsistent.

**Alias / temporal gaps:**

5. **"When was my last lab test?" → `unresolved_temporal_phrase`** (B08).
   - `temporal_spans.py:82-83` rejects any bare `last`/`next`/`this`/… that isn't a supported period.
   - The only rewrite, `query_selector.py:134`, covers "last time … tested", not "last lab test".
   - "lab test" and "lab results" are also not aliases of the labs record, so V13 gets `unresolved_metric_or_record`.
6. **"what should I focus on next" → `profile_history_unavailable`** (B15). The same `temporal_spans.py:82` rule reads "next" as a time phrase, which then trips `query_selector.py:303-304`. Without it, the question would still hand off as `profile_read_intent_unconfirmed` (`query_selector.py:316`); B18 hands off for that reason.
7. **"last-night" (hyphenated)** (B17). `query_selector.py:333` checks `\blast night\b` only, so the bare "last" falls to `temporal_spans.py:82`. "76" is also `unbound_quantity`.
8. **Bare "readiness" / "recovery"** (V11, V12, B07). These aren't aliases of `readiness_score` / `recovery_score`. The source word ("Oura", "Whoop") only binds when it sits right before an entity (`query_selector.py:269`), so it becomes `source_filter`. "Oura readiness score this week" binds.

**Contract mismatch (arguably a bug):**

9. **"sleep score last night" → `query_contract_unrepresentable`** (B04, B14).
   - `query_selector.py:333-334` accepts any metric whose name contains "sleep" for the night basis and builds `HealthRead(date_basis='sleep_end_day')`.
   - `query_plan.py:90` only allows `total_sleep`, `sleep_efficiency` and `sleep_deep`, and raises "Night basis requires sleep".
   - That message fails the reason-code regex at `query_selector.py:237` and becomes `query_contract_unrepresentable`.
   - The two checks disagree about which metrics are sleep metrics.

**By design:**
- Counts ("how many workouts", B01–B03, V07): `unsupported_operation` via `proposal_binding.py:310`.
- Source comparisons (B07): `source_filter` (`proposal_binding.py:307`) and `unsupported_operation`.
- Open-ended advisory (B20): `unresolved_metric_or_record`.

## Notes

- On load, transformers warns about an "incorrect regex pattern" in the tokenizer, advising `fix_mistral_regex=True`. The hosted and local outputs are identical, so it has no effect on behaviour here. It is worth confirming upstream that the warning is spurious for DeBERTa-v3.
- Artifacts: `model/` (downloaded and packed, untracked) and `.venv/`. The benchmark harness and raw JSON are in the session scratchpad (`bench.py`, `bench.json`). Nothing was committed.
