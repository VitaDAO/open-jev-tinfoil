# P0: safety and rule fixes (branch `p0-safety-and-fixes`, local only, no pushes)

Source: `retrain-plan-2026-09-26.md` §3 P0 and vita-agent spec v1.3 §3. The model is unchanged (DeBERTa v0.3.6 weights) and the contract stays v2.

## P0a: acute/crisis (ships first; G12 must pass)
- [x] Acute/crisis rules evaluated before everything else, on the current request and recent requests: symptoms, self-harm, dangerous readings, medication/insulin changes → handoff `acute_or_crisis_requires_model`
- [~] ≥40-case acute test set with 0 misses; false-positive check on ordinary reads
  - The in-repo tests (52 acute) pass. On independent sets the keyword rule alone catches 28/28 on held-out v2 (seen, after the category fixes) but only 36/90 (40%) on the fresh acute_v3 (unseen). The full selector plans **0/90** acute_v3 cases, because the other 54 hand off for other reasons.
  - Rules alone can't meet the 0-miss bar. A learned acute signal is needed (P2), and the Vita-side guard stays mandatory.
- [x] Publish reason codes: `reason_codes.json` plus a test that keeps it in sync with the source, and `/health` returns the list and its sha256

## P0b: rule fixes (only together with P0a; G12 re-run)
- [x] Regex cache: precompile the label patterns (identical plans, ~−80 ms)
- [x] hsCRP: map `hscrp` → catalog `hs_crp` in the index
- [x] "current medications": the profile filler strip also removes `latest`
- [x] Units vs the clock/timezone regex (mg/dL, mmol/L, …)
- [x] Research triggers: literature, papers, published, science (both routers plus the research-question grammar)
- [x] Labs: "last/latest/most recent lab test/results/blood test" → labs record
- [x] "last-night" / "last night's" → "last night"
- [x] Sleep night basis only for total_sleep, sleep_efficiency, sleep_deep; "how long did I sleep" → total_sleep
- [x] Operation: "what is/what's/was my X", bare "X", "X today" → latest (post-acceptance)
- [x] Periods: rolling vs calendar conventions for year/month/week
- [x] Follow-up pronouns ("show it for …") inherit the prior subject; never broaden
- [x] Aliases: readiness → readiness_score, recovery → recovery_score; calories rules
- [x] Future-dated metric reads → handoff (future calendar records stay valid)
- [x] Plan-breadth cap: >8 metrics only for an explicit overview or a single named area

## Verification
- [x] Unit tests green: 542 passed, 4 skipped (baseline 439). One expectation was changed on purpose: "What's my cholestrol?" → latest (spec §3.9)
- [x] 656-case replay: diffs vs v0.3.6 reviewed; no new false accepts
  - 606 of 656 identical. All 28 changed plans were reviewed and are intended: hsCRP, profile, labs, research routing, latest for single values, readiness/recovery with source, rolling year, S41 inheriting weight. A01/A02 (acute) now hand off.
  - Only 4 graded fixtures flip against old gold: 3 public research topics (the gold predates topic planning) and "Show my current medications." with profile available. A test confirms it still hands off when profile is unavailable.
- [x] Held-out v2 (520): bound re-computed
  - False accepts 77/172 (v0.3.6) → **23/176 (P0)**. Correct plans 87 → 150. 95% CP bound 51% → 18.0%; direct subset 15/145 → 15.5%.
  - 12 of the 23 depend on labels or rulings (source filters are supported, future calendar, last night latest/trend, show/pull up).
  - **v2 was used to guide the P0 fixes, so it is now a development set.** A fresh unseen set is needed for a real bound.
- [~] Acute set: **0/90 acute planned** on the fresh independent acute_v3 (full pipeline, all P0 fixes), 3 benign planned, 7 benign blocked.
  - The keyword rule alone catches 36/90. The rest hand off for other reasons, so a learned acute signal (P2) and the Vita guard remain required.
- [x] Local latency (DeBERTa, P0 branch, regex cache): replay p50 101 ms / p95 255 ms; held-out p50 102 / p95 244 ms (v0.3.6: 110 / 331 ms). The fast students are not wired into this branch yet.

## Notes for vita-agent / spec
- "show my X" (no window) keeps today's behaviour (trend, and the 30-day default for steps/sleep). §3.9 names only "what is/what's my X", bare "X" and "X today". Needs a ruling.
- "CRP" alone is **not** bound to hscrp: standard CRP and hs-CRP are different tests. Needs a ruling.
- "platelets" keeps the exact inventory ID; the platelets/platelet_count duplicates need an inventory-side decision.
- Single past days ("steps yesterday", "last night") stay **trend**, as the reviewed fixtures expect: latest on a daily-summed metric returns the last reading, not the day total. Only "X today" → latest (spec §3.9). "Last night" needs a ruling.
- New published codes: acute_or_crisis_requires_model, future_period_unavailable, night_basis_metric_unavailable, plan_breadth_exceeded (+ multiple_or_open_periods, which was already emitted but unlisted).

## Spec v1.4 rulings (§3.10d), implemented
- [x] "show/pull up my X" keeps trend with default windows; single past day / "last night" keeps the one-day trend (no change needed)
- [x] Bare "CRP" → ambiguous_metric_alias (never bound to hscrp)
- [x] Overview on a large inventory (>16) → curated set (sleep core, HRV/RHR, steps, VO2 max, weight, key labs), never the full inventory
- [x] Sleep area by wording: quantity → total_sleep; qualitative → core 5; "full breakdown/everything" → full area
- [x] Acute look-alikes tuned: rowing "stroke rate/volume", "heart attack risk", history/family-history and education framing unless a present-tense cue appears. Benign blocks on acute_v3 went 7 → 0, recall unchanged (36/90 rule-only; 28/28 held-out v2)
- Verification round 4: 566 tests pass; acute_v3 full pipeline **0/90 acute planned**, 0/90 benign blocked; replay regressions vs v0.3.6 are still only the 4 reviewed-intended; held-out v2 (dev) FA 22/176 (bound 17.4%), direct 14/145; p50 101-103 / p95 244-252 ms

## Next (agreed order with vita-agent)
- [x] Held-out v3 (620 cases, independent, spec v1.4), graded before any tuning on it:
  - v0.3.6: 127 wrong / 238 planned (bound 58.8%)
  - **P0: 35 wrong / 259 planned (bound 17.5%)**; correct plans 109 → 216; direct subset 24/219 (15.1%)
  - Of P0's 35: ~20 are latest-vs-trend shape on the right data, and 7 are a record-operation label artefact (records are always "latest" listings). **8 are wrong or dropped data** (bound 5.6%):
    - good/bad cholesterol aliases;
    - "recovery + strain" drops strain;
    - "week of sept 7" becomes one day;
    - "weight trend this year and body fat right now" merged;
    - "workouts and calories from workouts" drops workouts;
    - "now show me body fat" doesn't inherit the window;
    - "september so far" = full month.
  - Any fix informed by v3 makes it a dev set; the next honest bound needs v4.
- [ ] Learned acute signal (P2); the rule catches 36/90 on acute_v3
- [ ] Wire the fast ModernBERT students into this branch (encoder interface, adapters, pins)
- [ ] Release evidence bundle lists every file Vita must sync at the pin bump: vendored query_plan.py (NIGHT_METRICS), schema_index.py, metadata/selector-index.v1.json, metadata/reason-codes.v1.json + sha256, new selector_sha256 / adapter_sha256 / model revision

## Held-out progression (each set written independently to spec v1.4; graded before tuning on it)
| Set | v0.3.6 wrong/planned (95% bound) | P0 at that time | Notes |
|---|---|---|---|
| v3 (620) | 127/238 (58.8%) | 35/259 (17.5%) | then used as a dev set |
| v4 (700) | 109/287 (42.9%) | 21/303 (9.8%) | then used as a dev set |
| **v5 (700)** | **127/263 (53.5%)** | **12/280 (6.9%)**; direct 9/243 (6.4%); correct plans 134 → 263 | **clean held-out, not tuned on** |

v5 remaining classes:
- **Heart-rate variants (6):** max/peak/activity/workout HR bind to generic heart_rate or a workouts listing.
- **"X also" follow-up (1).**
- **Arguably correct or label artefacts (5):** a split mixed-window clause, the platelets/WBC duplicate ids, an extra labs read.

The next fixes (HR variants, a trailing "also") would make v5 a dev set, so v6 would be needed.

Current branch state: 587 tests pass. The replay vs v0.3.6 shows only the 4 reviewed-intended changes, with no correct plan lost. acute_v3: 0/90 acute planned, 0/90 benign blocked. p50 ~101-107 / p95 ~250-262 ms (DeBERTa + regex cache).

## Spec v1.5 round (heart-rate variants, trailing "also")
- [x] HR variants → day_max_hr / day_avg_hr / workout_max_hr / workout_avg_hr / activity_hr_* / hr_zone_N. An HR question with no bound metric hands off (never a workouts listing).
- [x] "X also" follow-up inherits the prior window.
- Verification: 595 tests pass; acute_v3 0/90 planned, 0/90 benign blocked; replay still only the 4 reviewed changes, with no correct plan lost; v5 (now a dev set) 6/292.
- Latency in this run was p50 127 / p95 323 ms, with machine load average ~4.7 from other sessions. The new rewrites cost 0.05 ms/call, so re-measure on a quiet machine before quoting.
- **Held-out status:** v5's clean number (12/280, ≤6.9%) is the reported figure. The next honest bound needs v6. vita-agent proposes making it direct-answer-class-balanced, if the user adopts the risk-tiered policy (spec v1.5).

## Speed work (fast encoders on the current branch)
- Stage profile (fast 2-student version): rules 4–7 ms; encoder ~30 ms (two ModernBERT-base passes of ~15 ms).
- Thread count barely matters at batch 1 (1 thread 14.3 ms vs 4 threads 15.0 ms): per-operation overhead, not FLOPs. So 1 thread per request, and capacity can scale by running parallel worker processes (4 vCPU ≈ 4 concurrent requests).
- In-process parallel passes: only 28 → 25 ms (GIL). TorchScript trace+freeze: bit-identical, −11% per pass.
- Layer dropping: 22 → 16 / 11 / 8 layers = 14.0 → 10.3 / 7.6 / 5.6 ms per pass.
- Re-harvested teacher labels on branch code (14,569 strings). Trained one-pass students:

| Variant | v5 (dev) correct / FA | Replay | acute_v3 | p50 / p95 (loaded machine) |
|---|---|---|---|---|
| DeBERTa branch | 281 / 6 of 292 | clean (4 reviewed) | 0/90 | ~105 / 257 ms |
| 2× ModernBERT-base (22L) | 253 / 9 of 267 | 7 plans lost | 0/90 | 37 / 45 ms |
| 1× 16-layer student | 229 / 5 of 239 | **1 new wrong (m04)** ✗ | 0/90 (was 1/90 before the guard) | 26 / 29 ms |
| **1× 11-layer student** | 195 / 4 of 204 | no new wrong; 11 correct plans lost to handoff | **0/90** | **19 / 23 ms** |

- New guard: an extra sentence with no metric/record/date/read wording hands off (`unbound_request_constraint`).
  - Found because the 16-layer student planned "i dont see the point in anything anymore. whats my sleep score" (indirect suicidal language that the keyword rule misses and DeBERTa had rejected only incidentally).
  - Catches 8/90 acute_v3 and 0/90 benign; no replay regressions after refinement.

## Learned acute classifier (P2 signal, local experiment)
- A fine-tuned copy of the 11-layer ModernBERT plus a logistic head, trained on independent acute_train.jsonl (2,797 examples; 72% of acute ones indirect) plus 3,000 ordinary requests. Threshold chosen on its own validation split for 99% recall.
- **Independent results:**
  - acute_v3: model + rule catch **89/90** (rule alone 36/90); benign flagged 7/90.
  - Should-plan requests flagged: v5 8/580, v6 **0/627**.
  - The only miss is "relapsed on the sh thing…" (self-harm slang); add slang to the training data.
- The frozen linear head was rejected (64% false positives at 99% recall).
- Cost: +1 thin pass (~7.6 ms), or 0 if merged as a head into the main student.

## High-level review (2026-09-26): priorities
1. **Coverage is the #1 product gap.**
   - Even the DeBERTa branch hands off 293/580 plannable v5 requests.
   - Top causes: learned intent gate rejecting bare lookups (90; 177 on thin11) and unrecognised metric names (71–81).
   - Fixes: rule-certify bare metric lookups; build a full alias table for all 123 metrics in the shared index.
2. Safety must be positive certification, not incidental: learned acute (done), whole-message guard (done), real agreement check replacing the dead pass 2 (to do), Vita guard.
3. Freeze regex growth; move toward span tagging plus deterministic resolvers.
4. Train students on spec-correct labels as well as teacher labels, to lift the teacher ceiling.
5. Measurement pipeline: dev vs rotating frozen test sets, one label spec, per-class bounds, x86 parity, proper server integration.
6. Latency is now dominated by Vita-side attestation/network (~0.6 s/turn): overlap the selector with turn setup.

## Audit on current code (all sets; v2–v5 are dev sets now, so bounds are optimistic)
| Set | DeBERTa branch FA/accepted (bound) | thin11 FA/accepted (bound) |
|---|---|---|
| v2 | 15/185 (12.2%) | 11/146 (12.2%) |
| v3 | 9/264 (5.9%) | 9/189 (8.2%) |
| v4 | 5/315 (3.3%) | 5/245 (4.2%) |
| v5 | 6/292 (4.0%) | 4/204 (4.4%) |
| acute_v3 | 0/90 planned | 0/90 planned |
| replay | 4 reviewed changes only | + 11 correct plans lost |

## Learned catalog-driven parser (in progress; replaces the regex layer)
- jevparse.py (model + deterministic calendar resolver + generic date-syntax parser) and jevtrain.py written.
- [x] Data in: gold_train_A/B/C (12.8k rows after normalisation + augmentation), vocabulary_enriched.json (191 catalog items).
- [x] v3 + agreement check (whole-input link set must equal span-decode set, else hand off): v6 dev 439 ok / 27 wrong of 476 (bound 7.7%), acute 0/90.
- [x] Root cause of the last 27 wrong plans on v6 (by layer):
  - Training alignment bug: BPE offsets include the leading space, so `a >= s` dropped the FIRST word of every labelled span (role tags + span-contrastive loss). Caused "RDW"→"DW", "last weekend"→"weekend", "since June 2025"→June 20. Fixed in jevtrain (strip leading space, overlap test).
  - Decode bug: tagged sub-tokens were joined with spaces ("2025"→"20 25"). Fixed: rebuild runs from original characters. Alone: v6 439→452 ok.
  - Data gap (14 operation errors): training labels are consistent ("results/readings/progress/been" → trend, 100%), but only in verb-led forms; bare "RDW results" → latest. v3 predates the verbless augmentation (gold_train_D_aug); v4 includes it.
  - Catalog text (6): strain_score's text never said "strain score"; item text now includes the id name.
  - Data gap: no "book/schedule/order" action handoffs → added 36.
  - Held-out label error: "lean mass since July" labelled lean_mass_percent; catalog and every training row map bare "lean mass" → fat_free_mass. Still counted as wrong.
  - Harness bug: the research-topic digit check was lowercase-only, so "HbA1c" counted as a leaked value; now case-insensitive in jeveval.
  - Open: "calories from my workouts" adds the workouts record; "now average" follow-up keeps max HR; "last blood panel" → platelets.
- [x] Honest v7 (untouched, errors not inspected) for v3 + fixed decode: 391 ok / 23 wrong of 424 accepted (bound 7.6%).
- Honest v7 comparison (740 rows, same grader):
  | system | ok plans | wrong / accepted | 95% bound | latency |
  |---|---|---|---|---|
  | v0.3.6 (shipped) | 163 | 100 / 265 | 42.9% | ~100 ms |
  | P0 rules + DeBERTa | 284 | 14 / 302 | 7.2% | ~107 ms |
  | P0 rules + thin11 | 202 | 8 / 214 | 6.6% | ~22 ms |
  | jevparse v3 + agreement | 391 | 23 / 424 | 7.6% | ~15–20 ms |
- [x] Selective prediction: joint confidence = weakest decision (plan prob, link margin, operation/period/source/inherit heads); threshold chosen by conformal risk control (Learn-then-Test, fixed sequence, alpha 2%, delta 5%) on v6, reported on v7 (jevconformal.py). On v3 it cannot reach alpha: remaining errors are confident (systematic), so they must be fixed in data/model.
- [x] Catalog-driven training data (catalog_synth.py): every catalog item x its unambiguous names x latest/trend phrasing x time windows x follow-ups; 3,445 rows, 158/164 metrics (6 have only shared names). New catalog items get training data with no code change. Parked as catalog_E.pending.jsonl until v4 is calibrated (jevtrain globs gold_train_*).
- [x] Two-model agreement (jevagree.py): plan only when two independently trained models emit the identical plan.
- [ ] v4: train with fixed alignment + augmentation + id-name item text + merged crisis head; calibrate on validation; grade v6 then v7.
- Budget: ≤20–30 ms per request (target ~10–15 ms: one thin ModernBERT pass + precomputed catalog matrix).
- JevBench top-3 is a separate problem (general typed-decision model; top entries are 4B+ GPU models). The user has to choose the track.
