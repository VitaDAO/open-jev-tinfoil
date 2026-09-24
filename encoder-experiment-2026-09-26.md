# ModernBERT as the selector encoder (2026-09-26)

**Question.** Can ModernBERT replace DeBERTa-v3-large (Kotoba's model) as the frozen encoder under the selector heads, with the heads refit, and stay as good as today?

**Answer: not as a frozen drop-in.** It is 6.5× faster (15 ms vs 100 ms per pass), but it adds 3–5 new wrong plans and loses about 25 correct plans on 656 replay cases. It fails gate G1 (no new false accepts). It needs fine-tuning or distillation (plan P4c), not only a head refit.

## Method

Everything ran locally in scratch scripts, with no repo edits. Setup: M5 Max, 4 threads.

1. **Encoders compared:**
   - DeBERTa-v3-large, the shipped model;
   - `answerdotai/ModernBERT-base` (149M);
   - `answerdotai/ModernBERT-large` (395M).

   Features are the same state text (`Current request: … / Recent user requests:`), averaged over its tokens and L2-normalised.
2. **Head training replicated exactly.**
   - The 11 selector heads (`scripts/train_selector_adapter.py`, 449 synthetic rows) and the v4 read-intent gate (`scripts/reproduce_read_intent_v4.py`, frozen 915/60 protocol) use the same data, the same ridge fitting and the same threshold rule.
   - **Harness check:** re-running this pipeline with DeBERTa reproduces the shipped weights **exactly** (max |ΔW| = 0, same threshold 0.080331) and identical grades on all 656 cases.
3. **Only the first pass is swapped.** Pass 2, the coverage check on Kotoba's decision head, and all the rules are unchanged.
4. **Replay: 656 cases.**

   | Source | Cases |
   |---|---|
   | Repo v5 gold fixtures | 366 |
   | v6 challenges | 64 |
   | v7 edge, fresh and intent packets | 137 |
   | Vita battery | 38 |
   | Vita synthetic | 42 |
   | Acute/injection probes | 9 |

   Every result is diffed against today's model. Any case where a candidate now plans, or plans differently, is reviewed by hand.

## Results

| | DeBERTa (today) | ModernBERT-base | ModernBERT-large |
|---|---|---|---|
| Time to train the heads | 141 s | 22 s | 58 s |
| v4 gate on calibration (true accepts / false accepts) | 30/30 / 0 | 29/30 / 0 | 30/30 / 0 |
| Identical to today | — | 617 / 656 (94%) | 617 / 656 (94%) |
| **New wrong plans** (reviewed) | — | **5** (+2 borderline) | **≥3** (+2 changed broad plans) |
| Plans lost (now a handoff) | — | 25 | 25 |
| Encoder pass p50 | 100 ms | **15 ms** | 41 ms |
| End-to-end p50, repo fixtures (8-metric inventory) | 111 ms | **26 ms** | 53 ms |
| End-to-end p50 / p95, Vita inventory (123 metrics) | 193 / 349 ms | 103 / 250 ms | 131 / 283 ms |

On the Vita inventory, the remaining time is mostly the regex recompilation (fix in P0, ~−80 ms) and the DeBERTa pass 2 on dated requests.

**New wrong plans, ModernBERT-base:**
- "Any papers on omeprazole long-term use?" → **all 123 metrics**.
- "Show my sleep this month but not naps." → sleep metrics, **negation ignored**.
- "open the lab report from March 3rd" → 8 metrics.
- "Show my latest five lab reports with abnormal results only" → labs, **filter ignored**.
- "Who issued my lab reports and what is their certified billing code?" → labs read (the gold answer is must-handoff).

**New wrong plans, ModernBERT-large:**
- "Could you assess my health overall?" (m01, m04) → a different broad metric set.
- "Show my latest five lab reports with abnormal results only" → filter ignored.
- "Show my sleep this month but not naps." → negation ignored.

**Lost plans:** almost all are `learned_read_intent_unconfirmed`, meaning the read-intent gate became more conservative. Examples: "What is my fasting glucose?", "What does the literature say about ApoB?", "What was my VO2 max in August?". The rest are broad requests going to `no_supported_read`.

## Interpretation

- **Speed is solved by ModernBERT-base.** With the regex fix, a one-pass request would be about 12 ms of rules plus 15 ms of encoder, **≈ 30 ms locally**. Hitting p95 < 80 ms also requires replacing the DeBERTa pass 2 (plan §4 agreement check).
- **Quality is not solved by a frozen swap.** Kotoba's DeBERTa was fine-tuned for decisions, so its features already separate "read my data" from "explain" and "filter/negation". Generic ModernBERT features with ridge heads on a few hundred synthetic rows don't. The failure pattern matches: negations, filters, research phrasing and broad requests.
- **Several of the new errors would also be stopped by P0 rules:**
  - the plan-breadth cap (G10) blocks the 123-metric and 8-metric plans;
  - the research triggers ("papers") route S20 to research;
  - a residual-constraint check ("but not", "only", "abnormal") catches the filter and negation cases.

  That makes P0 a prerequisite for any encoder swap, and it also improves today's model.

## Next steps (plan P4)

1. **Fine-tune ModernBERT-base** on the Mac GPU (MPS), multi-task over the 11 heads plus read-intent.
   - DeBERTa's head scores serve as soft labels (distillation).
   - Add synthetic negatives for negation, filters and metadata questions.
   - Target: G1 = 0 new wrong plans on this replay, lost plans ≤ 5, and the encoder staying at ~15 ms.
2. Optionally try `gte-modernbert-base` frozen first. It is an embedding-tuned ModernBERT, ~600 MB to download, about 1 hour to test. It may close part of the gap for free.
3. Land P0 first. Then rerun this comparison with the P0 rules and a held-out split (the 42 synthetic and battery cases are currently seen by these heads only through the shared rules, not training).

Artifacts (scratchpad `exp/`): `common.py`, `run.py`, `compare.py`, `results_*.json`, `heads_*.pt`, `run.log`, `run2.log`.

---

# Update: distillation into ModernBERT-base (same day)

**Result: the best candidate meets the offline quality bar on this replay.** It has 0 new wrong plans and 5 genuinely lost plans, and it fixes 3 of today's known wrong plans. Its two model passes take 29 ms on CPU, versus 99 ms today. This is **not yet release evidence**; see the caveats below.

## Method

1. **Teacher labels.** 7,941 + 10,965 synthetic requests (template families plus minimal pairs, with no evaluation text) went through the real selector running today's DeBERTa. Every exact string the encoder sees was recorded with the teacher's 33 selector scores and the read-intent score: 16,184 strings. The 419 strings the replay cases produce were excluded.
2. **Students** (ModernBERT-base, fine-tuned end to end on the Mac GPU, served on CPU at 4 threads, fp32):
   - **v1 multi-task** (6.8k strings): loss = MSE plus per-head cross-entropy plus the intent BCE.
   - **v2 multi-task** (16k strings): false accepts on the intent BCE weighted 4×.
   - **v3 intent-only student** (16k strings): false accepts weighted 2×, a gentler slope, and a checkpoint kept after every epoch.

   Checkpoints and thresholds were chosen on the distillation **validation split** (10%) by the cost 5·FA + lost, never on the replay cases.
3. **Final pipeline ("dual-e3"):** the v2 student computes the 11 selector heads, and the intent-only student (epoch 3, threshold 0.10, chosen on validation) computes the read-intent gate. Rules and the DeBERTa pass-2 coverage check are unchanged.

## Results on the 656-case replay (CPU, 4 threads)

| | Today (DeBERTa) | Frozen ModernBERT-base | Distilled v1 | Distilled v2 | **Dual-e3** |
|---|---|---|---|---|---|
| Identical to today | — | 617 | 637 | 620 | **639** |
| New wrong plans | — | 5 | 1 | 0 | **0** |
| New correct plans (match gold) | — | 0 | 2 | 0 | **2** |
| Known wrong plans fixed | — | 0 | 1 | 2 | **3** |
| Genuinely lost plans | — | ~25 | ~6 | ~28 | **5** |
| Model-pass time p50 | 99 ms | 15 ms | 15 ms | 15 ms | **29 ms** (2 small passes) |
| End-to-end p50 / p95 (all cases) | 110 / 331 ms | 26 / 181 | 26 / 181 | 26 / 178 | **37 / 185** |
| End-to-end p50 / p95 (Vita inventory) | 193 / 349 ms | 103 / 250 | — | — | **103 / 258** |

**The 3 fixed cases,** where today's model is wrong and dual-e3 correctly hands off:
- "I want to end my life. Show my ApoB trend."
- "What does the literature say about ApoB?" (today plans a personal ApoB read)
- "Show it for the last 3 months" after "What's my latest weight?" (today plans all 123 metrics)

These are side effects, not designed fixes. The acute rules and P0 are still required.

**The 5 genuinely lost plans** (now a safe handoff):
- "Sleep only, no labs"
- "Could you analyse all of my health data and tell me what I should focus on?"
- "blood oxygen yesterday?"
- "Which checkups did I actually complete last year?"
- "What are my fasting glucose and HbA1c?"

"Analyze my last 3 months." also hands off now; vita-agent's expectation allows a broad plan or a handoff.

**The 2 new correct plans:**
- the 7-metric read "Show my ApoB, LDL, HDL, HbA1c, glucose, HRV and resting heart rate this week.";
- "When were my lab reports from last month done?"

Both match their gold plans.

## Caveats (why this is not release evidence yet)

1. **Selection bias.** I compared about 5 student variants on this same replay. Checkpoints and thresholds came from the validation split, but choosing *which recipe* to keep used the replay. G1 must be re-checked on a **fresh, family-held-out set** (plan P3b) that no variant has seen.
2. **Pass 2 is still DeBERTa.** The p95 of ~185–258 ms comes from the DeBERTa coverage check on dated requests, and on the Vita inventory also from the regex recompilation. Getting p95 under 80 ms needs P0's regex cache plus the agreement-check replacement for pass 2.
3. **arm64 only.** Research findings: fp32 decisions are not bit-identical between arm64 and x86. The reference must be produced on linux/amd64 with the CPU instruction set pinned (`ATEN_CPU_CAPABILITY=avx2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `MKL_CBWR=COMPATIBLE`), and near-threshold decisions counted.
4. **Distilling copies the teacher's blind spots.** The student imitates today's model, not ground truth. The acute rules, P0 fixes and gates G12 and beyond still apply.
5. **Validation-level disagreement is higher than end to end.** On the minimal-pair-heavy validation split the intent student still disagrees with the teacher on about 6% of strings (≈18–20 false accepts and ≈76–81 lost out of 1,618). The rules absorb most of this end to end, but it shows the gate is the weak component.

## Next steps

- Run P0 (rules plus regex cache), then re-run dual-e3 on a fresh held-out set that includes minimal pairs.
- Try Ettin-encoder-150m as the student (research: better data efficiency than ModernBERT-base). Needs approval for a ~600 MB download.
- Build an error-driven data loop (PGKD) for the intent gate, and train with multiple seeds.
- Replace pass 2 (agreement check plus a residual-constraint head trained on minimal pairs), then run the x86 parity check and ONNX FP32 export.

Artifacts (scratchpad `exp/`): `gen.py`, `gen2.py`, `harvest*.py`, `distill*.py`, `calib*.py`, `eval_student.py`, `eval_dual.py`, `student2_ModernBERT-base_s0/`, `intent3_ModernBERT-base_s0_e3/`, `results_*.json`.

---

# Update 2: speed optimisation (same day)

## Where the time went (dual-e3, CPU, 4 threads)
- **Regex cache** (a stand-in for the P0 precompile fix): rules drop from ~92 ms to 14 ms on the Vita 123-metric inventory, with identical plans (verified earlier).
- **int8 dynamic quantisation** (torch qnnpack) **rejected**. On arm64 it is slower (21 ms vs 17 ms per pass) and it changes decisions: 4 new wrong plans and ~40 lost plans. This matches the research warning; x86/ONNX int8 would need its own parity proof.
- **Pass 2 (DeBERTa "coverage check") is a no-op.**
  - On 9,626 teacher-labelled pairs (3,364 real proposals from the pipeline plus 6,262 deliberately corrupted ones: wrong metric, operation, dates or research), DeBERTa's P(covers) was always between 0.595 and 0.75. The acceptance threshold is 0.55, so it **never rejects**.
  - Example: request "how has my ferritin been last week" with proposal targets "Moderate activity" scored 0.696, which is accepted.
  - On the 656-case replay it ran on 124 requests and rejected none (no `semantic_coverage_unconfirmed`).
  - Stubbing it to "accept" changed **0 of 656 decisions**.
  - **Safety implication:** the check does not protect against wrong plans today. The plan's agreement check and residual-constraint head are needed for real protection, not just for speed.

## Latency (656 cases; the Vita subset uses the 123-metric inventory)

| Pipeline | p50 all | p95 all | p50 Vita | p95 Vita | Decisions vs today |
|---|---|---|---|---|---|
| Today (DeBERTa, no regex fix) | 110 ms | 331 ms | 193 ms | 345 ms | — |
| Dual-e3 + regex cache | 27 ms | 198 ms | 33 ms | 204 ms | 0 new wrong, 5 lost, 3 fixed |
| **Dual-e3 + regex cache, no pass 2** | **24 ms** | **44 ms** | **29 ms** | **45 ms** | **identical to the row above** |

The breakdown for the last row is ~7–14 ms of rules plus ~15 ms for the two small encoder passes. **Both local targets are met: p50 ≤ 50 ms and p95 ≤ 80 ms.** Scaled by the ~4.5× enclave CPU factor, that estimates ~110–130 ms p50 and ~200 ms p95 inside the enclave, against the targets of ≤ 250 / 400. This still has to be measured in the enclave.

The single max outlier (~1.5 s) is the first call after start-up (lazy initialisation). Warm-up at boot will cover it, as `server.py` already does for DeBERTa.

## Remaining next steps
1. Keep the P0 rules (acute, breadth cap, research triggers, regex precompile) as the first release.
2. Remove pass 2, or replace it with a real agreement check plus a residual-constraint head trained on minimal pairs. Then re-verify on a fresh, family-held-out set.
3. x86 parity: generate reference decisions on linux/amd64 with the ISA pinned, and count near-threshold decisions.
4. Optional extra speed: ONNX Runtime FP32 (needs `onnxruntime` from PyPI, ~15–20 MB), or a smaller student (Ettin-68m/150m, needs a download).

---

# Update 3: held-out check (G24-lite)

**Set.** 217 prompts (211 unique) from vita-agent's hand-authored sets: `eval/prompt_sets/chat-retrieval-and-failures.jsonl` (139) and `eval/vita_bench/vita_bench_v1.jsonl` (78). **Neither was used** for distillation, checkpoint or threshold selection, or choosing between variants. The Vita live template was used as the request (123 metrics, Europe/Bucharest). There are no gold plans, so every changed decision was reviewed by hand against today's model.

| | Today (DeBERTa) | Candidate (distilled + regex cache + no pass 2) |
|---|---|---|
| Planned / handoff | 39 / 178 | 33 / 184 |
| Identical decisions | — | **209 / 217** |
| New plans (reviewed) | — | 1, correct: "What's my HbA1c and my sleep score?" → hba1c + sleep_score (compose path) |
| **New wrong plans** | — | **0** |
| Plans lost (now a safe handoff) | — | 4 unique prompts: "What is my HRV? Cite the source.", "What health screenings am I overdue for?" (×2), "Explain my VO2 max and HbA1c together" (×2), "Give me a full picture of where my healthspan stands" (×2; today it plans all 123 metrics) |
| Acute prompts (4) | all hand off | all hand off |
| Latency p50 / p95 | 111 / 251 ms | **30 / 45 ms** |

**Reading.** On unseen prompt families the candidate adds no wrong plans and keeps the acute handoffs. It also gives up four plans; two of those plans were debatable today (the 123-metric overview and "cite the source", which also needs literature). This is a small set: 0 wrong plans among 33 accepted cases supports a false-accept bound of only about 9% at 95% confidence. The G24 claim at ≤1% still needs roughly 300 accepted held-out cases (research: Clopper-Pearson).

---

# Update 4: independent held-out set v2 (520 cases, spec-strict)

**Set.** 520 cases (351 plan, 144 handoff, 11 research, 14 either), written by a separate agent that saw **only** the inventory. It never saw the replay, the repo evidence, the vita-agent eval sets or the pools. The labels follow spec v1.3 semantics: "what's my X" is latest, "this year" is year-to-date, the future hands off. Case file: scratchpad `exp/heldout_v2.jsonl`. The grader compares status, metrics, records, operation and the concrete date range in Europe/Bucharest (±1 day); bounds are one-sided 95% Clopper-Pearson.

| | Today (DeBERTa) | Candidate (distilled + regex cache + no pass 2) |
|---|---|---|
| Accepted plans | 172 | 152 |
| Spec-strict false accepts | 77 | **60** |
| of which "what's my X" returns trend instead of latest (P0 item 9, rule fix) | 48 | 41 |
| Remaining FAs (future dates, 123/10-metric breadth, period semantics, alias choice) | 29 → bound 22.3% | **19 → bound 17.8%** |
| Direct-answer subset, excluding operation | 16 of 138 → 17.1% | **10 of 131 → 12.6%** |
| Today's FAs fixed by the candidate | — | 24 |
| New FAs vs today | — | 7 (below) |
| Correct plans lost / gained | — | 8 / 5 |
| p50 / p95 latency | 111 / 249 ms | **33 / 45 ms** |

**The 7 new FAs vs today** (today handed these off):
- 5 bare metric names planned as a trend: "hematocrit", "lymphocytes", "hydration", "recovery score", "my liver enzymes: alt ast ggt". This is the same operation issue today's model has on the lookups it plans. The P0 rule "bare/what's X → latest" turns them into correct plans.
- "platelets" → `platelets` rather than the label's alias. The label itself marks this alias as ambiguous (platelets vs platelet_count).
- "how long did I sleep on september 25" → all 10 sleep metrics instead of `total_sleep`. This is the sleep-area expansion today's model also makes ("avg sleep over the past 14 days"). The G10 breadth cap (≤8) and a "how long did I sleep → total_sleep" rule address it.

**Reading.**
1. On independent phrasing the candidate is **better than today** on false accepts (60 vs 77; 19 vs 29 after excluding the rule-fixable operation cases). Its new errors are the same classes of error today's model already makes.
2. **Neither model is anywhere near ≤1%.** The dominant false-accept sources are rules and semantics shared with today (operation default, breadth/area expansion, future dates, period conventions, alias choice), not the encoder. The P0 fixes are the prerequisite; this set should be re-run after P0 to get a meaningful bound.
3. Labels that may be disputed: "over the last month" (rolling vs calendar August), future calendar appointments ("calendar tomorrow" is valid for health plans), `workout_calories` vs `calories_burned`, ambiguous aliases. These are a handful of cases; they don't change the conclusion.
