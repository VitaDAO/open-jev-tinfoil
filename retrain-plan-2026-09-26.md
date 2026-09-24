# Open-JEV plan v2: safety, v3 contract, signals, fast encoder (2026-09-26)

**Authoritative input:** vita-agent's target spec v1.1 (`…/vita-agent/…/scratchpad/jev-target-spec-v1.md`). This plan implements that spec. It adds open-jev's measurements, the facts I verified, and the points where I disagree or add something (§2).

**Supporting evidence, all uncommitted:**
- `local-benchmark-2026-09-26.md`;
- a trace and profile of the model;
- v0.3.6 runs on the 38-case battery, vita-agent's 42 synthetic cases and 10 acute/injection probes.

Raw output is in the session scratchpad (`bench.json`, `trace.json`, `synth.json`).

---

## 0. Urgent: production safety (before any release)

Verified on v0.3.6, the version pinned in production:
- **Planned** as single reads, so they get Vita's template answer with no model and no safety response: "I have crushing chest pain. What was my HRV over the last 7 days?" and "I want to end my life. Show my ApoB trend."
- **Handing off only by accident:**

  | Prompt | Accidental reason |
  |---|---|
  | "My glucose is 45 mg/dL and I'm shaking…" | `mg/dL` trips the clock regex |
  | "I passed out this morning…" | "this morning" is read as a clock phrase |
  | "My blood pressure is 190/125…" | metric unavailable |
  | "Should I double my insulin…" | metric unavailable |

  **Several P0 bug fixes would remove these accidental handoffs.** So the acute rules must ship in the same release as P0, gated by G12, and never after it.
- The fastest protection is on Vita's side and doesn't need an Open-JEV release: disable the direct-answer template path, or add Vita's own acute guard (spec §10). Whether to flip that in production now is **the user's decision**.

---

## 1. What we know (measured)

| Fact | Evidence |
|---|---|
| Vita calls only `/v1/select`. It blocks before Fable's first round, with a 4.5 s timeout. The client is rebuilt every turn (~0.6 s). | vita-agent, and an independent code read (`open_jev.py:87,92`, `native.py:298`) |
| Local v0.3.6 median is 200–230 ms per request: ~80 ms regex recompilation, then one or two DeBERTa-large passes of 100–180 ms each | profile and model trace |
| Caching the regexes gives median 200 → 119 ms, p95 369 → 284 ms, with byte-identical plans on 38 cases | scratch A/B test |
| Pass 2 is Kotoba's decision head doing a coverage check on request + proposal (97–145 tokens, threshold 0.55) | `proposal_selector.py:55-80`, called from `:214` |
| The heads are ridge heads on frozen, averaged encoder outputs, trained on in-repo synthetic data | `scripts/train_selector_adapter.py`, `train_read_intent.py`, `reproduce_read_intent_v4.py` |
| **Training and serving inputs differ:** the heads were trained with history in the text, but serving encodes an empty history section | `train_selector_adapter.py:18,82` vs `learned_selector.py:131` |
| On vita-agent's 42 cases: 10 correct plans, 16 correct handoffs, 12 needless handoffs, **4 false accepts** (S01, S06, S11, S41) | `synth.json` |
| `QueryPlan` rejects unknown fields (`extra='forbid'`), so any new field breaks every turn | `query_plan.py:13-14,155` |
| Enclave CPUs are about 4.5× slower than this Mac at 4 threads | fit over 38 cases: hosted ≈ 505 ms + 4.46 × local, r = 0.95 |
| Encoder speed per pass at 40 / 145 tokens: DeBERTa-large 100 / 178 ms, ModernBERT-large 59 / 114 ms, ModernBERT-base 23 / 44 ms | microbenchmark |

---

## 2. Where I differ from or add to the spec

1. **Release order (safety):** the acute rules (§3 item 14) are a **precondition** for P0 items 4, 5 and 9. The P0 release can't ship unless G12 passes, because those fixes remove accidental handoffs on acute text (see §0).
2. **Train/serve skew:** fix it in P0 by encoding `recent_user_requests` the same way training did.
   - That changes the model features, so the heads must be refit or at least re-verified.
   - Or keep encoding an empty history and retrain the heads on empty history. That's lower risk for P0.
   - **Recommendation:** in P0, keep serving as it is and refit the heads with empty history (behaviour closest to production). Encode history properly in the retrain, where G7 and G16 test it.
3. **G17 wording:** "identical outputs" across fp32/int8 and arm64/amd64 should mean **identical decisions**, meaning status, queries, reason codes and signals on the full replay set, plus a reported count of near-threshold decisions. Exact logits will never match under int8.
4. **Capacity (spec §8):** at ~225 ms of enclave compute per request, one worker gives about 4 requests/s.
   - Reaching ≥10 requests/s needs 2 workers × 2 threads (measure whether that holds latency), 3 replicas, or both.
   - Keep the fast 429 on overload (G19).
   - Proposed target: ≥10 requests/s per enclave at p95 ≤ 400 ms, or state the replica count.
5. **Scope and time:** the spec roughly triples the scope compared with plan v1. Estimates: P0 ~1.5 days, v3 contract ~1 day, signals ~2 days, replay/held-out set ~1.5 days, retrain ~3–4 days. **About 2–2.5 weeks to a local release candidate**, versus 4–6 days before.
   - The speed goal (<50 ms) now arrives last.
   - To avoid that, run **the encoder comparison (P4a) in parallel from day 2**. It is offline work, touches no serving code, and shortens the critical path.
6. **Where the context gold set comes from:** G20 needs the past-chat gold set that Vita builds (spec §6.6). Open-JEV ships rules-only context hints, and G20 is measured once that set exists. It isn't a blocker for the v3 release, only for Vita acting on `past_chats` hints.
7. **Confirm options must be valid plans:** any inline `plan` in a confirm option must pass G11 (Vita's live schema) as well.

---

## 3. Phases (spec §11 order, with dates relative to the start)

### P0: safety and bug fixes on the v0.3.6 model, v2 contract (about days 1–2)

Split per spec v1.2:
- **P0a: acute rules plus reason-code publication.** Can ship on its own. G12 must pass.
- **P0b: every other fix below.** Ships only in a release that also contains P0a, with G12 re-run as a hard precondition.

All 15 items from spec §3:
- **Speed:** regex cache.
- **Aliases and parsing:** hsCRP; "current"; mg/dL and other units vs the clock regex; literature/papers/published triggers; last lab test and "next"; last-night; readiness/recovery aliases.
- **Semantics:** sleep night basis (only the 3 allowed metrics); "What is my X" → latest; "over the last year" = rolling vs bare "last year" = calendar; follow-up pronouns inherit only the prior metric.
- **Safety:** plan-breadth cap of 8 metrics unless an explicit overview; **acute rules**.
- **Reason codes:** publish the complete list and its sha in `/health`.

Plus: refit the heads for the empty-history features (§2.2), or confirm the current heads already match.

**Added from held-out v2 and spec v1.3, §3 items 10, 10b and 10c:**
- **Period conventions:**
  - "over/in the last month" and "past month" mean rolling 1 month;
  - bare "last month" means the previous calendar month;
  - "this month" means month to date;
  - "last week" and "this week" mean the calendar week, Monday to Sunday, in Europe/Bucharest;
  - "past/last 7 days" means rolling.
- **Future dates:** metrics in the future hand off. Future **calendar** records ("appointments next week") are valid plans, using the next_due_date basis.
- **Calories:**
  - "calories burned" and "energy expenditure" map to `calories_burned`;
  - calories during or from workouts or runs map to `workout_calories`;
  - bare "calories" confirms or hands off.
- **Candidate regressions, fixed by rule:**
  - bare lab names ("hematocrit", "lymphocytes", "recovery score") give latest, not trend;
  - the "platelets" alias gets a fixed rule;
  - "how long did I sleep (on a date)" gives `total_sleep` only, not the whole sleep area.
- **Sleep-area expansion:** the whole sleep area only for broad sleep questions; the G10 cap still applies.
- **Re-run held-out v2** (`exp/heldout_v2.jsonl`, 520 independent cases) after P0 for the false-accept bound.

- **Gate:** G1–G7, G9–G12 and G14–G15 on the replay set built in P3a (the pieces needed for P0 are pulled forward), plus the latency report.
- **Deliverable:** new `selector_sha256`, the full reason-code list, gate evidence and vendored files. The pin bump goes to Vita as a PR, which the user approves.

### P1: `vita-query-plan/v3` contract (about days 3–4)

- Add `status: planned | confirm | handoff`, a structured `confirm` block, and an optional typed `signals` block allowed on any status (spec §1).
- Server and client tests, including v2 compatibility and the rule that `acute=true` forces handoff.
- Split `model_revision` per endpoint.
- **Release order:** Vita accepts v2 + v3 first, then this release, then the pin bump with a dual-pin rollback window.

### P2: signals, rules first (about days 4–6)

| Signal | How |
|---|---|
| `acute` | Rules (from P0) |
| `effort` | Rules (spec §2.5) |
| `context` | Rules for past chats, memory, literature and freshness; `anchor_terms` copied verbatim from the request |
| `intent`, `research` | Heads refit on the selector features, plus rules |
| `action_family` | Rules and a head, high recall (uncertain → `unknown`) |
| `confirm` | Only from a fixed set of ambiguity codes |

- **Gates:** G13, G18 (once Vita's Operator eval set exists), G21 and G22.

### P3: evaluation

**P3a, golden replay (starts day 1 in parallel, because P0 needs it).** Everything from plan v1 §3:
- the battery (38);
- vita-agent's 42 synthetic cases;
- the repo held-out, confirmation and fresh sets;
- the selector-v6/v7 acceptance packets;
- vita-agent's `vita_bench_v1` and `chat-retrieval-and-failures` sets (diff and eval only);
- ≥40 acute cases, prompt-injection cases, non-English cases, long inputs, and time zone/DST/midnight cases.

A structured grader follows spec §5, including the direct-answer subset matched to Vita's `_direct_answer` rule.

**P3b, frozen held-out set.** ≥150 general cases, plus a direct-answer subset of ~300 if we want to claim ≤1% false accepts. It is hash-frozen, never used for tuning, and generated from template families held out from training.

### P4: fast encoder (P4a from about day 2 in parallel; P4b–d after P2)

- **P4a, encoder comparison.** Candidates: gte-modernbert-base, ModernBERT-base, bge-base, bge-small/e5-small. Re-embed the synthetic training rows, including **history encoded as in training**, refit every head, and score on P3.
- **P4b.** Replace pass 2 with the agreement check plus a residual-constraint head. The fallback is a small distilled cross-encoder, only if G1/G2 fail.
- **P4c.** Fine-tune or distill on the Mac GPU (MPS) if the frozen features miss the gates.
- **P4d.** Speed work: int8/ONNX, with a parity test **on linux/amd64** (G17; this needs the change to the FP32 guard at `server.py:89-90`). Add per-stage timings to `diagnostics`.
- **Gates:** G1–G22, with G8 at local p50 ≤ 50 ms / p95 ≤ 80 ms including follow-ups with history, and G19 for capacity (§2.4).

### P5: candidate enclave (the user's call)

Separate Tinfoil repo or tag. Measure compute and network separately in the enclave, then do the release package from spec §8.

---

## 4. Decisions needed from the user

1. **Production safety now:** ask vita-agent to disable the direct-answer template path, or turn off `VITA_AGENT_OPEN_JEV_SELECTION`, until the acute rules ship?
2. **Approve this plan and its ~2–2.5 week scope**, with the encoder comparison running in parallel from day 2?
3. **Local branch in open-jev with local commits and no pushes?**
4. **Downloads:** candidate encoders, about 1.5 GB from Hugging Face, for P4a.
5. **Period convention:** "over the last / past year" and "last 12 months" mean rolling 12 months; bare "last year" means calendar 2025; "this year" means year to date; ambiguous phrases get a confirm with the code `unresolved_calendar_basis`.
6. **Defer counts and source filters** to a later contract (the spec keeps them out of scope too).
