# Vita decision-model direction and evaluation correction

2026-09-24. Audited source and saved synthetic results at
`804cfd98d5a10db1cb12d03b0eaebc9699733719`. This assessment ran no new model
workloads, trained no weights and changed no Vita source or deployment.

## Decision

Both small models can be patched and fine-tuned. Start with native Open-JEV
DeBERTa as the CPU-enclave baseline because serving work already exists. Keep
Laya as a challenger on the same corrected task and compiler. This is an
engineering choice, not a demonstrated accuracy ranking. Do not keep extending
the frozen whole-request ridge classifier merely to fit the 59 examples.

Open-JEV 27B is a separate, much larger Qwen-backed model. Its documented BF16
path requires roughly 54 GB for 27 billion weights alone, before runtime memory.
It does not meet an 8/16 GB CPU-enclave target. A later GPU comparison could
inform labels or quality expectations, but it has not been run for Vita.

## Why the previous scores do not answer the question

1. `scripts/evaluate_laya_vita.py` supplies eight available metrics but omits
   questions for `oxygen_saturation` and `respiratory_rate`. It creates only
   the missing `custom_metric` question. The extracted consumer looks up every
   metric answer for a targeted read, so even otherwise perfect answers can
   raise `KeyError`. The evaluator catches that as an ordinary fallback.
2. Laya MLX 0.1.0 silently truncates oversized question prefixes and state.
   A tokenizer-only audit of the actual questionnaire found:

   | Question | Options | Prefix tokens | State tokens left |
   | --- | ---: | ---: | ---: |
   | rolling_amount | 125 | 266 | 245 |
   | start_year | 204 | 503 | 8 |
   | end_year | 205 | 507 | 4 |

   All three retain only `choice question: Interpret current_request.` from
   their instructions. The year questions lose the actual user request because
   application metadata precedes it. The period-kind question loses its full
   instruction too. The prefix builder's minimum option allotment can exceed
   `head_max_len`; checking option-marker count does not detect these losses.
3. `m11` expects a successful weight lookup, but weight is absent from the
   shared evaluation inventory.
4. All 59 fixture rows lack exact `concepts` and `range` expectations. The
   scorer checks four upstream labels plus plan existence. Counts, source
   filters, record ordering/completeness and database results are unchecked.
   For `m26`, ridge and native Open-JEV produce identical plans, yet receive
   opposite scores because their purpose labels differ. The compiler always
   makes this record read `latest` and caps lab detail at three.
5. Thirty-two fixtures expect handoff. Rejecting every request scores 32/59.
   `selected_valid` means selection on a nominally positive case, not a
   correct accepted plan. Native selection status can even accompany no plan.

| Historical candidate | Original score | Selected statuses / executable plans | Selected outputs satisfying original scorer |
| --- | ---: | ---: | ---: |
| Frozen Open-JEV encoder + ridge | 45/59 | 16 / 16 | 14 |
| Native Open-JEV question adapter | 33/59 | 5 / 4 | 1 |
| Full Laya questionnaire | 24/59 | 23 / 23 | 0 |

These are descriptions of stored outputs under a flawed scorer, not exact-plan
accuracy. The native adapter also deliberately hands off history, source
filters, counts and research, confounding its raw score with coverage choices.

The separate compact Laya experiment changed at least one answer under reversed
option order in 13/27 plain-text cases for both checkpoints. That warrants
investigation. Its joint-label accuracy is not conclusive: the text arm omits
history, and its research instruction differs from the expected-label policy.

## What is patchable

- **Adapter and input construction:** complete candidate inventory, preserved
  option keys/descriptions, full instructions and user state, explicit token
  budgets, and visible errors distinct from model abstention.
- **Model weights:** both small model families expose training code and
  Apache-2.0 checkpoints. Fine-tune their question-conditioned decision heads
  and, if needed, encoder layers on reviewed Vita examples. Frozen pooled
  features plus 449 synthetic examples are not either model's quality ceiling.
- **Argument binding:** parse literal dates and counts, resolve timezone
  arithmetic and link metric/source candidates in code. Ambiguous binding can
  be a small learned decision or span-linking task; validate every constraint
  against the request and executable capability before accepting a plan.
- **Runtime:** Laya MLX targets Apple hardware. A Tinfoil Linux CPU candidate
  needs upstream PyTorch/ONNX measurement. Short Mac predictions of about
  8–45 ms do not imply that latency for the full selector or enclave request.

Open-JEV packs state and questions into a 512-token sequence with a 256-token
state budget. Laya independently encodes the state with each question and
batches those sequences. Neither should receive hundreds of numeric options
or one question per full-catalog metric as the default Vita design.

## Implementation sequence and acceptance

1. **Repair the common evaluation contract.** Every case declares an exact
   plan, a valid no-read result, a specific clarification or a specific handoff.
   Give it the required canonical inventory, trusted clock and context. Treat
   missing answers, truncation and compiler exceptions as harness errors.
2. **Use compact semantic questions and bounded argument candidates.** Use the
   same semantics and compiler for both models; retain dynamic option identity.
   Code owns permissions, date arithmetic, units, limits and execution.
3. **Rerun both native models before training.** Measure exact-plan correctness,
   supported-request coverage, wrong executable plans, fallback and total
   selector latency separately. Check option permutations explicitly.
4. **Fine-tune if semantic errors remain.** Use diverse reviewed query families,
   contrasting examples and shuffled choices. Keep training, threshold
   calibration and fresh family/combination evaluation separate.
5. **Require real outcomes.** All 27 declared selectable cases must produce
   their exact required plans and correct results from synthetic health or
   literature fixtures; handoff fails those cases.
   Assert the other 32 outcomes explicitly. Verify source, dates, latest versus
   aggregate, ordering, count, missing data and admitted evidence. Then test
   fresh cases and measure the full request on the target CPU enclave.

The current 59 are development regressions now. Passing them is a useful gate,
but cannot alone establish generalization, safe clinical behavior or production
readiness. No claim of 59/59 or a winning native model is justified today.

## Primary sources

- [Open-JEV DeBERTa model and limitations](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
- [Open-JEV training code](https://github.com/kotoba-lang/typed-decisions)
- [Laya model family, training and limitations](https://huggingface.co/convaiinnovations/laya)
- [Laya fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
- [Laya MLX runtime](https://github.com/mizorewww/laya-mlx)
- [Open-JEV 27B adapter and base-model requirements](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1)
