# Local Laya-MLX Vita selector diagnostic

**Correction, 2026-09-24:** this run is not a valid model-accuracy comparison.
The questionnaire omits required metric questions, large option lists silently
truncate instructions and the user request, and the scorer does not verify exact
plans. The raw outputs and timings below are preserved as historical diagnostic
evidence. See [the evaluation audit and direction](model-direction.md).

Run on 2026-09-24 on the user's Apple M5 Max, 128 GB Mac. This is a local
synthetic diagnostic, not a hosted Tinfoil or Vita deployment.

## Full Vita questionnaire

`scripts/evaluate_laya_vita.py` converted a synthetic questionnaire generated
from read-only Vita `backbone/local_jev.request_body()` into Laya's `choice`
and `noul` schema. It kept 26 questions after limiting the dynamic inventory
to the same eight synthetic metrics used in the Open-JEV diagnostic. The
questionnaire source is pinned by SHA-256
`1fb60c4263d7e45898424e56019dfa0319c2e2de462190cf5369ec3cd85bad4b`.
The script mapped Laya answers back to Vita's original answer keys and ran the
extracted Vita `read_arguments` contract. It did not run a database or see
personal data.

Model: `aac6fef/laya-mlx` at
`20aed815fc6acde75733882e7ec0e3f28aeb9717`, `laya-mlx==0.1.0`, MLX GPU.
59 synthetic cases, fixture SHA-256
`576e34e4cd0b5c78ced289a25d2fb1db67be22eade9a21d5a30409977eca830d`.

- 24/59 under the original coarse-label/fallback scorer, not exact-plan accuracy.
- 15/27 expected selectable cases accepted; this does not mean they were correct.
- 23 executable plans, all marked incorrect by that scorer; 36 requests produced
  no Vita plan, including compiler failures incorrectly counted as fallback.
- Warm model-questionnaire prediction median 1,230.43 ms; P95 1,413.69 ms.
  This excludes model load, date/metric parsing, database reads and network.
  The first model load took 180.3 ms from a local cached checkpoint.

The model often chose `latest` and `broad` even for scoped trends and records.
Its `act_probability` can be 1.0 on a wrong selection; it is not a plan
correctness measure. There was no confidence gate or deterministic validation
added to this raw questionnaire test. Such gates could reject some bad plans,
but they cannot establish the missing correct plans or database answers.

## One-factor question experiments

`scripts/ablate_laya_questions.py` asked only four compact semantic questions
on the 27 expected selectable cases, changing state form and option order.
This measures joint **head labels**, not full Vita plans.

| Checkpoint | State | Option order | Joint correct | Median request time | Changed choices after reversing order |
| --- | --- | --- | ---: | ---: | ---: |
| English | Plain request text | Original | 3/27 | 36.52 ms | 13/27 |
| English | Plain request text | Reversed | 2/27 | 36.02 ms | 13/27 |
| English | Structured current/history | Original | 0/27 | 44.63 ms | 7/27 |
| Typed decisions | Plain request text | Original | 3/27 | 8.12 ms | 13/27 |
| Typed decisions | Plain request text | Reversed | 3/27 | 8.04 ms | 13/27 |
| Typed decisions | Structured current/history | Original | 1/27 | 10.20 ms | 10/27 |

The specialized checkpoint was `aac6fef/laya-typed-decisions-mlx` at
`f9e501c2080cc57c13d6887820329758f5351125`. The option-order changes are evidence
of instability under these prompts. The joint-accuracy figures have additional
limitations: plain-text inputs omit history, and the compact research question
asks about explicit research requests while some expected labels require
research implicitly. These results do not establish the best achievable adapter
or rank model quality. Fix instruction/label agreement and input completeness
before repeating the comparison. Domain-specific training and independent Vita
labels remain possible next steps. Exact date/count handling, canonical metric
binding, permissions and plan validation remain software responsibilities.

The local Mac MLX timings do not predict Linux Tinfoil CPU latency. No Laya
candidate is deployed, and the existing Vita selector remains active.
