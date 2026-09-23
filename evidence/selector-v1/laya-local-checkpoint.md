# Local Laya-MLX Vita selector diagnostic

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

- 24/59 correct selector dispositions and plans.
- 15/27 expected selectable cases accepted.
- 23 incorrect plans accepted; 36 requests produced no Vita plan.
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
`f9e501c2080cc57c13d6887820329758f5351125`. These results reject a
prompt-only or checkpoint-swap replacement. They do not prove Laya is
unfixable: domain-specific training plus independent Vita labels could change
the result, but that work and its compute budget have not been approved or
measured. Exact dates, counts, metric identity and database permissions should
still be handled by code, and any model proposal must pass plan validation.

The local Mac MLX timings do not predict Linux Tinfoil CPU latency. No Laya
candidate is deployed, and the existing Vita selector remains active.
