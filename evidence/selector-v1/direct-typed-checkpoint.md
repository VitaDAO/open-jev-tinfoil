# Native typed-question selector diagnostic

**Correction, 2026-09-24:** the original 59-case scorer checks four upstream
labels, not exact executable-plan correctness. It can score identical plans
differently, and a reject-everything baseline scores 32/59. The comparison below
is historical diagnostic evidence, not a model ranking. See
[the evaluation audit and direction](model-direction.md).

The experimental `direct_selector.py` calls the published Open-JEV model's
`decide` method directly with four typed choice questions in one forward pass.
Dates and entity candidates come from deterministic local parsers; cardinality,
source filtering and unsupported operations hand off. This is a diagnostic
candidate only. It is not exposed by the API or deployed to Tinfoil or Vita.

On the same 59 assistant-authored diagnostic cases (`model-heldout.json`, SHA-256
`576e34e4cd0b5c78ced289a25d2fb1db67be22eade9a21d5a30409977eca830d`),
the native-question candidate scored **33/59, with five selected statuses and
54 unsupported statuses**. Four selected outputs compile to plans; only one
selected output satisfies the original coarse scorer. The fifth selected
output has no plan. The previous frozen-encoder ridge adapter scored 45/59,
with 16 selected outputs, 14 satisfying the coarse scorer, and 43 fallbacks.
Neither score establishes exact-plan correctness or supports replacement.

The four selected outputs that fail the coarse scorer show why isolated probes were
misleading: `m13` and `m17` selected latest where the request needed a trend;
`m26` selected a latest lab-record read for a bounded-period report request
(the ridge candidate compiles to the same plan but receives the opposite score);
`m27` selected current plans rather than completed screening events and compiles
to no plan. The
four-question prompt was not tuned on these cases. The 59 cases are now a
diagnostic development set, not an independent future validation set.

Both candidates use the exact same pinned Open-JEV model. The ridge candidate
uses pooled encoder features and a separate 449-example synthetic classifier;
the native candidate uses the model's own trained typed-decision head. The
published model card warns that healthcare questions are outside its three
training domains. Neither candidate has a proven Vita database answer or an
attested hosted selector evaluation. The existing Vita selector remains active.
