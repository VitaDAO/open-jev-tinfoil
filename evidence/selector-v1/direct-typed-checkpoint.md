# Native typed-question selector diagnostic

The experimental `direct_selector.py` calls the published Open-JEV model's
`decide` method directly with four typed choice questions in one forward pass.
Dates and entity candidates come from deterministic local parsers; cardinality,
source filtering and unsupported operations hand off. This is a diagnostic
candidate only. It is not exposed by the API or deployed to Tinfoil or Vita.

On the same 59 assistant-authored diagnostic cases (`model-heldout.json`, SHA-256
`576e34e4cd0b5c78ced289a25d2fb1db67be22eade9a21d5a30409977eca830d`),
the native-question candidate had **33/59 correct dispositions, 5/27 valid
requests accepted, 4 incorrect accepted plans, and 54 fallbacks**. The previous
frozen-encoder ridge adapter had 45/59, 15/27, 2 and 43 respectively. This
direct candidate is worse on the actual plan gate and must not replace it.

The four incorrect native-question plans show why isolated probes were
misleading: `m13` and `m17` selected latest where the request needed a trend;
`m26` selected a latest lab-record read for a bounded-period report request;
`m27` selected current plans rather than completed screening events. The
four-question prompt was not tuned on these cases. The 59 cases are now a
diagnostic development set, not an independent future validation set.

Both candidates use the exact same pinned Open-JEV model. The ridge candidate
uses pooled encoder features and a separate 449-example synthetic classifier;
the native candidate uses the model's own trained typed-decision head. The
published model card warns that healthcare questions are outside its three
training domains. Neither candidate has a proven Vita database answer or an
attested hosted selector evaluation. The existing Vita selector remains active.
