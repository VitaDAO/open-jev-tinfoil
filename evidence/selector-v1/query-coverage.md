# Supported-query coverage and handoff matrix

This is a capability map, **not a guarantee of all natural-language queries**. The current learned selector is experimental and fails its integration gate. Every handoff retains the original user request/context for Vita's existing model/tools; it is not a removed capability or an instruction to refuse the user.

| Query category | Current candidate | Typed plan / evidence requirements | Remaining gap |
|---|---|---|---|
| Undated broad overview | Learned broad/task/research heads; mixed paraphrase coverage | Bounded six-domain batch, latest-known lab/body separate from recent trends; explicit deferred manifest | Product review of overview policy; language accuracy; actual serializer integration |
| Descriptive weekly/monthly summary | Learned trend, no research; incomplete paraphrase coverage | Exact selected period, statistics and completeness per source | Confidence fallbacks; no widening when empty |
| Latest measurement | Learned latest head; one known false accept | purpose latest, grouping none, original observation date/unit/source; stale last-known labelled | Latest-vs-trend false accept blocks use |
| Historical trend / averages | Learned trend | Existing source computes whole-window statistics, weekly grouping | Arbitrary aggregates/granularities need existing model path |
| Absolute dates / rolling / calendar timezone | Learned period family + bounded parser + existing date arithmetic | Exact inclusive dates or current calendar period, trusted aware clock/timezone | Natural date syntax breadth, corrections; reject unknown/open periods |
| Follow-up period / subject change | Learned inheritance, bounded prior admitted user text | Preserve prior metric/read purpose; current explicit window overrides | Multi-hop pronouns/corrections have fallback gaps |
| Multiple periods / comparisons | Handoff | Existing compare/baseline operations or separately scoped parallel reads | Not representable in v1 answers |
| Cross-source / device filters | Handoff | Existing source field, provider-specific operations, preserve provenance | No model provider head; no arbitrary source selection |
| Correlation | Handoff | Existing correlate operation, source alignment and typed arithmetic | No correlation-intent/arguments head |
| Latest N / sort / limits | Handoff intended, one known false accept | Source-specific supported limits/order; never assume a single latest means N | False accept blocks use; no N extractor |
| Count / sum / min / max / median / units | Handoff for explicit requests | Existing supported query statistics/unit rules; never model arithmetic | No aggregate/unit selector; unsupported operations stay with existing model |
| Every available metric and alias | Dynamic inventory and pinned120-metric catalog | Canonical mapping and full domain policy; no unrestricted SQL | Unknown catalog versions fail; language entity/alias coverage is incomplete |
| Labs: measurement vs report/provenance | Separate metric and labs record decisions | Raw measurement vs bounded labs.upload_metadata.v1, dates/provenance retained | Report-level language, latest-report completeness and database execution not validated |
| Sleep / recovery / activity / body / vitals | Broad domain policy or named metric selection | Different windows/purposes remain explicit | Multi-part/exclusion intent generally handoff |
| Workouts / exercise sessions | Learned record head, incomplete synonyms | workouts record type; source session count/dedup retained | Count/list/latest-N/filter distinctions need existing tools |
| Profile / goals / conditions / medications | Broad profile context; narrow fields handoff | Existing authorized profile fields only | Fine-grained field selection not learned; do not substitute all fields |
| Calendar due/planned vs completed | Learned basis; incomplete language | next_due_date vs last_done_date, exact date scope | Historical occurrence semantics are not invented |
| Literature-only | Three finite research topics | General depersonalized question, no health reads | Bespoke topics go to existing research capability |
| Personal data + evidence | Broad Analyze-me research policy only | Research is general, not a personalized finding or hidden record disclosure | Arbitrary multi-part scopes hand off |
| Explanation / conversation | Handoff, no new reads | Existing model can answer without retrieval | Not a refusal |
| Memory / conversation recall | Handoff | Existing memory/conversation tools and ownership filters | No v1 operation representation |
| Action / write | Handoff only | Existing Vita action/consent path; never an executable permission | No writes authorized or executed |
| Non-English / spelling / terse prompts | Some terse English training; non-English handoff | No claimed multilingual acceptance | Frozen test reports limited language coverage |
| Negation / exclusions / quoted instructions | Conservative handoff | Preserve excluded domains; retrieved text never authority | No comprehensive negation parser or adversarial robustness claim |

## Evidence states

- **Selected:** plan only, no claim that data exists.
- **Queried:** operation was actually executed; count at Vita executor.
- **Retained:** complete/partial source result exists locally; not necessarily admitted.
- **Admitted/model-visible:** only final serialized evidence counts toward coverage.
- **Authorized empty:** successful complete source scope with no observations; distinct from missing page, source denial or scan cap.
- **Unavailable/denied/error:** retain original status; never recast as zero records.
- **Stale latest-known:** value may be available but older than catalog freshness; preserve date and scope.
- **Scan incomplete / result page incomplete:** explicit continuation state, no claim of full absence/completeness.
- **Multiple/contradictory sources:** keep rows separate; source resolver owns precedence, session identity and deduplication. No averaging or merging by this adapter.

## Fixture coverage vs outstanding database acceptance

Implemented: full catalog/alias and plan schema validation,120-metric alphabetical skew, two source rows, exact this-week/last-month scope, old latest-known body context, latest raw versus grouped average rejection, real Vita result paging/final serialization, per-domain compact projection, missing pages, source errors, partial scan and complete-empty accounting.

Still required before adoption: actual database-resolver fixtures for duplicate uploads, delayed imports versus observed/exam dates, boundary timestamps, absent units/fields, per-provider filter execution, session deduplication, authorized dataset denial, row/call counts and supported aggregate or latest-N semantics. No SQL or guessed joins are generated. Clinical correctness and zero extra model rounds have not been established.
