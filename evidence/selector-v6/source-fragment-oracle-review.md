# Reviewed source-filter capability change

The original Linux result remains **429/430 strict matches**:205 correct plans,224 handoffs and1 `wrong_plan`, with no invalid or missed plans. Neither the frozen fixture nor the original result is edited.

Two independent reviewers accepted exactly the plan in `reviewed-source-fragment-oracle.json` for bB-075, “Oura sleep efficiency for the past week.” The historical [v5 result](../selector-v5/additional-160.json) rejected it for `source_filter`. Existing [fragment grammar](../../selector.py) permits an omitted read verb and confirmed default trend; no imperative-wording restriction is justified.

The sole metric sleep_efficiency and provider Oura are admitted in the saved request and compatible in the [metric catalog](../../metadata/health_metrics.v1.json). “Past week” means seven preceding days under the [temporal contract](../../compat/jev_dates.py). No subject, source or time constraint is lost; the result neither requests a write nor acquires extra metrics.

The [offline native proof](source-fragment-native-proof.json) validates the actual v32 schema at8ffcdde225e7e0296038078c1679a999699cb9fb plus hashed local source. It preserves sourceoura and concept sleep_efficiency and binds2026-09-16T12:00Z through2026-09-23T12:00Z. Health-source I/O calls:0. Paths in the published hash manifest are workspace-relative.

Independent verification passed92 focused source/compiler tests. Removing the metric or source from inventory caused rejection. Six adversarial strings still handed off with deliberately permissive mocked semantics: conflicting Oura/Garmin sources, Non-Oura, mixed filtered/unfiltered subjects, excluded weekends, sending results to Garmin and an8a.m. qualifier. These are parser/contract checks, not new inference. The saved CI report remains the real-model evidence.

The alternate oracle binds frozen fixture SHA256, case ID, original null gold, exact admitted request digest and a separately constructed complete HealthRead. Its evaluator preserves the strict grade and reports reviewed acceptance separately. Any changed source, metric, interval, operation or additional query fails this alternative. It grants no authority and proves no available data or final answer. Native schema compatibility and raw strict-grade cross-counts also remain separate.
