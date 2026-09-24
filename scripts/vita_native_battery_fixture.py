"""Fictional records and independent expectations for the canonical Vita prompts.

These values are invented for this test. They are not copied from a user's
account or from historical browser answers. No database/provider client exists
in this module; only Vita's production resolver reads these local rows.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

NOW = datetime(2026, 9, 23, 12, tzinfo=UTC)
USER_ID = '11111111-1111-4111-8111-111111111111'
METRICS = ('total_sleep', 'sleep_score', 'heart_rate_variability',
           'recovery_score', 'readiness_score', 'apob', 'fasting_glucose',
           'hba1c', 'hs_crp')
SOURCES = ('whoop', 'oura', 'lab_upload')
WORKOUT_DATES = ('2026-04-01', '2026-05-17', '2026-06-02', '2026-08-17',
                 '2026-08-18', '2026-09-01', '2026-09-20', '2026-09-22')

# These expected facts are specified independently of the planner or result
# renderer. Acquiring them is necessary, not sufficient, for a correct answer.
EXPECTED = {
    1: {'count': 3, 'records': ['workouts'], 'answer_facts': 'Exactly 3 fictional workouts in September through September 23.'},
    2: {'count': 3, 'records': ['workouts'], 'answer_facts': 'Exactly 3 fictional workouts from May 17 through August 17 inclusive.'},
    3: {'count': 8, 'records': ['workouts'], 'answer_facts': 'Exactly 8 fictional workouts in all recorded history; no source restriction.'},
    4: {'metrics': ['sleep_score'], 'answer_facts': 'Oura sleep score 76 for the explicit sleep episode ending September 23.'},
    5: {'metrics': ['total_sleep'], 'answer_facts': 'Six sparse fictional sleep durations: 390, 400, 410, 420, 430, 450 minutes. No claim of complete nightly sampling.'},
    6: {'metrics': ['heart_rate_variability'], 'answer_facts': 'Five WHOOP samples in the seven-day window: 45, 50, 55, 60, 65 ms; mean 55 ms, latest 65 ms. Two days have no observations.'},
    7: {'metrics': ['recovery_score', 'readiness_score'], 'sources': ['whoop', 'oura'], 'answer_facts': 'WHOOP recovery 72 and 74 percent; Oura readiness 80 and 82 score. Different provider constructs, no definitive interchangeability.'},
    8: {'records': ['labs'], 'answer_facts': 'Latest fictional exam date September 15, 2026, despite upload timestamp September 20.'},
    9: {'metrics': ['apob'], 'answer_facts': 'ApoB 90 mg/dL June 15 then 78 mg/dL September 15; decrease 12 mg/dL, two observations only.'},
    10: {'metrics': ['fasting_glucose', 'hba1c', 'hs_crp'], 'answer_facts': 'Fasting glucose 92 mg/dL, HbA1c 5.4 percent, hsCRP 1.2 mg/L on September 15.'},
    11: {'profile_fields': ['allergies', 'medications'], 'answer_facts': 'Fictional penicillin allergy; current cetirizine 10 mg daily. This does not establish treatment appropriateness.'},
    12: {'profile_fields': ['goals'], 'answer_facts': 'Fictional goals: cardiovascular health, endurance, and healthy longevity.'},
    13: {'profile_fields': ['chronic_conditions'], 'answer_facts': 'Fictional chronic condition: migraine.'},
    14: {'metrics': ['sleep_score'], 'records': ['workouts'], 'answer_facts': 'Sleep started September 22 at 23:00 UTC and ended September 23 at 06:30; fictional workout September 22 at 18:00. Temporal coexistence, no causal conclusion.'},
    15: {'metrics': ['apob'], 'profile_fields': ['goals'], 'answer_facts': 'Use two-point ApoB trend and explicitly recorded goals. Individual treatment priorities are not established by these sparse data.'},
    16: {'research': True, 'answer_facts': 'The synthetic literature contains no treatment threshold for ApoB 78; say so. Do not invent a real guideline or paper.'},
    17: {'research': True, 'answer_facts': 'Synthetic literature does not establish population percentiles or a universal good/bad cutoff for Oura 76.'},
    18: {'research': True, 'profile_fields': ['medications', 'goals'], 'answer_facts': 'Current fictional medication and goals can scope a search; synthetic literature does not establish a longevity benefit of cetirizine.'},
    19: {'metrics': ['apob', 'total_sleep'], 'profile_fields': ['goals'], 'answer_facts': 'Ground overview in actual sparse fictional data, acknowledge missing sampling, distinguish known conditions from inferences.'},
    20: {'research': True, 'metrics': ['apob'], 'profile_fields': ['goals'], 'answer_facts': 'Combine observed fictional data with limited fictional literature; priorities require qualified reasoning, not unsupported clinical claims.'},
}


def records():
    rows = {key: [] for key in ('wearable_records', 'biomarkers', 'workout_sessions',
                               'lab_uploads', 'profile_health', 'profile_preferences',
                               'health_calendar_events', 'health_calendar_groups')}
    def metric(name, day, value, unit, source, *, episode=None):
        row = {'record_id': f'fictional-{name}-{source}-{day}', 'record_type': name,
               'recorded_at': day + 'T08:00:00Z', 'source': source, 'value': value, 'unit': unit}
        if episode:
            row['sleep_episode'] = episode
        rows['biomarkers' if source == 'lab_upload' else 'wearable_records'].append(row)
    episode = {'schema_version': 1, 'start_at': '2026-09-22T23:00:00Z', 'end_at': '2026-09-23T06:30:00Z'}
    for day, value in zip(('2026-04-24', '2026-05-24', '2026-06-24', '2026-07-24', '2026-08-24', '2026-09-23'), (390, 400, 410, 420, 430, 450)):
        metric('total_sleep', day, value, 'min', 'oura', episode=episode if day == '2026-09-23' else None)
    metric('sleep_score', '2026-09-23', 76, 'score', 'oura', episode=episode)
    for day, value in zip(('2026-09-18', '2026-09-19', '2026-09-21', '2026-09-22', '2026-09-23'), (45, 50, 55, 60, 65)):
        metric('heart_rate_variability', day, value, 'ms', 'whoop')
    for day, whoop, oura in (('2026-09-22', 72, 80), ('2026-09-23', 74, 82)):
        metric('recovery_score', day, whoop, '%', 'whoop')
        metric('readiness_score', day, oura, 'score', 'oura')
    for name, day, value, unit in (('apob', '2026-06-15', 90, 'mg/dL'), ('apob', '2026-09-15', 78, 'mg/dL'), ('fasting_glucose', '2026-09-15', 92, 'mg/dL'), ('hba1c', '2026-09-15', 5.4, '%'), ('hs_crp', '2026-09-15', 1.2, 'mg/L')):
        metric(name, day, value, unit, 'lab_upload')
    for index, day in enumerate(WORKOUT_DATES):
        rows['workout_sessions'].append({'record_id': f'fictional-workout-{index}', 'source': 'whoop',
            'payload': {'started_at': day + 'T18:00:00Z', 'ended_at': day + 'T19:00:00Z',
                        'workout_type': 'running', 'duration_minutes': 60}})
    rows['lab_uploads'] = [{'record_id': 'fictional-lab-june', 'recorded_at': '2026-06-20T10:00:00Z',
                           'payload': {'exam_date': '2026-06-15', 'lab_provider': 'Fictional Lab'}},
                          {'record_id': 'fictional-lab-september', 'recorded_at': '2026-09-20T10:00:00Z',
                           'payload': {'exam_date': '2026-09-15', 'lab_provider': 'Fictional Lab'}}]
    rows['profile_health'] = [{'record_id': 'fictional-profile', 'payload': {
        'allergies': ['penicillin'], 'medications': ['cetirizine 10 mg daily'],
        'goals': ['cardiovascular health', 'endurance', 'healthy longevity'],
        'chronic_conditions': ['migraine']}}]
    return rows


class NoExternalStore:
    async def query_biomarkers(self, **kwargs):
        raise AssertionError('fixture_direct_store_forbidden')
    async def query_wearable_readings(self, **kwargs):
        raise AssertionError('fixture_direct_store_forbidden')
    async def query_records(self, **kwargs):
        raise AssertionError('fixture_unknown_dataset_forbidden')
    async def inventory_biomarkers(self, **kwargs):
        from vita_agent.domain import BiomarkerInventory, BiomarkerInventoryRow
        data = records()['biomarkers']
        return BiomarkerInventory(rows=tuple(BiomarkerInventoryRow(record_type=metric,
            source='lab_upload', record_count=len(group),
            first_recorded_at=min(r['recorded_at'] for r in group),
            last_recorded_at=max(r['recorded_at'] for r in group))
            for metric in sorted({r['record_type'] for r in data})
            if (group := [r for r in data if r['record_type'] == metric])),
            complete=True, scanned_rows=len(data))


async def open_health(*, exhausted=True, withheld=()):
    from vita_agent.health.turn_materialization import LiveHealthTurnMaterialization, MaterializationRecordsAdapter, DOMAIN_DATASETS
    from vita_agent.health.resolver import HealthResolver
    data = records()
    calls = []
    async def backend(**kwargs):
        calls.append(deepcopy(kwargs))
        domain = next((k for k, v in DOMAIN_DATASETS.items() if v == kwargs['dataset_id']), None)
        if domain not in data:
            raise AssertionError('fixture_unknown_dataset_forbidden')
        selected = deepcopy(data[domain])
        if kwargs.get('record_types'):
            selected = [r for r in selected if r.get('record_type') in kwargs['record_types']]
        if kwargs.get('source'):
            selected = [r for r in selected if r.get('source') == kwargs['source']]
        # The source seam obeys the requested window. Returning every row for
        # every slice would duplicate records when Vita extends cached ranges.
        if domain in ('wearable_records', 'biomarkers', 'workout_sessions'):
            def in_window(row):
                raw = row.get('payload', {}).get('started_at') or row.get('recorded_at')
                if not raw:
                    return True
                instant = datetime.fromisoformat(raw.replace('Z', '+00:00'))
                return ((not kwargs.get('start_at') or instant >= kwargs['start_at'])
                        and (not kwargs.get('end_at') or instant <= kwargs['end_at']))
            selected = [r for r in selected if in_window(r)]
        return {'status': 'ok', 'records': selected, 'rows_scanned': len(selected),
                'rejected_rows': 0, 'exhausted': exhausted}
    async def fence():
        return {'source_revision': 1, 'as_of': NOW.isoformat()}
    materialization = await LiveHealthTurnMaterialization.open_live(owner_user_id=UUID(USER_ID),
        authority_scope_digest='fictional-canonical20', granted_domains=frozenset(data) - set(withheld),
        slice_reader=backend, fence_reader=fence)
    store = NoExternalStore()
    resolver = HealthResolver(store, records_reader=MaterializationRecordsAdapter(materialization, store))
    return resolver, materialization, calls


async def make_manager(question, *, model_id='synthetic', max_rounds=4, max_output_tokens=1400):
    from backbone.synthetic import fixture
    from backbone.domain import manager_for_turn
    from backbone.test_research import ResearchBoundary
    from vita_agent.health.query_tool import build_health_query_executor
    from vita_agent.kernel.agent_factory import ToolRuntime
    from vita_agent.kernel.turn_contracts import ContextCoverage
    from vita_agent.kernel.health_evidence import HealthEvidenceRegistry
    resolver, materialization, calls = await open_health()
    def frozen_executor(*args, **kwargs):
        return build_health_query_executor(*args, **kwargs, clock=lambda: NOW)
    boundary = ResearchBoundary()
    boundary.bundle['citations'][0].update(title='Fictional study of measurement interpretation',
        doi='10.1234/fictional-canonical20', url='https://doi.org/10.1234/fictional-canonical20')
    passage = ('Fictional evaluation passage, not a real medical publication. Provider scores use different proprietary '
               'scales. These fictional observations do not establish a population cutoff for sleep score 76, an '
               'ApoB treatment threshold at 78 mg/dL, or a longevity benefit for cetirizine. No causal or individual '
               'treatment conclusion can be drawn. Real clinical guidance is outside this synthetic corpus.')
    boundary.bundle['passages']['paper-1'] = passage
    boundary.bundle['claims'][0]['text'] = 'This fictional corpus cannot establish clinical thresholds or treatment benefits.'
    boundary.bundle['as_of'] = NOW.isoformat()
    with patch('vita_agent.kernel.agent_factory.build_health_query_executor', frozen_executor):
        base, _, policy = fixture(question=question, resolver=resolver, metrics=METRICS,
            model_id=model_id, max_rounds=max_rounds, user_id=USER_ID,
            categories=frozenset({'wearable', 'biomarker', 'clinical'}), health_turn=materialization)
        policy[0] = replace(policy[0], allowed_operations=frozenset({'read', 'external_research'}))
        context = base.context
        context.operation = SimpleNamespace(user_id=USER_ID)
        context.turn_store = boundary
        context.evidence_provider = boundary
        runtime = ToolRuntime(model=model_id, health=resolver, evidence=boundary,
            coverage=ContextCoverage(slices=()), health_evidence=HealthEvidenceRegistry(allowed_metrics=METRICS))
        manager = manager_for_turn(context=context, runtime=runtime, model_id=model_id,
            max_rounds=max_rounds, health_turn=materialization, client_time_zone='UTC')
    boundary.authorize = manager.check_authority
    manager.agent.model_settings = replace(manager.agent.model_settings, max_tokens=max_output_tokens)
    manager.agent.instructions += ('\nEvaluation context: all accessible health records and research are explicitly '
        'fictional synthetic fixtures. Label them fictional. The trusted health query clock is '
        '2026-09-23T12:00:00Z and timezone is UTC; last night ends September 23. Preserve the exact user request. '
        'Available metric IDs: ' + ', '.join(METRICS) + '. Available record groups: profile, workouts, labs, calendar. '
        'Possible providers: whoop, oura, lab_upload. These inventory facts do not imply complete daily sampling. '
        'Use the actual acquire_sources tool schema for data operations; research uses a fictional local provider.')
    return manager, policy, materialization, calls, boundary
