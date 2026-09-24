# Extracted unchanged functions/constants from the read-only Vita consumer.

import json

from datetime import datetime, UTC

from zoneinfo import ZoneInfo

from .jev_dates import resolve_range

GROUPS = {
    'sleep': ('total_sleep', 'sleep_efficiency'),
    'recovery': ('heart_rate_variability', 'resting_heart_rate'),
    'activity': ('steps',),
    'body': ('weight', 'body_fat'),
    'lipids': ('apob', 'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides'),
    'glucose': ('hba1c', 'glucose'),
    'fitness': ('vo2_max',),
    'blood_pressure': ('systolic_blood_pressure', 'diastolic_blood_pressure'),
}

RECORD_DATASETS = {
    'profile': 'profile.health.v1',
    'workouts': 'workouts.sessions.v1',
    'labs': 'labs.upload_metadata.v1',
    'calendar': 'calendar.events.v1',
}

OVERVIEW_PROFILE_FIELDS = (
    'goals', 'chronic_conditions', 'medications', 'no_regular_medications',
    'allergies', 'sex', 'birth_year', 'height_cm', 'weight_kg',
)

RESEARCH = {
    'broad_overview': 'For adults, summarize systematic review and randomized trial evidence on sleep duration and regularity, physical activity, and diet for cardiometabolic health. Identify practical interventions, studied populations, and applicability limits; distinguish associations from intervention effects.',
    'sleep_activity': 'For adults, what systematic review and randomized trial evidence supports improving sleep duration and regularity and physical activity? Describe practical interventions, outcomes, and applicability limits.',
    'cardiometabolic': 'For adults, what systematic review and randomized trial evidence supports physical activity and dietary changes for lipid and glucose control? Describe practical interventions, outcomes, and applicability limits.',
}

def read_arguments(response, available, *, literature_available=False, record_types=(),
                   now=None, time_zone='UTC'):
    now = now or datetime.now(UTC)
    answers = response['answers']
    task = answers['task']['choice']
    purpose = answers['purpose']['choice']
    research = answers['research']['choice'] if literature_available else 'none'
    if task == 'other' or research == 'unsupported':
        return None
    requested_range = None
    if task == 'health':
        if purpose == 'unsupported':
            return None
        try:
            requested_range = resolve_range(answers, now=now, time_zone=time_zone)
        except ValueError:
            # Unsupported arguments go to the existing model in this round, not a retry.
            return None
    broad = answers['coverage']['choice'] == 'broad'
    reads, by_window = [], {}
    if task == 'health':
        sparse = set(GROUPS['lipids'] + GROUPS['glucose'])
        frequent = {metric for metrics in GROUPS.values() for metric in metrics} - sparse
        for metric in sorted(set(available)):
            if not broad and answers[f'metric__{metric}']['choice'] == 'skip':
                continue
            default_range = ({'kind': 'all_history'} if metric not in frequent or purpose == 'latest'
                             else {'kind': 'relative', 'unit': 'days', 'amount': 30})
            selected_range = requested_range or default_range
            bucket = json.dumps(selected_range, sort_keys=True)
            if bucket not in by_window:
                read = {'operation_id': len(reads) + 1, 'purpose': purpose,
                        'concepts': [], 'range': selected_range, 'time_zone': time_zone}
                reads.append(read)
                by_window[bucket] = read
            by_window[bucket]['concepts'].append(metric)
        for name in RECORD_DATASETS:
            if name not in record_types or (not broad and answers[name]['noul'] <= 0.5):
                continue
            read = {'operation_id': len(reads) + 1, 'purpose': 'latest',
                    'record_types': [name], 'range': {'kind': 'all_history'}, 'time_zone': time_zone}
            if name == 'profile':
                read['profile_fields'] = list(OVERVIEW_PROFILE_FIELDS)
            elif name == 'workouts':
                read['range'] = {'kind': 'relative', 'unit': 'days', 'amount': 30}
            else:
                read['record_detail_dataset'] = RECORD_DATASETS[name]
                read['record_detail_limit'] = 3 if name == 'labs' else 8
            if name == 'calendar':
                basis = answers['calendar_basis']['choice']
                if basis != 'current_plans':
                    read['calendar_date_basis'] = basis
                    if requested_range is not None:
                        read['range'] = requested_range
                        if requested_range['kind'] == 'between' and 'T' in requested_range['end_at']:
                            # Stored calendar fields describe whole days, unlike measurement instants.
                            read['range'] = {**requested_range,
                                'end_at': now.astimezone(ZoneInfo(time_zone)).date().isoformat()}
                elif requested_range is not None and requested_range['kind'] != 'all_history' and not broad:
                    return None
                # Broad summaries use current plans as context, not historical occurrences.
            elif requested_range is not None and name != 'profile':
                read['range'] = requested_range
            reads.append(read)
    arguments = {'health_reads': reads, 'required_operation_ids': [r['operation_id'] for r in reads]}
    if research != 'none':
        operation_id = len(reads) + 1
        arguments['literature_reads'] = [{'operation_id': operation_id,
            'question': RESEARCH[research], 'subject_basis': 'general_overview_research',
            'basis_source_ids': [], 'task': 'evidence_summary', 'depth': 'fast'}]
        arguments['required_operation_ids'].append(operation_id)
    return arguments if arguments['required_operation_ids'] else None
