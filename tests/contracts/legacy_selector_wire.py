"""Offline v1 regression fixture adapter; never used by the serving client."""
def from_venice_request(body):
    """Explicit wire adapter for local_jev.request_body(); no criteria forwarded.

    Only versioned known question keys are accepted. Available metric identifiers
    are taken from the full dynamic inventory, not from a hardcoded group list.
    """
    questions = body['questions']
    fixed = {'task', 'coverage', 'purpose', 'calendar_basis', 'period_kind',
             *DATE_KEYS, 'profile', 'workouts', 'labs', 'calendar'}
    if not fixed <= questions.keys() or any(key not in fixed | {'research'} and not key.startswith('metric__') for key in questions):
        raise ValueError('Unsupported upstream selector contract')
    state = body['state']
    return {'schema_version': 'vita-selector/v1',
            'state': {key: state[key] for key in ('current_request', 'recent_user_requests', 'reference_date', 'time_zone')},
            'available_metrics': sorted(key[len('metric__'):] for key in questions if key.startswith('metric__')),
            'literature_available': 'research' in questions}


DATE_KEYS = ('calendar_unit', 'calendar_offset', 'rolling_amount', 'rolling_unit',
             'start_year', 'start_month', 'start_day', 'end_year', 'end_month', 'end_day')
