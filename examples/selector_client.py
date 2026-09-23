"""Versioned attestation-first client; release approval intentionally not preset.

The new endpoint is not in the currently deployed release. Bind release_digest
and selector_sha256 to reviewed measured artifacts, never to remote responses.
"""
import os
import re

from examples.vita_client import HOST, REPO, MODEL_REVISION

class VitaSelectorClient:
    def __init__(self, *, release_digest, selector_sha256, adapter_sha256):
        if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
               for value in (release_digest, selector_sha256, adapter_sha256)):
            raise ValueError('Reviewed release and selector SHA256 pins required')
        from tinfoil import SecureClient
        self.verifier = SecureClient(enclave=HOST, repo=REPO, transport='tls')
        self.http = self.verifier.make_secure_http_client()
        document = self.verifier.get_verification_document()
        if document is None or not document.security_verified or document.release_digest != release_digest:
            self.http.close()
            raise RuntimeError('Unapproved selector enclave release')
        self.selector_sha256 = selector_sha256
        self.adapter_sha256 = adapter_sha256
        self.token = os.environ['OPEN_JEV_API_KEY']

    def select(self, *, current_request, available_metrics, reference_date, time_zone,
               recent_user_requests=(), literature_available=False,
               available_record_types=('profile','workouts','labs','calendar')):
        body = {'schema_version': 'vita-selector/v1', 'available_metrics': list(available_metrics),
                'available_record_types': list(available_record_types),
                'literature_available': literature_available, 'state': {
                    'current_request': current_request, 'recent_user_requests': list(recent_user_requests),
                    'reference_date': reference_date, 'time_zone': time_zone}}
        response = self.http.post(f'https://{HOST}/v1/select',
            headers={'Authorization': f'Bearer {self.token}'}, json=body, timeout=15)
        response.raise_for_status()
        result = response.json()
        if (result.get('schema_version') != 'vita-selector/v1'
                or result.get('selector_sha256') != self.selector_sha256
                or result.get('adapter_sha256') != self.adapter_sha256
                or result.get('model_revision') != MODEL_REVISION
                or result.get('implementation') != 'open_jev_structured_proposal_experimental'
                or result.get('advisory') is not True
                or result.get('status') not in ('selected', 'unsupported')):
            raise RuntimeError('Unexpected selector identity/contract')
        # Validate every returned decision; caller still applies read_arguments
        # and independent capability checks.
        expected = _answer_options(available_metrics)
        answers = result.get('answers')
        if not isinstance(answers, dict) or set(answers) != set(expected):
            raise RuntimeError('Incomplete selector answers')
        for name, template in expected.items():
            answer = answers[name]
            field = 'noul' if name in ('profile', 'workouts', 'labs', 'calendar') else 'choice'
            if not isinstance(answer, dict) or set(answer) != {field}:
                raise RuntimeError('Invalid selector answer')
            if field == 'noul':
                if type(answer[field]) not in (int, float) or answer[field] not in (0, 1):
                    raise RuntimeError('Invalid record decision')
            elif not isinstance(answer[field], str) or answer[field] not in template:
                raise RuntimeError('Invalid choice decision')
        if result['status'] == 'unsupported' and answers['task']['choice'] != 'other':
            raise RuntimeError('Unsafe unsupported response')
        return result

    def plan(self, *, current_request, available_metrics, now, time_zone,
             recent_user_requests=(), literature_available=False, record_types=(),
             operation_budget=8, summary_token_budget=16000):
        from zoneinfo import ZoneInfo
        from plan_adapter import build_plan
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError('Aware trusted clock required')
        inventory=list(available_metrics)
        selection=self.select(current_request=current_request,available_metrics=inventory,
            reference_date=now.astimezone(ZoneInfo(time_zone)).date().isoformat(),time_zone=time_zone,
            recent_user_requests=recent_user_requests,literature_available=literature_available,
            available_record_types=record_types)
        return build_plan(selection,inventory,now=now,time_zone=time_zone,record_types=record_types,
            literature_available=literature_available,operation_budget=operation_budget,
            summary_token_budget=summary_token_budget)

    def close(self):
        self.http.close()


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

def _answer_options(metrics):
    options = {
        'task': {'health', 'research', 'other'}, 'coverage': {'broad', 'targeted', 'none'},
        'purpose': {'latest', 'trend', 'unsupported'},
        'calendar_basis': {'next_due_date', 'last_done_date', 'current_plans'},
        'research': {'broad_overview', 'sleep_activity', 'cardiometabolic', 'none', 'unsupported'},
        'period_kind': {'calendar', 'rolling', 'absolute', 'all_history', 'unstated', 'unsupported'},
        'calendar_unit': {'day', 'week', 'month', 'year', 'none'},
        'calendar_offset': {'previous', 'current', 'next', 'none'},
        'rolling_unit': {'days', 'weeks', 'months', 'years', 'none'},
        'rolling_amount': {str(n) for n in (*range(1,121),180,365,730)} | {'none','out_of_range'},
    }
    for boundary in ('start','end'):
        options[boundary+'_year'] = {str(n) for n in range(1900,2101)} | {'reference_year','none','out_of_range'}
        options[boundary+'_month'] = {str(n) for n in range(1,13)} | {'none','out_of_range'}
        options[boundary+'_day'] = {str(n) for n in range(1,32)} | {'none','out_of_range'}
    options['end_year'].add('inherit_start'); options['end_month'].add('inherit_start')
    options.update({f'metric__{metric}': {'include','skip'} for metric in metrics})
    options.update({record: {0,1} for record in ('profile','workouts','labs','calendar')})
    return options
