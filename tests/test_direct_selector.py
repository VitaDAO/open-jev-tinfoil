"""Structural safety checks for the experimental native-question path."""
from direct_selector import DirectSelector
from selector import SelectorRequest


class FakeModel:
    def __init__(self, purpose='latest individual measurement or record'):
        self.purpose = purpose
        self.calls = 0
        self.tok = lambda text, **_: {'input_ids': text.split()}

    def decide(self, state, questions):
        self.calls += 1
        assert len(questions) == 4
        labels = ['read my health information', 'specific measurement or record',
                  self.purpose, 'descriptive personal data summary only']
        return [{'choice': label, 'confidence': .9} for label in labels]


def request(text):
    return SelectorRequest.model_validate({'schema_version': 'vita-selector/v1',
        'state': {'current_request': text, 'recent_user_requests': [],
                  'reference_date': '2026-09-23', 'time_zone': 'UTC'},
        'available_metrics': ['respiratory_rate', 'weight'], 'literature_available': True})


def test_native_question_can_select_exact_latest_metric():
    model = FakeModel()
    result = DirectSelector(model).select(request('Please show my latest respiratory rate'))
    assert model.calls == 1
    assert result['status'] == 'selected'
    assert result['answers']['purpose'] == {'choice': 'latest'}
    assert result['answers']['metric__respiratory_rate'] == {'choice': 'include'}
    assert result['answers']['metric__weight'] == {'choice': 'skip'}


def test_native_wrong_purpose_and_embedded_count_handoff():
    result = DirectSelector(FakeModel('summary or trend over time')).select(
        request('Please show my latest respiratory rate'))
    assert result['status'] == 'unsupported'
    assert 'latest_purpose_conflict' in result['reason_codes']
    count = DirectSelector(FakeModel()).select(request('Show my latest 3 lab reports'))
    assert count['status'] == 'unsupported'
    assert 'exact_record_count_requires_existing_tools' in count['reason_codes']
