from datetime import datetime,UTC
import pytest
from temporal_spans import extract_temporal,extract_count_constraint
from compat.jev_dates import resolve_range

@pytest.mark.parametrize('text,kind',[
 ('My steps on 2024-02-29 please','absolute'),
 ('Show my steps in January 2025 please','absolute'),
 ('Review sleep during the previous calendar month','calendar'),
 ('Steps for the past three weeks, please','rolling'),
 ('Show my steps from 2025-12-28 through 2026-01-03','absolute'),
 ('Show my steps today','calendar'),
 ('Show my most recent steps','unstated'),
])
def test_span_extraction_without_requiring_learned_period_prediction(text,kind):
    result=extract_temporal(text,'2026-09-23')
    assert result['status'] in ('resolved','unstated')
    assert result['date_fields']['period_kind']==kind
    if result['spans']:
        span=result['spans'][0]
        assert text[span['start']:span['end']]==span['phrase']

@pytest.mark.parametrize('text',[
 'Steps on February 30, 2026','Steps since August 1','Steps before August',
 'Compare steps this month with last month','Steps on 09/10/26','Steps last spring',
 'Steps past 999 days','Steps in 1800',
])
def test_invalid_or_multiple_periods_never_become_a_valid_broader_subperiod(text):
    assert extract_temporal(text,'2026-09-23')['status']=='unsupported'


def test_count_is_distinct_from_duration():
    assert extract_count_constraint('Show my last 5 weight readings')['limit']==5
    assert extract_count_constraint('Show my latest three respiratory rate measurements')['limit']==3
    assert extract_count_constraint('Show my last 5 days of weight readings') is None
    assert extract_count_constraint('Show my weight for the last 5 days') is None


def test_time_zone_calendar_boundary_uses_existing_date_math():
    parsed=extract_temporal('Show my steps last month','2024-03-01')
    answers={k:{'choice':v} for k,v in parsed['date_fields'].items()}
    result=resolve_range(answers,now=datetime(2024,2,29,12,tzinfo=UTC),time_zone='Pacific/Kiritimati')
    assert result=={'kind':'between','start_at':'2024-02-01','end_at':'2024-02-29'}
