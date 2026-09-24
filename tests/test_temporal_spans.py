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
 'Steps on February 30, 2026','Steps before August','Steps since August 1 until today',
 'Compare steps this month with last month','Steps on 09/10/26','Steps last spring',
 'Steps past 999 days','Steps in 1800',
])
def test_invalid_or_multiple_periods_never_become_a_valid_broader_subperiod(text):
    assert extract_temporal(text,'2026-09-23')['status']=='unsupported'


@pytest.mark.parametrize('text,start',[
 ('Steps since August 1',('2026','8','1')),('Steps since January',('2026','1','1')),
 ('Steps since November',('2025','11','1')),('Steps since 2026-09-01',('2026','9','1')),
 ('Steps since August 15, 2025',('2025','8','15')),
])
def test_since_is_an_exact_window_ending_on_the_reference_day(text,start):
    fields=extract_temporal(text,'2026-09-23')['date_fields']
    assert (fields['start_year'],fields['start_month'],fields['start_day'])==start
    assert (fields['end_year'],fields['end_month'],fields['end_day'])==('2026','9','23')


@pytest.mark.parametrize('text',['Steps since 2027-01-01','Steps since August and last week','Steps since August 1 3'])
def test_since_never_widens_or_combines(text):
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
