"""Date syntax of the learned parser's deterministic resolver (spec v1.8 10g.3, 10g.19)."""
from datetime import date

import pytest

from learned_parser.parse import AmbiguousDate, InvalidDate, parse_time_span

REF = date(2026, 9, 26)  # a Saturday


def span(text, mode='past'):
    got = parse_time_span(text, REF, mode)
    return None if got is None else (got[0].isoformat(), got[1].isoformat())


@pytest.mark.parametrize('text,expected', [
    ('sept 7 to 13', ('2026-09-07', '2026-09-13')),
    ('from 1 to 10 september', ('2026-09-01', '2026-09-10')),
    ('1-14 September', ('2026-09-01', '2026-09-14')),
    ('between 1 and 15 July', ('2026-07-01', '2026-07-15')),
    ('since January of 2025', ('2025-01-01', '2025-01-31')),
    ('the first half of September', ('2026-09-01', '2026-09-15')),
    ('last week of August', ('2026-08-25', '2026-08-31')),
    ('the 17th of August', ('2026-08-17', '2026-08-17')),
    ('on the 3rd', ('2026-09-03', '2026-09-03')),
    ('Monday the 28th', ('2026-09-28', '2026-09-28')),
    ('13.08.2026', ('2026-08-13', '2026-08-13')),
    ('01.08.2026', ('2026-08-01', '2026-08-01')),
    ('on 04.05', ('2026-05-04', '2026-05-04')),
    ('2026.08.01', ('2026-08-01', '2026-08-01')),
    ('from june 1 to now', ('2026-06-01', '2026-09-26')),
    ('yesterday morning', ('2026-09-25', '2026-09-25')),
    ('October', ('2025-10-01', '2025-10-31')),
    ('in September', ('2026-09-01', '2026-09-26')),
])
def test_observation_dates(text, expected):
    assert span(text) == expected


@pytest.mark.parametrize('text', ['morning! only resilience stress', 'ok now my deep sleep', 'good morning'])
def test_greetings_and_follow_up_markers_are_not_dates(text):
    assert span(text) is None


def test_next_weekend_is_future_for_observations():
    assert span('next weekend')[0] > REF.isoformat()


def test_slash_dates_stay_ambiguous_but_dotted_dates_are_day_first():
    with pytest.raises(AmbiguousDate):
        parse_time_span('03/04', REF)                 # spec §2.2
    assert span('01.08.2026') == ('2026-08-01', '2026-08-01')   # spec v1.8 10g.21


@pytest.mark.parametrize('text', ['at 04.05', '04.05 pm', 'HbA1c 5.6', 'glucose 10.05 mmol/L'])
def test_clock_times_and_values_are_not_dates(text):
    assert span(text) is None


def test_impossible_dotted_date_is_invalid():
    with pytest.raises(InvalidDate):
        parse_time_span('31.02', REF)


@pytest.mark.parametrize('text,mode,expected', [
    ('November', 'next', ('2026-11-01', '2026-11-30')),       # due wording: the next occurrence
    ('March', 'next', ('2027-03-01', '2027-03-31')),
    ('August', 'year', ('2026-08-01', '2026-08-31')),         # neutral wording: within the current calendar year
    ('September', 'year', ('2026-09-01', '2026-09-30')),      # the whole current month, future days included
    ('next week', 'next', ('2026-09-28', '2026-10-04')),
    ('last August', 'next', ('2026-08-01', '2026-08-31')),
])
def test_calendar_record_modes(text, mode, expected):
    assert span(text, mode) == expected
