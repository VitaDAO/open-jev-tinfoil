"""W6: the learned path fails closed. An exception inside the learned parser is a published handoff, never HTTP 500; "N units ago" past
the bound (days 3660, weeks 520, months/years 120) is unsupported_period (spec v1.8 10g.24); an undated 29 February that does not exist
in its resolved year is invalid on every date syntax (10g.21)."""
from datetime import date

import pytest

import query_selector
from learned_parser import parse
from learned_parser.decode import ago_out_of_range
from test_p0_rule_fixes import request, selector  # noqa: F401  (fixture)


class Raises:
    def __init__(self, exc): self.exc = exc
    def select(self, req): raise self.exc


@pytest.mark.parametrize('exc', [OverflowError('date value out of range'), RuntimeError('boom'), TypeError('x'), IndexError('y'), ZeroDivisionError()])
def test_an_exception_in_the_learned_parser_is_a_published_handoff(monkeypatch, selector, exc):
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: Raises(exc))
    result = selector.select_query(request('HRV 99999999 days ago'))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['query_contract_unrepresentable']
    assert result['reason_codes'][0] in query_selector.REASON_CODES and 'boom' not in str(result)


@pytest.mark.parametrize('text', [
    'HRV 99999999 days ago', 'HRV 3661 days ago', 'HRV 521 weeks ago', 'HRV 99999999 wks ago', 'HRV 121 months ago', 'HRV 121 mos ago',
    'weight 121 years ago', 'weight 99999999 yrs ago', 'steps since 99999999 days ago', 'steps since 600 weeks ago', 'steps since 200 months ago',
    'steps since 99999999 years ago', 'LDL 99999999 months back', 'RHR 99999999 days back', 'HRV 99999999 weeks before today',
    'HRV 1000000000000000000000 days ago'])
def test_units_ago_past_the_bound_is_out_of_range(text):
    assert ago_out_of_range(text)


@pytest.mark.parametrize('text', ['HRV 3660 days ago', 'HRV 520 weeks ago', 'HRV 120 months ago', 'weight 120 years ago', 'HRV two weeks ago',
                                  'steps since 2 weeks ago', 'LDL 2 months back', 'weight a year ago', 'steps past 400 days', 'steps'])
def test_units_ago_within_the_bound_is_read(text):
    assert not ago_out_of_range(text)


REF = date(2026, 9, 26)


@pytest.mark.parametrize('text', ['steps on 29.02', 'steps on 29/02', 'steps on 02/29', 'steps on February 29', 'steps on Feb 29th',
                                  'steps on the 29th of February', 'steps between Feb 20 and Feb 29'])
def test_undated_29_february_is_invalid_on_every_syntax(text):
    with pytest.raises(parse.InvalidDate):
        parse.parse_time_span(text, REF)


def test_undated_29_february_in_a_leap_resolved_year_and_stated_leap_years_are_read():
    leap = (date(2024, 2, 29), date(2024, 2, 29))
    assert parse.parse_time_span('steps on 29.02', date(2024, 3, 1)) == leap
    assert parse.parse_time_span('steps on 29/02', date(2024, 3, 1)) == leap
    assert parse.parse_time_span('steps on February 29', date(2024, 3, 1)) == leap
    for text in ('steps on 29.02.2024', 'steps on 29/02/2024', 'steps on February 29 2024'):
        assert parse.parse_time_span(text, REF) == leap
    assert parse.parse_time_span('steps on 28/02', REF) == (date(2026, 2, 28), date(2026, 2, 28))
    assert parse.parse_time_span('steps on February 28', REF) == (date(2026, 2, 28), date(2026, 2, 28))
