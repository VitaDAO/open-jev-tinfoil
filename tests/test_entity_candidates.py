from entity_candidates import catalog_candidates


def test_full_catalog_alias_longest_match_and_source_spans():
    text='Show my LDL Cholesterol and resting heart rate from Oura'
    r=catalog_candidates(text,['ldl','heart_rate','resting_heart_rate'])
    assert {(x['kind'],x['value']) for x in r['entities']}=={('metric','ldl'),('metric','resting_heart_rate'),('source','oura')}
    assert all(text[x['start']:x['end']] for x in r['entities'])


def test_explicit_new_inventory_and_exclusion_are_proposals():
    r=catalog_candidates('new biomarker and sleep, no lab reports',['new_biomarker','total_sleep'])
    assert any(x['value']=='new_biomarker' and x['kind']=='metric' for x in r['entities'])
    assert any(x['value']=='labs' and x['polarity']=='exclude' for x in r['entities'])
    assert r['advisory'] is True


def test_ambiguous_binding_does_not_invent_semantics():
    r=catalog_candidates('not only sleep but also activity',['total_sleep','steps'])
    assert r['status']=='ambiguous'
    assert catalog_candidates('breathing speed',['respiratory_rate'])['entities']==[]
