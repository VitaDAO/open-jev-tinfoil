"""Experimental structured decoder; evaluate before changing the serving path."""
import hashlib
import json
import math
import re
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path
from zoneinfo import ZoneInfo

from compat.jev_dates import resolve_range
from proposal_binding import canonicalize, entities, erase, temporal, resolve_context
from query_plan import QueryRequest, QueryPlan, HealthRead, ResearchRead, validate_inventory, request_identity, query_operation_count, NIGHT_METRICS
from selector import SelectorRequest, AREAS
from learned_selector import dynamic_metrics
from schema_index import INDEX
from trained_proposal_selector import TrainedProposalSelector
from routing import MODEL_REVISION

ROOT=Path(__file__).resolve().parent
INTENT_SHA256='79057d0e2673aa2813a8ea64193d74182a33392b1221c7c06d4afc7afe18dc36'
SPELLINGS={'stpes':'steps','slep':'sleep','wk':'week','hscrp':'hs crp'}
SOURCES={'oura':'oura','garmin':'garmin','whoop':'whoop','fitbit':'fitbit',
         'withings':'withings','apple health':'apple_health','polar':'polar'}
FIELDS={field.replace('_',' '):field for field in INDEX['records']['profile']['backend_fields']}
FIELDS.update({'conditions':'chronic_conditions','meds':'medications','bio':'bio'})
_FEATURES=ContextVar('query_request_features',default=None)
REASON_CODES_BYTES=(ROOT/'metadata/reason-codes.v1.json').read_bytes()
REASON_CODES=frozenset(json.loads(REASON_CODES_BYTES)['codes'])


def read_restricted(text):
    # Restrictions are not permissions a classifier may override. Ambiguous
    # scope or later revocation is left intact for the native conversation.
    text=canonicalize(text)
    return bool(re.search(r"\b(?:do not|don't|never|must not|stop|avoid)\s+(?:show|read|access|fetch|retrieve|open|use|look (?:at|up))\b",text)
        or re.search(r'\b(?:keep|leave) (?:my |the )?(?:health |personal )?(?:records|data|reports|profile) (?:closed|unopened|private|off[- ]limits)\b',text))


ACUTE=re.compile(r"\b(?:"
    # Acute symptoms. High recall: a false positive only costs a handoff.
    r"chest (?:pain|pressure|tightness|hurts?|is hurting)|crushing|heart attack(?! risk)|stroke(?! (?:rate|volume|count|length|power|rhythm))|face (?:is )?drooping|slurred speech"
    r"|(?:can(?:no|')?t|cannot|unable to|struggling to|hard to|trouble) breath(?:e|ing)|short(?:ness)? of breath|gasping"
    r"|faint(?:ed|ing)?|pass(?:ed|ing) out|black(?:ed|ing) out|unconscious|collapsed|seizures?|convuls\w*"
    r"|cough(?:ing|ed)? up blood|vomit(?:ing|ed)? blood|bleeding (?:heavily|a lot|won'?t stop)|severe (?:bleeding|headache|allergic)"
    r"|anaphyla\w*|throat (?:is )?(?:closing|swelling)|heart (?:is )?(?:racing|pounding)|racing heart|palpitations"
    r"|(?:isn'?t|is not|not|stopped|stop) breathing|chok(?:e|ing|ed)|worst headache|sudden (?:severe )?headache|thunderclap"
    r"|(?:feel(?:s|ing)? like )?(?:i'?m|i am) dying|going to die|allergic reaction|poison(?:ed|ing)?"
    r"|dizz(?:y|iness)|light ?headed|numb(?:ness)?|confus(?:ed|ion)|shak(?:y|ing)|hypo(?:glyc\w*)?\b|(?:sugar|glucose) (?:is |was )?(?:crashing|very low|too low|dangerously)"
    # Crisis and self-harm.
    r"|suicid\w*|kill(?:ing)? my ?self|end (?:my life|it all)|take my (?:own )?life|self[- ]?harm\w*|hurt(?:ing)? my ?self|cut(?:ting)? my ?self"
    r"|(?:do not|don'?t) want to (?:live|be alive|wake up|be here)|want to die|better off dead|no reason to live|overdos\w*"
    r"|too many (?:of (?:my|the) )?(?:pills|tablets|meds|medications|sleeping pills)|on purpose"
    # Medication or insulin changes.
    r"|(?:double|triple|increase|decrease|skip|stop|quit|change|adjust|raise|lower|cut|halve|up) (?:my |the )?(?:dose|dosage|insulin|medications?|meds|pills?|tablets?)"
    r"|how (?:much|many units of) insulin|(?:more|less) insulin"
    r")\b")
READING=re.compile(r"\b(?P<what>glucose|blood sugar|sugar|bg|bp|blood pressure|resting (?:heart rate|hr|pulse)|heart rate|pulse|hr|spo2|oxygen(?: saturation)?|o2(?: sats?)?|potassium|sodium)"
                   r"\b[^0-9]{0,24}?(?<![\w.])(?P<a>\d+(?:\.\d+)?)(?:\s*(?:/|over)\s*(?P<b>\d+(?:\.\d+)?))?\s*(?P<unit>mg/dl|mmol(?:/l)?|meq(?:/l)?|%|bpm)?"
                   # A stated reading, not a digit inside a word ("a1c") or a duration/count ("2 weeks").
                   r"(?![\w%])(?!\s*(?:days?|weeks?|months?|years?|hours?|minutes?|mins?|times?|readings?|results?|measurements?)\b)")


# History, family history or an explicit general/educational framing, with no sign the
# user is affected now. Present-tense cues always win, so recall is kept.
NOT_CURRENT=re.compile(r"\b(?:turned out to be|years? ago|months ago|in (?:19|20)\d\d|(?:grand)?(?:father|mother|dad|mom|pa|ma) had|family history|runs in (?:my|the) family"
    r"|general knowledge|for (?:my )?(?:exam|class|course|poster|homework)|studying|asking for (?:a friend|general)|what (?:are|is) the (?:warning )?signs? of|difference between)\b")
CURRENT=re.compile(r"\b(?:now|right now|currently|today|tonight|this (?:morning|afternoon|evening)|still|again|just|since|keeps?|i (?:am|'m|feel|have|can'?t)|i think i)\b")


def acute_or_crisis(text):
    """Acute symptoms, crisis language, dangerous stated readings or dose changes.

    Evaluated before any other rule. It only ever forces a handoff; it never
    enables a read, so recall is preferred over precision.
    """
    text=canonicalize(text)
    if ACUTE.search(text) and not (NOT_CURRENT.search(text) and not CURRENT.search(text)):return True
    for m in READING.finditer(text):
        what,a,b,unit=m['what'],float(m['a']),m['b'],m['unit']
        if what in ('glucose','blood sugar','sugar','bg'):
            mmol=unit and unit.startswith('mmol') or (not unit and a<35)
            if (a<3.0 or a>16.7) if mmol else (a<54 or a>300):return True
        elif what in ('bp','blood pressure') and b is not None:
            if a>=180 or float(b)>=120:return True
        elif what.startswith(('spo2','oxygen','o2')):
            if 50<=a<90:return True
        elif what=='potassium':
            if a>=6.0 or a<=2.5:return True
        elif what=='sodium':
            if a<120 or a>160:return True
        elif what.startswith('resting') or not re.search(r'\b(?:workouts?|runs?|running|exercise|training|max|maximum|peak|zones?|during)\b',text):
            # Exercise heart rates of 150+ are normal; resting or unqualified ones are not.
            if a>=150:return True
    return False


OVERVIEW_METRICS=('total_sleep','sleep_efficiency','sleep_deep','sleep_rem','sleep_score','heart_rate_variability',
    'resting_heart_rate','steps','vo2_max','weight','apob','ldl','ldl_cholesterol','hba1c','fasting_glucose','hscrp')
CORE_SLEEP=('total_sleep','sleep_efficiency','sleep_deep','sleep_rem','sleep_score')


def sleep_scope(text,metrics,request):
    """Scope the "sleep" area by wording (spec §3.10d): quantity → total_sleep, quality → core 5, full breakdown → all."""
    area=set(dynamic_metrics('sleep',request.available_metrics))
    if len(metrics)<=len(CORE_SLEEP) or not set(metrics)<=area or not area:return metrics
    text=canonicalize(text)
    if re.search(r'\b(?:everything|full|complete|all (?:of )?my|every|breakdown|all the)\b',text):return metrics
    if re.search(r'\b(?:how (?:long|much)|hours?|hrs?|avg|average|mean|duration|total)\b',text):
        return [m for m in metrics if m=='total_sleep'] or metrics
    return [m for m in CORE_SLEEP if m in metrics] or metrics


SMALL_TALK=re.compile(r"(?:hi|hey|hello|ok|okay|so|thanks|thank you(?: so much| very much)?|please|go|quick question|good (?:morning|afternoon|evening)|morning|question|one more thing|also|and|btw|sorry)(?: there)?[\s,!.?]*")


READ_WORDS=re.compile(r"\b(?:latest|most recent|newest|trend\w*|only|just|skip|leave|include|duration|efficiency|reading|value|number|numbers|pull (?:that|it|them) up|show (?:it|that|them)|don'?t need|no need|different question)\b")


def unexplained_sentence(request):
    """True if the request has a further sentence with no bound metric, record or date."""
    # Clauses joined by ";" or dashes are handled by the clause binder; only whole sentences are checked here.
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',request.state.current_request) if re.search(r'[A-Za-z]',s)]
    if len(sentences)<2:return False
    for sentence in sentences:
        text=canonicalize(sentence)
        # Emoji and other symbols carry no request content.
        if SMALL_TALK.fullmatch(re.sub(r'[^\w\s\',.!?-]','',text).strip()) or entities(enhance(sentence),request.available_metrics):continue
        if temporal(text,request.state.reference_date)[1] or re.search(r'\b(?:research|studies|literature|papers?|evidence)\b',text):continue
        # Read instructions and profile fields belong to the read ("just the latest reading is fine", "show my allergies").
        if READ_WORDS.search(text) or any(re.search(r'\b'+re.escape(alias)+r'\b',text) for alias in FIELDS):continue
        return True
    return False


def explicit_overview(text):
    return bool(re.search(r"\b(?:analy[sz]e (?:me|my (?:health|data|everything))|(?:full|complete|whole) (?:health )?(?:summary|overview|report|analysis|picture of my health)"
                          r"|everything|all (?:of )?my (?:health )?data|health (?:overview|summary|analysis|assessment|report|rundown)|overall health"
                          r"|how am i doing|how'?s my health|assess (?:my|me)|evaluate my health)\b",canonicalize(text)))


TREND_WORDS=re.compile(r'\b(?:trends?|history|over|since|evolved|changed|changing|been|progress|graph|chart|average|avg|across|all|every|values|readings|results|levels)\b')
def single_value_question(text,request):
    """Latest unless the request asks for a series (spec §3.9, §3.10d).

    A window other than today/now, a trend word, or show/list wording keeps the
    trend; everything else ("what's my X", a bare "X", "X today") is one value.
    Metric names are removed first, so "daily average heart rate" is not "average".
    """
    text=canonicalize(text)
    rest=erase(text,[(a,b) for a,b,_ in entities(text,request.available_metrics)])
    # Scope instructions ("sleep only, no labs") restrict a default read; they are not a value question.
    if re.search(r'\b(?:show|pull up|display|graph|chart|plot|list|only|just|no)\b',rest) or TREND_WORDS.search(rest):
        return False
    _,spans,error=temporal(text,request.state.reference_date)
    if error:return False
    return all(re.fullmatch(r'(?:for |on )?(?:today|now|right now)',text[start:end]) for start,end in spans)


def lab_metadata_request(text):
    """Complete personal date/issuer questions over the catalog's lab records.

    Unlike a metric mention, these constructions explicitly request recorded
    facts. Every clause must bind; definitions, filters and other people do not.
    """
    aliases='(?:'+'|'.join(re.escape(a) for a in INDEX['records']['labs']['aliases'])+')'
    subject=r'my '+aliases
    date=r'(?:when were '+subject+r' (?:done|issued|performed)|what (?:are|were) (?:the )?(?:exam |examination )?dates (?:on|of|for) '+subject+r')'
    issuer=lambda target:r'(?:which (?:lab|laboratory|provider) (?:issued|produced) '+target+r'|who (?:issued|produced) '+target+r')'
    text=canonicalize(text).strip(' ?!.')
    field=r'(?:(?:test |exam |examination )?dates|(?:issuing )?(?:labs|laboratories|providers))'
    listing=r'(?:list|show(?: me)?) (?:the )?'+field+r'(?: and '+field+r')? (?:on|of|for) '+subject
    if re.fullmatch(date+'|'+issuer(subject)+'|'+listing,text):return True
    # An anaphor is permitted only after its own explicit personal subject.
    return bool(re.fullmatch(date+r',? and '+issuer(r'them'),text))


def identity():
    from trained_proposal_selector import identity as base_identity
    return hashlib.sha256(base_identity().encode()+b''.join((ROOT/p).read_bytes() for p in
        ('query_plan.py','query_selector.py','query_execution.py','metadata/reason-codes.v1.json'))+INTENT_SHA256.encode()).hexdigest()


# Catalogue words of 8+ letters. A token one edit away from exactly one of them
# is a typo; the length floor keeps ordinary words such as "testing" intact.
from schema_index import CATALOG as _CATALOG
_VOCAB=frozenset(w for metric,d in _CATALOG.items()
                 for label in [metric.replace('_',' '),d.get('display_name',''),*d.get('aliases',[])]
                 for w in re.findall(r'[a-z]+',label.lower()) if len(w)>=8)


def _one_edit(a,b):
    if a==b or abs(len(a)-len(b))>1:return False
    if len(a)==len(b):
        diff=[i for i in range(len(a)) if a[i]!=b[i]]
        return len(diff)==1 or (len(diff)==2 and diff[1]==diff[0]+1 and a[diff[0]]==b[diff[1]] and a[diff[1]]==b[diff[0]])
    short,long_=(a,b) if len(a)<len(b) else (b,a)
    return any(long_[:i]+long_[i+1:]==short for i in range(len(long_)))


def _correct_typos(text):
    def fix(match):
        word=match[0]
        if word in _VOCAB:return word
        candidates=[v for v in _VOCAB if _one_edit(word,v)]
        return candidates[0] if len(candidates)==1 else word
    return re.sub(r'\b[a-z]{8,}\b',fix,text)


RESEARCH_QUESTION=re.compile(
    r"(?:please |so )?(?:"
    r"what (?:do|does) (?:the )?(?:latest |recent |current |new )?(?:published |scientific |clinical )?"
    r"(?:research|studies|study|trials|trial|evidence|science|literature|randomi[sz]ed trials)"
    r"(?: say| show| suggest| tell us)? (?:about|on|regarding) "
    r"|is there (?:any |good |strong |scientific )?(?:evidence|research) (?:that|for|on|about) "
    r"|(?:find|show me|summari[sz]e|what is|what's) (?:the )?(?:latest |recent |current )?"
    r"(?:research|studies|evidence|literature) (?:on|about|for|regarding) "
    r"|(?:latest|recent|new) (?:research|studies) (?:on|about) "
    r"|(?:are there |is there )?(?:any )?(?:published |scientific |recent )?(?:papers|studies|research|literature|trials) (?:on|about|regarding|for) "
    r"|what (?:does|do) (?:the )?science (?:say|show|suggest) (?:about|on) "
    r")(?P<topic>[^?.!;]+?)[?.!]*")
RESEARCH_CONTEXT=re.compile(r"(?:given|considering)(?: that)? (?P<context>[^,]+),\s*(?P<rest>.+)")


def research_topic(text,request):
    """A minimised public research question from the user's own wording, or None.

    Keeps subject, intervention, outcome and catalogue metric names; drops values,
    dates and first-person context ("given my triglycerides of 220, ..." keeps
    "triglycerides" and reads the user's latest value separately).
    """
    text=re.sub(r'\s+',' ',text.strip().lower())
    context=RESEARCH_CONTEXT.fullmatch(text)
    if context:text=context['rest']
    question=RESEARCH_QUESTION.fullmatch(text)
    if not question:return None
    topic=question['topic'].strip()
    # A stated value never leaves as research text (#243): keep the subject, drop the number and unit.
    topic=re.sub(r'\s*\b(?:of|at|around|about|near|=)\s+\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|mmol|nmol/l|ng/ml|g/l|mg|%|bpm|ms)?(?!\w)','',topic)
    topic=re.sub(r'^(?:an?|the) ','',topic).strip()   # never strip "my": personal topics must stay unbound
    metrics=[]
    if context:
        metrics=sorted({v for _,_,values in entities(context['context'],request.available_metrics) for k,v in values if k=='metric'})
        if not metrics:return None
        for metric in metrics:
            name=metric.replace('_',' ')
            if name not in topic:topic+=' and '+name
    try:
        read=ResearchRead(topic=topic)
    except ValueError:
        return None
    if not metrics:return read
    return [read,HealthRead(metrics=metrics,operation='latest',period={'kind':'all_history'})]


def enhance(text):
    text=canonicalize(text)
    for word,replacement in SPELLINGS.items():text=re.sub(r'\b'+word+r'\b',replacement,text)
    text=_correct_typos(text)
    # A current value is the latest one; calendar periods were already rewritten by canonicalize.
    text=re.sub(r'\bcurrent\b(?! (?:calendar )?(?:week|month|year|day)\b)','latest',text)
    text=re.sub(r'\b(?:the )?last time (?:i |it |they |we )?(?:was |were |got |had (?:it |them )?)?(?:tested|measured|checked|taken|drawn)\b','latest',text)
    text=re.sub(r'^different question\s*[-:,]\s*','',text)
    # A discourse marker before an explicit new read is not a list item.
    # Leave projection/period corrections ("actually, just...", "make that...")
    # intact for the history resolver.
    text=re.sub(r'^actually\s*,\s*(?=(?:please\s+)?(?:show|fetch|retrieve|view|inspect)\b)','',text)
    text=re.sub(r'\blipid (?:picture|profile)\b','lipids',text)
    text=re.sub(r'\bhow much (?:have i been|am i) sleeping\b','show my total sleep',text)
    # Duration questions ask for total sleep, not the whole sleep area (before the clock check sees "hours").
    text=re.sub(r'\bhow (?:long|much|many hours) (?:did|do|have) i (?:sleep|slept)\b','what is my total sleep',text)
    text=re.sub(r"\blast[- ]night(?:'s)?\b",'last night',text)
    # Rolling windows ("over the last month") versus bare calendar periods ("last month"), spec §3.10.
    text=re.sub(r'\b(?:(?:over|in|during|for) the (?:last|past)|(?:the )?past) (year|month|week)\b',
                lambda m:{'year':'past 12 months','month':'past 1 months','week':'past 7 days'}[m[1]],text)
    text=re.sub(r'\blast 12 months\b','past 12 months',text)
    # Lab-record phrasings bind to the labs record rather than a bare "last" date phrase.
    text=re.sub(r'\bwhen was my (?:last|latest|most recent) (?:lab (?:test|report|result)|blood (?:test|work)|bloodwork|labs?)\b','when were my lab reports done',text)
    text=re.sub(r'\b(?:the )?(?:last|latest|most recent|recent) (?:lab (?:tests?|results?|reports?)|blood ?(?:tests?|work)|labs)\b','latest lab reports',text)
    text=re.sub(r'\benergy expenditure\b','calories burned',text)
    # Heart-rate variants name distinct inventory metrics (spec v1.5); workout and activity context first.
    hr=r'(?:heart rate|hr|pulse)'
    text=re.sub(r'\b(?:max|maximum|peak|highest) '+hr+r' (?:during|in|on|for|from) (?:my |the )?(?:workouts?|runs?|rides?|training|exercise|sessions?)\b','workout max hr',text)
    text=re.sub(r'\b(?:avg|average|mean) '+hr+r' (?:during|in|on|for|from) (?:my |the )?(?:workouts?|runs?|rides?|training|exercise|sessions?)\b','workout avg hr',text)
    text=re.sub(r'\b(?:max|maximum|peak|highest) '+hr+r' during (?:my )?activit(?:y|ies)\b','activity hr max',text)
    text=re.sub(r'\b(?:min|minimum|lowest) '+hr+r' during (?:my )?activit(?:y|ies)\b','activity hr min',text)
    text=re.sub(r'\bactivity '+hr+r'(?: (?:avg|average|mean))?\b','activity hr average',text)
    text=re.sub(r'\bactivity hr average (max|maximum|peak)\b','activity hr max',text)
    text=re.sub(r'(?<!workout )(?<!activity hr )\b(?:max|maximum|peak|highest) '+hr+r'\b|\b'+hr+r' peak(?:ed)?(?: at)?\b','day max hr',text)
    text=re.sub(r'\b(?:daily |day )?(?:avg|average|mean) '+hr+r' today\b','day avg hr today',text)
    text=re.sub(r'\b(?:time in |minutes in )?(?:heart rate |hr )?zone ([0-5])(?: time)?\b',r'hr zone \1',text)
    text=re.sub(r'\bmonth to date\b','this month',text)
    # "from June 2025 till now" is an open-ended window, not June alone.
    text=re.sub(r'\bfrom (.+?) (?:till|until|to|through) (?:now|today|date)\b',r'since \1',text)
    # Bare "recovery" is the recovery score; "high/daytime/sleep recovery" and "recovery duration/resilience" are other metrics.
    text=re.sub(r'(?<!high )(?<!daytime )(?<!sleep )\brecovery\b(?! (?:score|resilience|duration|time|high))','recovery score',text)
    text=re.sub(r'\bcalories (?:burned |burnt )?(?:during|from|in|on) (?:my )?(?:workouts?|runs?|rides?|exercise|training)\b','workout calories',text)
    text=re.sub(r'\b(?:all (?:of )?(?:my )?)?step history since i started tracking\b','steps all history',text)
    # Resolve a singular measurement anaphor while leaving numeric limits and
    # every source/date/action qualifier in place for the later checks.
    if re.search(r'\b(?:newest|latest|most recent)\b',text):
        text=re.sub(r'\bone(?= (?:i have|can you)|[,?!.]|$)','value',text)
        text=re.sub(r'\bnewest\b','latest',text)
    return text


def legacy_request(request,text,history=()):
    return SelectorRequest.model_validate({'schema_version':'vita-selector/v1',
        'available_metrics':request.available_metrics,'available_record_types':request.available_record_types,
        'literature_available':request.literature_available,'state':{**request.state.model_dump(),
        'current_request':text,'recent_user_requests':list(history)}})


class QuerySelector(TrainedProposalSelector):
    def __init__(self,model):
        super().__init__(model)
        raw=(ROOT/'adapters/vita-read-intent-v4.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=INTENT_SHA256:raise ValueError('Unapproved query intent weights')
        data=json.loads(raw)
        if data['format_version']!=1 or type(data['threshold']) not in (int,float) or not math.isfinite(data['threshold']):
            raise ValueError('Invalid intent artifact')
        if data['model_revision']!=MODEL_REVISION or data['question']!=self.question.instructions:raise ValueError('Encoder mismatch')
        self.intent_weights=self.torch.tensor(data['weights'],dtype=self.torch.float64)
        self.intent_threshold=data['threshold']
        if self.intent_weights.shape!=(1024,) or not self.torch.isfinite(self.intent_weights).all():raise ValueError('Invalid weights')

    def encode(self,current):
        # Only identical encoder inputs within this call share frozen features.
        # Context isolation prevents concurrent callers sharing private text;
        # select_query clears the dictionary even when inference raises.
        scope=_FEATURES.get()
        if scope is None or scope[0] is not self:return super().encode(current)
        features=scope[1]
        if current not in features:features[current]=super().encode(current)
        return features[current]

    def select_query(self,request):
        request=QueryRequest.model_validate(request)
        features={};token=_FEATURES.set((self,features))
        try:return self._select_query(request)
        finally:
            features.clear()
            _FEATURES.reset(token)

    def _select_query(self,request):
        queries=[];diagnostics={};reasons=[]
        try:
            # Acute or crisis content never takes a fast path, whatever else the request asks.
            if any(acute_or_crisis(s) for s in [request.state.current_request,*request.state.recent_user_requests]):
                raise ValueError('acute_or_crisis_requires_model')
            if any(read_restricted(s) for s in [request.state.current_request,*request.state.recent_user_requests]):
                raise ValueError('read_restriction_requires_native_context')
            if re.search(r'\b(?:translate|translation|rephrase|rewrite|paraphrase)\b',canonicalize(request.state.current_request)):
                raise ValueError('text_transformation_requires_native_context')
            # Every sentence must be part of the read. A separate sentence with no metric, record
            # or date ("i don't see the point anymore. what's my sleep score") is unexplained
            # content that only the model may interpret, whatever the encoder decides.
            if unexplained_sentence(request):
                raise ValueError('unbound_request_constraint')
            text=enhance(request.state.current_request)
            req=legacy_request(request,text,[enhance(s) for s in request.state.recent_user_requests])
            text=resolve_context(req)
            text=re.sub(r'^skip (?:the )?trend(?: line)?\s*[;,]\s*','',text)
            # Complete clauses have their own subjects and windows. Never split
            # dates such as "between June 3 and June 17" or a plain metric list.
            pieces=re.split(r'\s*(?:;|,?\s+(?:and|plus|then)\s+(?=(?:find|what|tell me|show|look at)\b))\s*',text)
            if len(pieces)>1 and ';' not in text and all(
                    not entities(p,request.available_metrics) and not re.search(r'\b(?:research|studies|trials|evidence|literature|papers?|published|science)\b',p)
                    for p in pieces[1:]):
                # An unbound tail may qualify the preceding overview. Keep the
                # whole utterance for the existing intent/coverage checks;
                # never silently delete the tail or invent another read.
                pieces=[text]
            if len(pieces)==1:
                candidate=re.split(r'\s+and\s+',text)
                if len(candidate)>1 and all(entities(s,request.available_metrics) and temporal(s,request.state.reference_date)[1] for s in candidate):
                    pieces=candidate
            for piece in pieces:
                overview=bool(re.search(r'\bhealth\b',piece) and re.search(r'\b(?:analysis|analyse|analyze|assessment|summary|rundown)\b',piece))
                if re.search(r'\b(?:research|studies|trials|evidence|literature|papers?|published|science)\b',piece) and not overview and not re.search(r'\bno (?:studies|research)\b',piece):
                    research=self.research(piece,queries,request)
                    queries.extend(research if isinstance(research,list) else [research]);continue
                queries.extend(self.health(piece,request,diagnostics))
            # Identical reads have the same subject, source, operation, window
            # and projection. Repeating a clause must not repeat acquisition.
            unique=[]
            for query in queries:
                if query not in unique:unique.append(query)
            queries=unique
            # Plan breadth cap (G10): more than 8 metrics only for an explicit overview or one named area.
            planned=sorted({m for q in queries if isinstance(q,HealthRead) for m in q.metrics})
            if len(planned)>8 and not explicit_overview(request.state.current_request) and not any(
                    re.search(r'\b'+area+r'\b',text) and set(planned)<=set(dynamic_metrics(area,request.available_metrics)) for area in AREAS):
                raise ValueError('plan_breadth_exceeded')
            result=QueryPlan(status='planned',queries=queries,time_zone=request.state.time_zone,
                selector_sha256=identity(),adapter_sha256=INTENT_SHA256,model_revision=MODEL_REVISION,
                request_sha256=request_identity(request),diagnostics=diagnostics)
            validate_inventory(result,request)
            # Budget counts actual metric chunks, not just input clauses.
            if sum(query_operation_count(q) for q in queries)>8:
                raise ValueError('operation_budget_exceeded')
            return result.model_dump(mode='json')
        except (ValueError,KeyError) as exc:
            # Closed, non-sensitive codes only. Never echo failed raw user input.
            code=str(exc)
            # Only published literals leave the selector (metadata/reason-codes.v1.json).
            if code not in REASON_CODES:code='query_contract_unrepresentable'
            reasons=[code]
        return QueryPlan(status='handoff',reason_codes=reasons,time_zone=request.state.time_zone,
            selector_sha256=identity(),adapter_sha256=INTENT_SHA256,model_revision=MODEL_REVISION,
            request_sha256=request_identity(request),diagnostics=diagnostics).model_dump(mode='json')

    def health(self,text,request,diagnostics):
        original=text;source=None;limit=None;fields=[];night=False
        if lab_metadata_request(text):
            if 'labs' not in request.available_record_types:
                raise ValueError('requested_record_category_unavailable')
            diagnostics.setdefault('clause_decisions',[]).append({
                'status':'selected','reason_codes':[],'method':'complete_lab_metadata_binding'})
            return [HealthRead(records=['labs'],operation='latest',
                               period={'kind':'all_history'},date_basis='exam_date')]
        # The date binder represents whole days and a trusted request timezone.
        # Never let a semantic confidence score erase an explicit clock/window
        # or a requested timezone override that this contract cannot represent.
        # Unit fractions (mg/dl, mmol/l, ...) are values, not IANA timezone names.
        if re.search(r'\b(?:\d{1,2}:\d{2}|\d{1,2}\s*[ap]\.?m\.?(?!\w)|time\s*zone|utc|gmt|noon|midnight|morning|afternoon|evening|hourly|hours?)\b|\b(?!(?:mg|mcg|ug|g|ng|pg|iu|u|meq|mmol|umol|nmol|pmol|mm|cm|kg)/(?:dl|l|ml|hg|m2)\b)[a-z_]+/[a-z_]+\b|\b(?:in|on|using)\b[^,;]*\btime\b',text):
            raise ValueError('clock_or_timezone_qualifier_unavailable')
        # Exact provider names are arguments; unknown provider text survives and
        # is rejected by the inherited constraint/semantic coverage checks.
        for alias,value in SOURCES.items():
            pattern=r'\b(?:from|using|recorded by|according to) (?:my |the )?'+re.escape(alias)+r'\b'
            spans=[(m.start(),m.end()) for m in re.finditer(pattern,text)]
            # A provider may directly qualify a recognized subject, as in
            # "Garmin steps". Bind only an adjacent entity; do not erase a
            # provider mention elsewhere in a comparison or an instruction.
            subjects=entities(text,request.available_metrics)
            for match in re.finditer(r'\b'+re.escape(alias)+r'\b',text):
                if re.search(r'\b(?:non|not|no|without|except|excluding|exclude|other than|all but)(?:[\s-]+(?:from|using|recorded|by|for|my|the|just|only|solely|exclusively))*[\s-]+$',text[:match.start()]):
                    raise ValueError('negated_source_filter_unavailable')
                if any(start>match.end() and not text[match.end():start].strip()
                       for start,_,_ in subjects):
                    if len(subjects)!=1:raise ValueError('source_subject_scope_ambiguous')
                    spans.append((match.start(),match.end()))
            if spans:
                if source and source!=value:raise ValueError('multiple_sources_require_separate_clauses')
                source=value;text=erase(text,spans)
        count=re.search(r'\b(?:latest|newest|most recent|last)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?=(?:lab |blood |health )?(?:reports|workouts|appointments|plans)\b)',text)
        if count:
            from proposal_binding import NUMBER_WORDS
            limit=int(count[1]) if count[1].isdigit() else {'zero':0,**NUMBER_WORDS}[count[1]]
            if not 1<=limit<=200:raise ValueError('record_limit_out_of_range')
            text=text[:count.start()]+'latest '+text[count.end():]
        if re.search(r'\b(?:uploaded|upload time|upload date|imported)\b',text):raise ValueError('upload_order_not_available')
        # Bare "CRP" is a different test from hs-CRP (spec §3.10d); never bind it to hscrp.
        if re.search(r'(?<!hs )(?<!hs-)\bcrp\b',text):raise ValueError('ambiguous_metric_alias')
        if re.search(r'\bweek of\b',text):raise ValueError('unresolved_temporal_phrase')
        if len({v for _,_,vs in entities(text,request.available_metrics) for k,v in vs if k=='metric'})>1 and re.search(r'\blatest\b',text) and temporal(text,request.state.reference_date)[1]:
            raise ValueError('multiple_periods_require_clause_binding')
        if re.search(r'\b(?:right now|currently|at the moment)\b',text) and temporal(re.sub(r'\b(?:right now|currently|at the moment)\b',' ',text),request.state.reference_date)[1]:
            raise ValueError('multiple_periods_require_clause_binding')
        # Quarters are not in the period grammar; never let them fall through as an unstated window.
        if re.search(r'\bq[1-4]\b|\bquarter\b',text):raise ValueError('unresolved_temporal_phrase')
        # A heart-rate question is never answered with a workouts listing (spec v1.5).
        if re.search(r'\b(?:heart rate|hr|pulse|bpm)\b',text) and not any(k=='metric' for _,_,vs in entities(text,request.available_metrics) for k,_ in vs):
            raise ValueError('ambiguous_metric_alias')
        # Bare "calories" could be burned, active or workout calories (spec §3.10c).
        if re.search(r'\bcalories\b',text) and not re.search(r'\b(?:calories (?:burned|burnt)|workout calories|active calories|calories active)\b',text):
            raise ValueError('ambiguous_metric_alias')
        entity_spans=entities(text,request.available_metrics)
        record_kinds={v for _,_,values in entity_spans for kind,v in values if kind=='record'}
        if record_kinds-{'profile'}:
            # Counts must be bound explicitly above. Never let the model erase
            # an alternate quantity construction and silently use the page cap.
            # Calendar quantities belong to their date span, not the row limit.
            from proposal_binding import NUMBER_WORDS
            _,date_spans,_=temporal(text,request.state.reference_date)
            remaining=erase(text,date_spans)
            if re.search(r'\b(?:\d+|zero|'+'|'.join(NUMBER_WORDS)+r')\b',remaining):
                raise ValueError('unbound_record_quantity')
        # Field-level profile requests are bound against the actual public schema.
        field_matches=[]
        for alias,field in sorted(FIELDS.items(),key=lambda item:-len(item[0])):
            for m in re.finditer(r'\b'+re.escape(alias)+r'\b',text):
                if not any(m.start()<b and a<m.end() for a,b,_ in field_matches):field_matches.append((m.start(),m.end(),field))
        if 'profile' in record_kinds or field_matches:
            if source or limit:raise ValueError('profile_source_or_limit_unavailable')
            if record_kinds-{'profile'}:raise ValueError('mixed_profile_scope_requires_clause')
            prefix=re.match(r'^(?:please )?(?:(?:can|could|would|may) you )?(?:please )?(?:show(?: me)?|fetch|retrieve|view|inspect|(?:pull|bring) up|(?:take|have) (?:a )?look at|what (?:are|is) my)\b',text)
            _,date_spans,date_error=temporal(text[prefix.end():] if prefix else text,request.state.reference_date)
            if date_spans or date_error:raise ValueError('profile_history_unavailable')
            complete=bool(re.search(r'\b(?:full|whole|complete) (?:health )?profile\b',text))
            fields=sorted({f for _,_,f in field_matches})
            restricted=bool(fields and (re.search(r'\b(?:only|just)\b',text) or
                re.search(r'\bfrom (?:my |the )?(?:full|whole|complete) (?:health )?profile\b',text)))
            additive=bool(re.search(r'\bprofile (?:and|plus)\b|\b(?:and|plus) (?:my |the )?(?:health )?profile\b',text))
            if additive and restricted:raise ValueError('ambiguous_profile_scope')
            if additive or (complete and not restricted) or not fields:fields=['all']
            rest=erase(text,field_matches)
            medication_read=bool(re.fullmatch(r'what\s+(?:am i taking|do i take)\s*[?!.]*',rest.strip()) and fields==['medications'])
            if medication_read:rest=''
            elif prefix:rest=rest[prefix.end():]
            else:raise ValueError('profile_read_intent_unconfirmed')
            # Parse read constructions before removing argument filler. Modal
            # treatment questions must not become reads by erasing "can I take".
            # enhance() rewrites "current" to "latest"; both are filler for a profile snapshot.
            rest=re.sub(r'\b(?:please|my|me|only|just|and|from|the|profile|health|full|whole|complete|current|latest|of|demographics)\b',' ',rest)
            if re.search(r'[a-z0-9]',rest):raise ValueError('unbound_profile_qualifier')
            return [HealthRead(records=['profile'],operation='latest',profile_fields=fields,
                               date_basis='current_snapshot',period={'kind':'all_history'})]
        # Exclusions operate on a named metric group, not arbitrary records or values.
        if re.search(r'\bsleep\b',text):
            only=re.search(r'\b(duration|efficiency) only\b',text)
            exclude=re.search(r'\b(?:except|excluding|without|leave) (?:sleep )?(duration|efficiency)(?: out(?: of it)?)?\b',text)
            if only or exclude:
                chosen=('total_sleep' if only[1]=='duration' else 'sleep_efficiency') if only else ('sleep_efficiency' if exclude[1]=='duration' else 'total_sleep')
                if only and exclude and only[1]==exclude[1]:raise ValueError('contradictory_metric_projection')
                spans=[(m.start(),m.end()) for m in (only,exclude) if m]
                text=erase(text,spans);text=re.sub(r'\bsleep\b',chosen.replace('_',' '),text)
                text=re.sub(r'[—,]+',' ',text)
        if re.search(r'\blast night\b',text):
            if any('sleep' not in v for _,_,vs in entities(text,request.available_metrics) for kind,v in vs if kind=='metric'):
                raise ValueError('night_requires_sleep')
            # The night basis is defined only for these metrics (query_plan.py HealthRead, Vita query_tool).
            if any(v not in NIGHT_METRICS for _,_,vs in entities(text,request.available_metrics) for kind,v in vs if kind=='metric'):
                raise ValueError('night_basis_metric_unavailable')
            night=True;text=re.sub(r'\blast night\b','today',text)
        # A model confidence score cannot certify an unbound list item. Check
        # each additional target independently so a new/unknown biomarker cannot
        # disappear from a supported multi-marker request.
        parts=re.split(r'\s*(?:,|\+|&|\band\b|\bplus\b)\s*',text)
        if len(parts)>1 and any(k=='metric' for _,_,vs in entities(text,request.available_metrics) for k,_ in vs):
            for tail in parts:
                if not tail or entities(tail,request.available_metrics):continue
                if re.fullmatch(r'(?:please )?no (?:labs|studies|research|recommendations)(?: please)?',tail):continue
                _,spans,error=temporal(tail,request.state.reference_date)
                remainder=erase(tail,spans)
                remainder=re.sub(r'\b(?:and|only|please|thanks|thank|you|values|measurements|results|levels|in|for|the|during|over|past|last|show|me|my|what|are|is|can|could|would|if|possible)\b',' ',remainder)
                if error or re.search(r'[a-z0-9]',remainder):raise ValueError('unbound_list_item')
        req=legacy_request(request,text)
        result=super().select(req)
        diagnostics.setdefault('clause_decisions',[]).append({'status':result['status'],'reason_codes':result['reason_codes']})
        if result['status']!='selected':raise ValueError((result['reason_codes'] or ['unsupported_read'])[0])
        proposal=result['diagnostics']['proposal']
        if proposal['task']!='health':raise ValueError('health_clause_required')
        answers=result['answers'];period=resolve_range(answers,now=request.reference_time,time_zone=request.state.time_zone)
        period=period or {'kind':'all_history'}
        if night:period={'kind':'between','start_at':request.state.reference_date,'end_at':request.state.reference_date}
        output=[];metrics=proposal['metrics']
        if proposal['coverage']=='broad':
            metrics=list(request.available_metrics)
            # A large inventory gets a curated overview, never the whole inventory (spec §3.10d).
            if len(metrics)>16:metrics=[m for m in OVERVIEW_METRICS if m in metrics]
        metrics=sleep_scope(original,metrics,request)
        operation=proposal['operation']
        if metrics:
            if limit:raise ValueError('metric_latest_n_not_supported')
            # Observations cannot exist in the future; future calendar records are handled below.
            if period.get('kind')=='between' and str(period['start_at'])[:10]>request.state.reference_date:
                raise ValueError('future_period_unavailable')
            # "What's my X?", a bare "X" or "X today" asks for one value (spec §3.9). Applied
            # after acceptance, so it never changes which requests are accepted.
            if (operation=='trend' and proposal['coverage']!='broad' and not night and not proposal['records']
                    and single_value_question(original,request)):
                operation='latest'
                if proposal['period']['period_kind']=='unstated':period={'kind':'all_history'}
            # The legacy fixture's default recent trend remains explicit.
            if proposal['period']['period_kind']=='unstated' and operation=='trend' and all(m in ('steps','total_sleep','sleep_efficiency') for m in metrics):
                period={'kind':'relative','amount':30,'unit':'days'}
            if proposal['coverage']=='broad' and proposal['period']['period_kind']=='unstated':
                frequent={'steps','total_sleep','sleep_efficiency'}
                for group,window in ((set(metrics)-frequent,{'kind':'all_history'}),
                                     (set(metrics)&frequent,{'kind':'relative','amount':30,'unit':'days'})):
                    if group:output.append(HealthRead(metrics=sorted(group),operation=operation,period=window,source=source))
            else:
                output.append(HealthRead(metrics=sorted(metrics),operation=operation,period=period,
                    source=source,date_basis='sleep_end_day' if night else 'observed_at'))
        for record in sorted(proposal['records'],key=('profile','workouts','labs','calendar').index):
            basis={'profile':'current_snapshot','labs':'exam_date','workouts':'started_at',
                   'calendar':proposal['calendar_basis'] if proposal['calendar_basis']!='current_plans' else 'next_due_date'}[record]
            record_period=({'kind':'relative','amount':30,'unit':'days'} if record=='workouts' and proposal['period']['period_kind']=='unstated' else period)
            if record in ('calendar','labs') and record_period['kind']=='between':
                # These datasets store dates, not instants. Year-to-date metric
                # bounds end at now; the equivalent record bound is today in
                # the trusted timezone. Explicit subday requests are rejected
                # upstream, never rounded here.
                record_period={**record_period,**{key:(datetime.fromisoformat(record_period[key]).astimezone(ZoneInfo(request.state.time_zone)).date().isoformat()
                    if len(record_period[key])>10 else record_period[key]) for key in ('start_at','end_at')}}
            output.append(HealthRead(records=[record],operation='latest',period={'kind':'all_history'} if record=='profile' else record_period,
                profile_fields=['all'] if record=='profile' else [],limit=limit,date_basis=basis,source=source))
        if proposal['research']!='none':
            output.append(ResearchRead(targets=['sleep','physical_activity','cardiometabolic_health'],interventions=['diet','exercise']))
        # Every metric or record the user named must survive into the plan; a silently dropped subject is a wrong answer.
        planned_metrics={m for r in output if isinstance(r,HealthRead) for m in r.metrics}
        planned_records={x for r in output if isinstance(r,HealthRead) for x in r.records}
        for _,_,values in entities(text,request.available_metrics):
            kinds={k for k,_ in values}
            if 'metric' in kinds and not planned_metrics&{v for k,v in values if k=='metric'} and proposal['coverage']!='broad':
                raise ValueError('unbound_list_item')
            if kinds=={'record'} and not planned_records&{v for _,v in values}:
                raise ValueError('unbound_list_item')
        return output

    def research(self,text,previous,request):
        if not request.literature_available:raise ValueError('research_not_available')
        try:
            return self.enum_research(text,previous,request)
        except ValueError as exc:
            # Topics outside the fixed vocabulary become a minimised public
            # question; unresolved references and anything personal still hand off.
            if str(exc) not in ('unbound_research_target','unbound_research_qualifier'):raise
            topic=research_topic(text,request)
            if topic is None:raise
            return topic

    def enum_research(self,text,previous,request):
        # Construct the external research question entirely from public enum slots;
        # never forward private request text, values or personal history.
        targets=[];interventions=[]
        for term,target in [('apob','apob'),('ldl','ldl_cholesterol'),('glucose','glucose'),('sleep','sleep'),
                            ('physical activity','physical_activity'),('getting more active','physical_activity')]:
            if re.search(r'\b'+term+r'\b',text):targets.append(target)
        if re.search(r'\bdiet(?:ary)?\b',text):interventions.append('diet')
        if re.search(r'\b(?:exercise|physical activity)\b',text):interventions.append('exercise')
        if re.search(r'\b(?:sleep and exercise|exercise and sleep)\b',text):targets.append('physical_activity')
        if re.search(r'\b(?:it|both|them)\b',text):
            if not previous:raise ValueError('unresolved_research_reference')
            for q in previous:
                if isinstance(q,HealthRead):
                    for metric in q.metrics:
                        target='sleep' if 'sleep' in metric else 'physical_activity' if metric=='steps' else metric
                        if target not in ('sleep','physical_activity','apob','ldl_cholesterol','glucose'):
                            raise ValueError('unbound_research_target')
                        targets.append(target)
        if not targets:raise ValueError('unbound_research_target')
        remaining=re.sub(r"don't need my own numbers",'',text)
        allowed=set('please what do does the studies study trials trial randomized randomised research evidence say says about on diet dietary exercise physical activity for lowering bringing down improving improve using sleep getting more active any are there good find published tell me and both it them then how supports support lipid glucose ldl cholesterol apob my to'.split())
        words=re.findall(r'[a-z]+|\d+',remaining)
        if any(w not in allowed for w in words):raise ValueError('unbound_research_qualifier')
        return ResearchRead(targets=sorted(set(targets)),interventions=sorted(set(interventions)),
                            goal='lower' if re.search(r'\b(?:lowering|bringing|down)\b',text) else 'improve')
