"""Experimental structured decoder; evaluate before changing the serving path."""
import hashlib
import os
import json
import math
import re
import unicodedata
from calendar import monthrange
from datetime import date, datetime, timedelta
from contextvars import ContextVar
from pathlib import Path
from zoneinfo import ZoneInfo

from compat.jev_dates import resolve_range
from proposal_binding import canonicalize, entities, erase, temporal, resolve_context
from query_plan import TOPIC_EXCLUDED, QueryRequest, QueryPlan, HealthRead, ResearchRead, validate_inventory, request_identity, query_operation_count, NIGHT_METRICS
from selector import SelectorRequest, AREAS
from learned_selector import dynamic_metrics
from schema_index import INDEX, metric_definition
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


_ACCESS=(r"(?:show|read|access|fetch|retrieve|open|opening|use|using|look(?:ing)? (?:at|up|into|through)|pull(?:ing)?|touch(?:ing)?|see|seeing|view(?:ing)?|peek(?:ing)?|dig(?:ging)? (?:up|into|through)"
         r"|go(?:ing)? (?:into|through|digging)|display(?:ed|ing)?|check(?:ing)?|brows(?:e|ing)|scan(?:ning)?|reading|opened|touched|read|accessed|shown|seen|used)")
READ_RESTRICTION=re.compile(
    r"\b(?:do not|don'?t|never|must not|stop|avoid)\s+(?:show|read|access|fetch|retrieve|open|use|look (?:at|up))\b"
    # a prohibition with what is off limits after it ("don't want you opening my sleep log", "can you not touch my notes", "rather you didn't see them")
    r"|\b(?:do not|don'?t|dont|never(?: ever)?|must not|mustn'?t|stop|avoid|shouldn'?t|should not|won'?t|can'?t|cannot|(?:can|could|would) (?:you|u) not|rather (?:you|u) (?:didn'?t|not)"
    r"|prefer (?:you|u|it) (?:not|didn'?t)|(?:don'?t|dont|do not) want (?:you|u)(?: to)?)(?:\s+[^\s,.;:!?]+){0,3}?\s+"+_ACCESS+
    r"\s+(?:my|any|anything|all (?:of )?my|them|those|these|it|(?:the )?(?:records?|data|profile|labs?|reports?|files?|logs?|notes?|messages?|numbers?|readings?))\b"
    r"|\b(?:without|refrain from|(?:as|so) long as (?:you|u)(?: don'?t)?)(?:\s+[^\s,.;:!?]+){0,3}?\s+"+_ACCESS+r"\s+(?:my|any|anything|all (?:of )?my|them|those|these|it|(?:the )?(?:records?|data|profile|labs?|reports?|files?|logs?|notes?|messages?|numbers?|readings?))\b"
    r"|\b(?:(?:weren'?t|aren'?t|isn'?t|wasn'?t|not|never|nothing|none)\b[^.?!]{0,30}\b(?:be )?(?:looked at|opened|accessed|read|touched|used|shown|pulled|viewed|gets used)|wide berth|well alone|go near|so much as glance|against my wishes"
    r"|withdraw\w* (?:access|permission|consent)|(?:taken|took|revoked?|withdrew) (?:back |away )?(?:my )?(?:permission|access|consent)|deny yourself|still blocked|keep my [\w ]{1,25}? out of it|out of bounds|not (?:to be )?(?:opened|read|touched))\b"
    r"|\b(?:off[- ]limits|hands off|no (?:peeking|snooping|reading|looking|opening|touching|accessing)|for my eyes only|not for you to (?:see|read|open)|stay (?:out of|away from)|keep (?:out of|away from)"
    r"|(?:nothing|none)\b[^.?!]{0,30}\b(?:should|must) be (?:opened|read|accessed|pulled|shown)|(?:only|just) (?:look at|use|read) my [\w ]{1,25}? and nothing else"
    r"|(?:marked )?private\b[^.?!]{0,30}\b(?:skip\w*|ignored?|off limits)|(?:not|n'?t) (?:be )?(?:displayed|shown|visible))\b")


def read_restricted(text):
    # Restrictions are not permissions a classifier may override. Ambiguous
    # scope or later revocation is left intact for the native conversation.
    text=canonicalize(text)
    return bool(READ_RESTRICTION.search(text)
        or re.search(r'\b(?:keep|leave) (?:my |the )?(?:health |personal )?(?:records|data|reports|profile) (?:closed|unopened|private|off[- ]limits)\b',text)
        or restricted_category(text))


def restricted_category(text):
    """Negation or privacy wording bound to a data category (safety review S6): "no access to my labs", "without my sleep data", "keep my labs
    out of this", "my sleep data is off the table", "my labs are private". The lists are in metadata/context-classes.v1.json."""
    data=r"(?:(?:my|the|any|all(?: of)?(?: my)?) )?(?:[\w-]+ )?"+_words('restricted_data')
    return bool(re.search(_words('restriction_before')+r"\s+"+data,text)
        or re.search(r"\b(?:keep|leave)\s+"+data+r"\s+out\b(?! of (?:range|the (?:normal |healthy |reference )?range))",text)
        or re.search(_words('restricted_data')+r"\s+(?:(?:is|are|stays?|remains?|(?:should|must|shall) (?:stay|be|remain))\s+)?(?:(?:strictly|completely|totally|absolutely|entirely|kept)\s+)?"
                     +_words('restriction_state'),text))


ACUTE=re.compile(r"\b(?:"
    # Acute symptoms. High recall: a false positive only costs a handoff.
    r"chest (?:pain|pressure|tightness|hurts?|is hurting)|crushing|heart attack(?! risk)|stroke(?! (?:rate|volume|count|length|power|rhythm))|face (?:is )?drooping|slurred speech"
    r"|(?:can(?:no|')?t|cannot|unable to|struggling to|hard to|trouble) breath(?:e|ing)|short(?:ness)? of breath|gasping"
    r"|faint(?:ed|ing)?|pass(?:ed|ing|es)? out|black(?:ed|ing|s)? out|unconscious|collapsed|seizures?|convuls\w*"
    r"|cough(?:ing|ed)? up blood|vomit(?:ing|ed)? blood|bleeding (?:heavily|a lot|won'?t stop)|severe (?:bleeding|headache|allergic)"
    r"|anaphyla\w*|throat (?:is )?(?:closing|swelling)|heart (?:is )?(?:racing|pounding)|racing heart|palpitations"
    r"|(?:isn'?t|is not|not|stopped|stop) breathing|chok(?:e|ing|ed)|worst headache|sudden (?:severe )?headache|thunderclap"
    # "I'm dying to know/see ..." is eagerness, not distress. "Dying for" stays flagged ("dying for air").
    r"|(?:feel(?:s|ing)? like )?(?:i'?m|i am) dying(?! to (?:know|see|try|hear|find out|learn|check|compare|read|watch)\b)|going to die|allergic reaction|poison(?:ed|ing)?"
    r"|dizz(?:y|iness)|light ?headed|numb(?:ness)?|confus(?:ed|ion)|shak(?:y|ing)|hypo(?:glyc\w*)?\b|(?:sugar|glucose) (?:is |was )?(?:crashing|very low|too low|dangerously)"
    # Crisis and self-harm.
    r"|suicid\w*|kill(?:ing)? my ?self|end (?:my life|it all)|take my (?:own )?life|self[- ]?harm\w*|hurt(?:ing)? my ?self|cut(?:ting)? my ?self"
    r"|(?:do not|don'?t) want to (?:live|be alive|wake up|be here)|want to die|better off dead|no reason to live|overdos\w*"
    r"|too many (?:of (?:my|the) )?(?:pills|tablets|meds|medications|sleeping pills)|on purpose"
    r"|relaps\w* (?:on |with |into )?(?:the )?(?:sh|self[- ]?harm\w*|cutting)|sh (?:relapse|urges?)|hide (?:the |my )?(?:cuts|marks|scars)"
    # Slang: "kms" (kill myself) unless it is kilometres ("5 kms", "in kms", "how many kms", "kms per week").
    r"|(?<![\d.])(?<!\d )(?<!\bin )(?<!\bmany )(?<!\bof )(?<!\bthe )(?<!\bmy )(?<!\btotal )(?<!\bcycling )kms(?!\s*(?:per\b|a day\b|/|this\b|last\b|today\b|yesterday\b|did\b|do\b|have\b|ran\b|run\b|walked\b|in total\b|so far\b))"
    r"|kys|unalive\w*|sewer ?slide\w*"
    # Euphemisms (spec v1.8 10g.33 item 1), guarded against look-alikes ("can't go on a run", "jump roping", "my baby would sleep forever").
    r"|self[- ]?delet\w*|(?:wanna|gonna|want to|going to|about to|ready to|thinking (?:about|of)|might|i'?ll) ctb|toaster bath"
    r"|(?:i|i'?d|i just|want to|wanna|wish i could|i could) (?:just )?(?:\w+ )?sleep forever"
    r"|can'?t go on(?= *(?:$|[.,!?;]|any ?more|any ?longer|like this|living))|can'?t do this any ?more|no point (?:in )?living"
    r"|(?:take|taking|took|swallow|swallowing|swallowed) all (?:of )?my (?:pills|meds|medications?|tablets|sleeping pills)"
    r"|jump(?:ing)? off (?:a |the |this |that )?(?:bridge|building|roof)|rope myself|(?<!jump )(?<!jump-)(?<!skip )(?<!skipping )roping"
    # Medication or insulin changes.
    r"|(?:double|triple|increase|decrease|skip|stop|quit|change|adjust|raise|lower|cut|halve|up) (?:taking )?(?:my |the )?(?:dose|dosage|insulin|medications?|meds|pills?|tablets?)"
    r"|ran out of (?:my )?(?:meds|medications?|insulin|pills|tablets|inhaler|prescriptions?)|go without (?:my |the )?(?:meds|medications?|insulin|pills|tablets|inhaler)?"
    r"|how (?:much|many units of) insulin|(?:more|less) insulin"
    r")\b")
# More acute and crisis wording (guards2 sweep). Same rule as ACUTE: high recall, a false positive only costs a handoff; every entry that could also
# be ordinary context is bound to a first-person or ingestion cue.
ACUTE_MORE=re.compile(r"\b(?:"
    # Stroke and neurological signs.
    r"(?:face|mouth|lip|smile)\b[^.?!]{0,30}\b(?:droop\w*|sag\w*|crooked|lopsided)|(?:arm|leg|hand|side)\b[^.?!]{0,20}\b(?:dead|limp|paraly\w*)"
    r"|(?:speech|words?|talking)\b[^.?!]{0,25}\b(?:slur\w*|mushy|garbled|jumbled|mixed up)|slurs|can'?t (?:find|get) (?:my |his |her |their )?words|(?:can'?t|cannot|unable to) speak"
    r"|sudden(?:ly)? (?:can'?t|cannot|confus\w*|numb\w*|weak\w*)|vision (?:\w+ ){0,2}(?:dark|black|double)|edges of my vision|seeing (?:spots|stars|double|flashing)|tunnel vision"
    r"|room (?:is |keeps |was )?(?:spinning|tilting|swaying)|(?:hit|banged|bumped|smashed|knocked) (?:my|his|her|their) head|head injury"
    # Breathing, colour, consciousness, collapse.
    r"|turn(?:ing|ed|s)? (?:blue|purple|grey|gray)|(?:lips?|tongue|fingers?|skin)\b[^.?!]{0,20}\b(?:blue|purple|bluish)|not (?:responding|responsive|waking|moving)|un(?:responsive|conscious)"
    r"|won'?t wake up|not wake up|(?:hasn'?t|has not) (?:come round|woken)|out cold|(?:baby|newborn|infant|toddler|child|kid|he|she)\b[^.?!]{0,15}\bfloppy|(?:limp|lethargic) (?:baby|child|kid|infant|newborn)"
    r"|(?:i|we) (?:nearly|almost) (?:fell|hit the floor)|(?:can'?t|cannot|unable to) (?:stand|walk|get up)\b[^.?!]{0,20}\b(?:dizz\w*|faint\w*|pain)"
    # Chest, heart and abdomen.
    r"|(?:pain|pressure|weight|tightness|squeez\w*|heaviness|elephant)\b[^.?!]{0,25}\b(?:chest|sternum|breastbone)|(?:chest|sternum|breastbone)\b[^.?!]{0,25}\b(?:squeez\w*|heav\w*|crush\w*|burning)"
    r"|(?:pain|ache|aching)\b[^.?!]{0,15}\b(?:radiat\w*|spread\w*|shoot\w*|going|down)\b[^.?!]{0,15}\b(?:arm|jaw|shoulder)|heart (?:is |keeps )?(?:flutter\w*|skipping|jumping|thumping|hammering|beating (?:fast|too fast|irregularly))"
    r"|(?:worst|severe|excruciating|10/10) (?:stomach|abdominal|belly|tummy)\b|rock[- ]hard (?:belly|stomach|abdomen)|stiff neck|light hurts|rash that doesn'?t fade"
    # Bleeding, poisoning, allergy, extreme heat and cold.
    r"|(?:vomit|puk|throw|threw|spit|spat|cough)\w*\b[^.?!]{0,20}\b(?:blood|coffee[- ]grounds?|bright red)|coffee[- ]grounds?|black (?:stool|poop|tarry)|tarry stool"
    r"|(?:pouring|spurting|gushing|pooling) (?:with )?blood|blood is (?:pouring|spurting|gushing)|soaked (?:through|thru)|(?:spurt\w*|pouring)\b[^.?!]{0,30}\b(?:towel|blood)|deep (?:cut|gash)|gash\w*"
    r"|(?:swallow\w*|drank|drink\w*|ate|ingest\w*|bit into|chew\w*|sipped|licked)\b[^.?!]{0,50}\b(?:poison\w*|toxic|bleach|drain cleaner|pesticide|antifreeze|rat poison|weed ?killer|paint thinner|dishwasher pods?|laundry pods?|detergent|button batter\w+)"
    r"|carbon monoxide|(?:lips?|tongue|face|throat|eyes?)\b[^.?!]{0,20}\b(?:swell\w*|swollen|puff\w*|balloon\w*)|(?:use[ds]?|inject\w*|jab\w*|gave|need\w*|reach\w*|grab\w*) (?:\w+ ){0,3}epi-?pen|heat ?stroke|hypotherm\w*|frostbite|drown\w*"
    r"|(?:had|having|threw|throwing) (?:a |two |three |another )?fits?|fit(?:s)? (?:back to back|earlier)"
    r"|(?:baby|fetus|foetus) (?:hasn'?t|has not|isn'?t|is not|stopped) (?:moved|moving|kicking)|reduced fetal movement"
    r"|voices (?:\w+ ){0,2}(?:telling|saying)|hearing voices|seeing things"
    # Crisis, self-harm and suicide: reasons, means, plans, preparations, slang and euphemisms.
    r"|(?:no|any) (?:reason|point) (?:to|in) (?:keep(?:ing)? )?(?:liv\w+|go(?:ing)? on|continu\w+|carry\w* on|be(?:ing)? (?:here|alive))|(?:life|living|it) (?:just )?(?:isn'?t|is not|ain'?t) worth (?:it|living)"
    r"|not worth (?:living|it anymore|going on)|(?:don'?t|do not) (?:see|find) (?:any |a )?(?:reason|point) (?:to|in) (?:keep\w* )?(?:liv\w+|go(?:ing)? on)|can'?t (?:keep|carry) (?:on|going)"
    r"|off(?:ing)? myself|yeet\w* myself|(?:check\w*|log\w*) (?:out|off)\b[^.?!]{0,15}\b(?:permanently|for good|from life)|sui(?:cide)? (?:thoughts?|ideation)|\bsui\b|\bsi\b (?:thoughts?|ideation)"
    r"|(?:tying|tied|tie) up (?:my )?loose ends|(?:stockpil\w*|hoard\w*|saving up) (?:my |the )?(?:pills|meds|medication\w*)|(?:research\w*|looking up|googling|searching)\b[^.?!]{0,30}\b(?:painless|ways to (?:die|go|end)|how (?:many|much)\b[^.?!]{0,15}\b(?:pills|to die))"
    r"|(?:writ\w+|wrote) (?:a |my )?(?:note|letters?)\b[^.?!]{0,20}\b(?:family|everyone|goodbye)|noose|(?:gave|giving|give|given) away (?:all )?(?:my|the) (?:stuff|things|belongings|possessions)|giving my (?:stuff|things|belongings|possessions) away"
    r"|won'?t (?:need|be needing) (?:them|it|this|any of)|(?:won'?t|will not) be (?:here|around) (?:tomorrow|much longer|for long)|(?:step\w*|jump\w*|walk\w*) (?:out )?in front of (?:a |the )?(?:train|bus|car|truck)"
    r"|(?:drive|driving|steer\w*|swerv\w*)\b[^.?!]{0,10}\b(?:car|truck)\b[^.?!]{0,10}\b(?:into|off)|(?:hang|hung|hanging) myself|(?:i'?d|i would) (?:rather|prefer to) be dead|rehears\w* (?:how )?my funeral|(?:picture|imagine)\w* (?:everyone|them)\b[^.?!]{0,30}\bafter i'?m gone"
    r"|(?:not|never|don'?t|dont|do not|won'?t) (?:to )?(?:exist|existing|wak\w+ up)\b|(?:i'?m|i am|i'?ll be|i will be|going to be|gonna be) gone (?:for good|forever|permanently)|(?:disappear|vanish)\w* (?:for good|forever|completely)|relief of (?:simply )?not exist\w*"
    r"|(?:pills?|razor|knife|rope|gun)\b[^.?!]{0,30}\b(?:end (?:it|things|my life)|make it stop|to die)|painless ways? to (?:go|die|end)|way to (?:go|die|end it)|(?:end|ending) (?:my|the) (?:pain|misery)"
    r"|(?:don'?t|do not|can'?t) (?:want|wanna|have the energy) (?:to be|for tomorrow|to see tomorrow)|(?:decided|made up my mind|mind is made up)\b[^.?!]{0,30}\b(?:not going to|end|leave|go)"
    r"|(?:better|easier) (?:off )?without me|everyone would be better off|end(?:ing|ed)? (?:it all|things)|(?:wrote|written|writing) (?:the )?letters"
    r"|elephant (?:is )?sitting on|(?:can'?t|cannot|unable to) (?:lift|raise|move) (?:my|his|her) (?:right |left )?(?:arm|leg|hand)|wheez\w*|(?:skin|ribs|chest) (?:is )?(?:sucking|pulling) in"
    r"|nose ?bleed\w*[^.?!]{0,20}\b(?:min|hour|won'?t stop|straight)|throat\b[^.?!]{0,25}\b(?:clos\w*|swell\w*)|acting drunk|(?:not|isn'?t) making sense|(?:explosive|sudden|worst|thunderclap)\b[^.?!]{0,15}\bheadache"
    r"|(?:took|taken|swallowed|ate|downed) (?:like |about |around )?\d{2,}\b[^.?!]{0,12}\b(?:tylenol|paracetamol|acetaminophen|pills|tablets|capsules|aspirin|ibuprofen|advil|meds|sleeping)"
    r"|got into (?:the )?(?:dishwasher|laundry|cleaning|bleach|medicine|pills|meds)|(?:belly|stomach|abdomen) (?:is )?rock[- ]hard|(?:pain|it) (?:is |was )?(?:a )?10/10|10/10 pain"
    r"|(?:ketones?|dipstick|keto[- ]?(?:strips?|sticks?|tests?|pee|paper|meter|strp)|urine (?:strips?|sticks?|tests?)|ketostix)\b[^.?!]{0,100}\b(?:darkest|dark|deep(?:est)?|maroon|purple|top colou?r|last block|large|80 ?\+|over 80|80 plus|maxed|off the chart|(?:8\d|9\d|[1-3]\d\d) ?mg)"
    r"|my mind is made up|(?:got|have|bought|found) (?:my|a|the) (?:rope|noose|gun|razor)|(?:scout\w*|looking (?:at|for)) (?:rooftops?|bridges?|cliffs?|high places)"
    r"|(?:after|once|when) i(?:'m| am| was| were)? gone\b(?! (?:on|for|to|from|next|this|tomorrow|away|home|abroad))|(?:if|when) i (?:was|were|am) gone|(?:not|no) (?:to have )?(?:a )?tomorrow"
    r"|who (?:gets|will get|should get|gets to have) my (?:stuff|things|belongings|possessions)|wish\w* (?:my )?(?:car|truck|bus)\b[^.?!]{0,15}\b(?:swerve|crash|hit|veer)|(?:my|my own) funeral"
    r"|disorient\w*|face\b[^.?!]{0,20}\b(?:wonky|numb|dead|funny|weird|tingl\w*)|bang(?:ed|ing) (?:my|his|her) head|repeating (?:myself|himself|herself)|unable to breath\b"
    r"|bright red (?:flecks|specks|blood|streaks)|(?:meter|glucometer|monitor|libre|cgm|dexcom|sensor)\b[^.?!]{0,25}\b(?:flash\w*|says|shows|reads|read|display\w*|went|showing)\b[^.?!]{0,10}\b(?:hi|lo)\b"
    # Overdose: a whole or half container of a medicine swallowed, "OD'd" (not "odd").
    r"|(?:took|taken|take|swallow\w*|downed|ate|eaten|chugged|popped)\b[^.?!]{0,20}\b(?:whole|entire|full|half (?:a|the)) (?:\w+ )?(?:bottle|pack|packet|box|strip|blister|jar|tub)s?\b[^.?!]{0,30}"
    r"\b(?:pills?|tablets?|meds|medications?|medicines?|capsules?|paracetamol|tylenol|acetaminophen|ibuprofen|advil|aspirin|sleeping|melatonin|antidepressants?|insulin|opioids?|\w+(?:pam|lam|done|codone|pine))|od(?:'d|-d|'ed|-ed|'ing|-ing)"
    r")\b")
# Crisis wording in pt/es/de/fr/it/ro/nl, matched without diacritics (spec v1.8 2.4 and 10g.33: non-English crisis text hands off with the acute code).
ACUTE_FOREIGN=re.compile(r"\b(?:"
    r"morrer|(?:me|te|quiero|quero|voy a|vou) matar|matar-me|suicid(?:io|ar\w*|er|armi)|nao (?:quero|aguento) (?:mais )?viver|nao aguento mais|tirar a (?:minha )?(?:propria )?vida|acabar com (?:tudo|a minha vida)"
    r"|(?:tomei|engoli) todos os (?:comprimidos|remedios|medicamentos)|sobredos\w*"
    r"|morir(?:me|se)?|matarme|quitar(?:me)? la vida|no quiero (?:seguir )?vivir|no aguanto mas|acabar con (?:todo|mi vida)|hacerme dano"
    r"|(?:tome|trague) todas (?:las|mis) (?:pastillas|pildoras|medicinas)"
    r"|sterben|umbringen|bringe? mich um|selbstmord|suizid\w*|nicht mehr leben|das leben nehmen|mich ritzen|ritze mich|alles beenden|tabletten (?:genommen|geschluckt)|uberdosis"
    r"|mourir|me tuer|en finir|plus (?:envie de )?vivre|me faire du mal|me scarifier|(?:pris|avale) toutes (?:mes|les) (?:pilules|comprimes|cachets|medicaments)|surdose"
    r"|morire|uccidermi|ammazzarmi|farla finita|voglio piu vivere|togliermi la vita|farmi del male|(?:preso|ingoiato) tutte le (?:pillole|compresse|medicine)|sovradosaggio"
    r"|sa mor|sinucid\w*|ma omor|sa traiesc|iau viata|(?:luat|inghitit) toate (?:pastilele|medicamentele)|supradoz\w*"
    r"|wil dood|zelfmoord|mezelf (?:doden|van kant maken)|niet meer leven"
    r"|me mato|dolor (?:de|en el) pecho|dor no peito|mal a la poitrine|douleur (?:a la poitrine|thoracique)|brustschmerz\w*|schmerzen in der brust|dolore al petto|durere (?:in|de) piept|pijn op de borst"
    r"|no puedo respirar|nao consigo respirar|(?:ne )?peux pas respirer|kann nicht (?:mehr )?atmen|non riesco a respirare|nu pot (?:sa )?respir"
    r")\b")
_READING_WORD=(r"(?:glucose|blood sugar|sugar|bg|bs|libre|cgm|dexcom|glucometer|meter|bp|blood pressure|cuff|systolic|diastolic|resting (?:heart rate|hr|pulse)|heart rate|pulse ?ox(?:imeter)?|(?:pulse )?oximeter|pulse|hr"
               r"|spo2|oxygen(?: saturation)?|o2(?: sats?)?|sats|(?:blood |urine )?ketones?|bhb|potassium|k\+|sodium|na\+?|thermometer|heart (?:is )?beating|temp(?:erature)?|fever)")
READING_AFTER=re.compile(r"(?<![\w.])(?P<a>\d+(?:\.\d+)?)(?:\s*(?:/|over|slash)\s*(?P<b>\d+(?:\.\d+)?))?\s*(?P<unit>mg/dl|mmol(?:/l)?|%|percent|bpm|times (?:a|per) minute|\u00b0 ?[cf]|degrees|celsius|fahrenheit)?"
    r"(?![\w%])(?:(?!\b"+_READING_WORD+r"(?![\w]))[^0-9]){0,30}?\b(?P<what>"+_READING_WORD+r")(?![\w])")
READING=re.compile(r"\b(?P<what>"+_READING_WORD+r")(?![\w])(?:(?!\b"+_READING_WORD+r"(?![\w]))[^0-9]){0,36}?(?<![\w.])(?P<a>\d+(?:\.\d+)?)(?:\s*(?:/|over|slash)\s*(?P<b>\d+(?:\.\d+)?))?\s*"
                   r"(?P<unit>mg/dl|mmol(?:/l)?|meq(?:/l)?|%|percent|bpm|times (?:a|per) minute|\u00b0 ?[cf]|degrees|celsius|fahrenheit|[cf](?![\w]))?"
                   # A stated reading, not a digit inside a word ("a1c") or a duration/count ("2 weeks").
                   r"(?![\w%])(?!\s*(?:days?|weeks?|months?|years?|hours?|minutes?|mins?|times?|readings?|results?|measurements?)\b)")
BP_PAIR=re.compile(r"(?<![\d/.:-])(\d{2,3})\s*(?:/|over|slash)\s*(\d{2,3})(?![\d/:]|\s*(?:mg|mmol|%|kg|km|steps))")
_NUMBER=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),range(20)))
_TEN=dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10)))
_ONES,_TENS='|'.join(list(_NUMBER)[:10]),'|'.join(_TEN)


def spelled_numbers(text):
    """Numbers written as several words next to a reading ("potassium six point four", "one eighty over one twenty", "two oh five") as digits; a single word stays."""
    teens='ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen'
    text=re.sub(rf'\b(one|two) oh ({_ONES})\b',lambda m:str(100*_NUMBER[m[1]]+_NUMBER[m[2]]),text)
    text=re.sub(rf'\b(one|two) ({teens})\b',lambda m:str(100*_NUMBER[m[1]]+_NUMBER[m[2]]),text)
    text=re.sub(rf'\b(one|two) ({_TENS})(?:[ -]({_ONES}))?\b',lambda m:str(100*_NUMBER[m[1]]+_TEN[m[2]]+(_NUMBER[m[3]] if m[3] else 0)),text)
    text=re.sub(rf'\b({_TENS})[ -]({_ONES})\b',lambda m:str(_TEN[m[1]]+_NUMBER[m[2]]),text)
    text=re.sub(rf'\b({_TENS}|{teens}) point ({_ONES})\b',lambda m:f'{_TEN[m[1]] if m[1] in _TEN else _NUMBER[m[1]]}.{_NUMBER[m[2]]}',text)
    return re.sub(rf'\b({_ONES}) point ({_ONES})\b',lambda m:f'{_NUMBER[m[1]]}.{_NUMBER[m[2]]}',text)


BPM=re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:bpm|beats? (?:a|per) minute|times (?:a|per) minute)\b")
EXERCISE=re.compile(r'\b(?:workouts?|runs?|running|exercise|training|max|maximum|peak|zones?|during)\b')
# The digits of a date, year or clock time are never a reading ("resting heart rate between 2026-09-01 and 2026-09-25").
# They are masked in place, so a reading next to a date is still found ("glucose on 2026-09-25: 45 mg/dl").
_DAY,_MONTH=r"(?:0?[1-9]|[12]\d|3[01])",r"(?:0?[1-9]|1[0-2])"
DATE_OR_TIME=re.compile(rf"\b(?:(?:19|20)\d\d([-/.]){_MONTH}(?:\1{_DAY})?|{_DAY}([-/.]){_MONTH}\2(?:19|20)?\d\d|{_MONTH}([-/.]){_DAY}\3(?:19|20)?\d\d"
                        r"|(?:19|20)\d\d(?!\s*(?:mg|mmol|meq|%|bpm))|(?:[01]?\d|2[0-3]):[0-5]\d)\b")


# History, family history or an explicit general/educational framing, with no sign the
# user is affected now. Present-tense cues always win, so recall is kept.
NOT_CURRENT=re.compile(r"\b(?:turned out to be|years? ago|months ago|in (?:19|20)\d\d|(?:grand)?(?:father|mother|dad|mom|pa|ma) had|family history|runs in (?:my|the) family"
    r"|general knowledge|for (?:my )?(?:exam|class|course|poster|homework)|studying|asking for (?:a friend|general)|what (?:are|is) the (?:warning )?signs? of|difference between)\b")
CURRENT=re.compile(r"\b(?:now|right now|currently|today|tonight|this (?:morning|afternoon|evening)|still|again|just|since|keeps?|i (?:am|'m|feel|have|can'?t)|i think i)\b")
# The exemption covers only the clause it governs (safety review S3): a clause in the first person ("i took all my pills", "my sugar is crashing")
# is never exempt, whatever frame ("for my class") the message carries elsewhere, unless it is dated history ("i fainted years ago").
_HISTORY=re.compile(r"\b(?:turned out to be|years? ago|months ago|in (?:19|20)\d\d|(?:grand)?(?:father|mother|dad|mom|pa|ma) had|family history|runs in (?:my|the) family)\b")
_CLAUSE_BREAK=re.compile(r"[.!?;:,\n]|\s[-\u2013\u2014]\s")
_SUBCLAUSE_BREAK=re.compile(r"[.!?;:,\n]|\s[-\u2013\u2014]\s|\b(?:and|but|or|so|because|while|although|then)\b")


def _not_current(text,match):
    """True if the acute match is exempt: the message has a history or general framing and no present-tense cue, and the match's clause is
    either dated history or has no first person (another person's "my dad" is not the user)."""
    if not NOT_CURRENT.search(text) or CURRENT.search(text):return False
    span=lambda pattern:text[max((b.end() for b in pattern.finditer(text,0,match.start())),default=0):
                             next((b.start() for b in pattern.finditer(text,match.end())),len(text))]
    if _HISTORY.search(span(_SUBCLAUSE_BREAK)):return True
    own=r"\b(?:i|i'?m|i'?ve|i'?d|i'?ll|me|myself|mine|my(?! (?:[\w-]+ )?"+_PERSON+r"\b))\b"
    return not re.search(own,NOT_CURRENT.sub(' ',span(_CLAUSE_BREAK)))


_WORDS=json.loads((ROOT/'metadata/foreign-function-words.v1.json').read_text())
FOREIGN_WORDS,ENGLISH_WORDS,NOT_ALONE=frozenset(_WORDS['foreign']),frozenset(_WORDS['english']),frozenset(_WORDS['not_alone'])


def foreign_language(text):
    """A sentence outside English (spec v1.8 10g.33) hands off the whole message: any letter of a non-Latin script (Greek letters stay:
    lab names such as "γ-GT"), at least two signs of another language that outnumber the English function words, or one sign and no
    English function word ("quero morrer.", "ich will sterben."; a sign that also reads as English, "mon-fri", "EST", counts only with another). A sign is a
    function word of another language (after dropping diacritics: "vreau sa vad", "și") or a word with Latin diacritics; one
    accented name ("Iași", "café") is not a foreign sentence."""
    if any(c.isalpha() and not c.isascii() and not unicodedata.name(c,'').startswith(('LATIN','GREEK','MICRO')) for c in text):return True
    if re.search(r'[\u0370-\u03ff\u1f00-\u1fff]{2,}',text):return True     # a Greek word, not a symbol ("γ-GT", "α-tocopherol", "µmol")
    for sentence in re.split(r'[.!?;\n]+',text.lower().replace('\u2019',"'")):
        words=re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*",sentence)
        plain=lambda w:unicodedata.normalize('NFKD',w).encode('ascii','ignore').decode()
        accent=lambda w:any(not c.isascii() and unicodedata.name(c,'').startswith('LATIN') for c in w)
        foreign=sum(accent(w) or any(plain(p) in FOREIGN_WORDS for p in {w,*w.split('-')}) for w in words)   # "azi-noapte": one word
        english=sum(w in ENGLISH_WORDS for w in words)
        if foreign>=2 and foreign>english:return True
        if not english and any(plain(p) in FOREIGN_WORDS-NOT_ALONE for w in words for p in {w,*w.split('-')}):return True   # "quero morrer."
    return False


def acute_or_crisis(text):
    """Acute symptoms, crisis language, dangerous stated readings or dose changes.

    Evaluated before any other rule. It only ever forces a handoff; it never
    enables a read, so recall is preferred over precision.
    """
    text=canonicalize(text)
    if ACUTE_FOREIGN.search(unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode()):return True
    if medication_change(text) or any(not _not_current(text,m) for pattern in (ACUTE,ACUTE_MORE) for m in pattern.finditer(text)):return True
    masked=spelled_numbers(DATE_OR_TIME.sub(lambda d:'#'*len(d[0]),text))
    for m in BP_PAIR.finditer(masked):      # "186 slash 104", "182 over 96 was what the cuff gave me": a pair of this size is a blood pressure
        top,bottom=int(m[1]),int(m[2])
        if 100<=top<=300 and 30<=bottom<top and (top>=180 or bottom>=120):return True
    if any(float(m[1])>=150 for m in BPM.finditer(masked)) and not EXERCISE.search(text):return True     # "watch says 154 bpm" while sitting
    forward=list(READING.finditer(masked))
    # A number after its reading word belongs to that word when it follows it closely ("sodium 139 potassium 4.3", "systolic averages 121, diastolic 79");
    # otherwise, or when a preposition ties it to the next word, it is read against both and any dangerous attribution flags the message
    # (safety review S4: "resting heart rate this month with 41.2 on the thermometer").
    close={f.span('a') for f in forward if re.fullmatch(r"\s*(?:[:=]\s*|[\w-]+\s+)?",masked[f.end('what'):f.start('a')])}
    tied=lambda m:re.search(r"\b(?:on|from|by|per|via|off|according to|using)\b",masked[max(m.end(g) for g in ('a','b','unit') if m[g]):m.start('what')])
    matches=forward+[m for m in READING_AFTER.finditer(masked) if m.span('a') not in close or tied(m)]
    for m in matches:
        what,a,b,unit=m['what'],float(m['a']),m['b'],m['unit']
        if what in ('glucose','blood sugar','sugar','bg','bs','libre','cgm','dexcom','glucometer','meter'):
            mmol=unit and unit.startswith('mmol') or (not unit and a<35)
            if (a<3.0 or a>16.7) if mmol else (a<54 or a>300):return True
        elif what in ('bp','blood pressure','cuff') and b is not None:
            if a>=180 or float(b)>=120:return True
        elif what in ('systolic','diastolic'):
            if a>=(180 if what=='systolic' else 120):return True
        elif what.startswith(('temp','fever','thermometer')):     # 40 C / 104 F and above, or below 32 C / 90 F; without a unit 30-45 is Celsius and 90+ is Fahrenheit
            if (unit[-1]=='f') if unit and unit!='degrees' else a>=90:
                if a>=104 or a<=90:return True
            elif a>=40 or 30<=a<=32:return True
        elif what.startswith(('spo2','oxygen','o2','sats','pulse ox','pulseox')) or what.endswith('oximeter'):
            if 50<=a<90:return True
        elif what.endswith(('ketone','ketones','bhb')):     # blood ketones from 3.0 mmol/L; urine strips in mg/dl from 80 ("large")
            if (a>=80) if unit and unit.startswith('mg') else a>=3.0:return True
        elif what in ('potassium','k+'):
            if a>=6.0 or a<=2.5:return True
        elif what in ('sodium','na','na+'):
            if a<120 or a>160:return True
        elif what.startswith(('resting','heart is','heart beating')) or not EXERCISE.search(text):
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


# Spec v1.8 10g.28 (with 10g.7, 10g.17 and the 10g.28 clarifications): the classes of content outside the read that hand
# off. The word lists are data (metadata/context-classes.v1.json); the grammar of a request, of a clinician's involvement,
# of another person and of another question is here.
CONTEXT=json.loads((ROOT/'metadata/context-classes.v1.json').read_bytes())
_words=lambda *names:r'\b(?:'+'|'.join(entry for name in names for entry in CONTEXT[name])+r')\b'
_any=lambda *patterns:re.compile('|'.join(patterns))
# Missed, stopped or changed medication always hands off, next to a plain read too (spec v1.8 10g.36.4, §2.4 class 1): stopping, skipping, missing,
# forgetting, running out of, doubling, halving, tapering or changing a dose or a drug. Supplements, vitamins and devices are not medication
# (not_medication); a supplement a clinician prescribed is medical context (the 'medical' list and the clinician grammar below).
_CHANGE,_MEDICATION,_STATUS,_NOT_MEDICATION=(re.compile(_words(name)) for name in ('medication_change','medication','medication_status','not_medication'))
_CLAUSE_END=re.compile(r"[.,;:!?]|\b(?:but|so|then|because|while|although)\b")


def _own_medication(text,med):
    """False when the medication word is a supplement or a device ("my vitamin D pills"): the words before it, back to a conjunction."""
    return not _NOT_MEDICATION.search(re.split(r'\b(?:and|or|plus)\b|[&,;.]',text[:med.start()])[-1]+med[0])


def medication_change(text):
    """A change of the user's own medication within one clause: a change verb with a medication word up to four words after it ("ran out of my
    blood pressure meds", "off my statin", "took double my dose"), or a status after the medication ("my refill ran out", "my dose was doubled")."""
    for verb in _CHANGE.finditer(text):
        tail=_CLAUSE_END.split(text[verb.end():],1)[0]
        for med in _MEDICATION.finditer(tail):
            if len(tail[:med.start()].split())>4:break
            if _own_medication(tail,med):return True
    for med in _MEDICATION.finditer(text):
        after=_CLAUSE_END.split(text[med.end():],1)[0]
        status=_STATUS.search(after)
        if status and len(after[:status.start()].split())<=3 and _own_medication(text,med):return True
    return False


def history_medication(text):
    """An earlier turn that states something about the user's own medication ("I gave my prescriptions a holiday this week", "haven't
    touched my BP pills since Monday"). In the current turn such a sentence hands off (10g.28 medical class; 10g.36.4 when it is a
    change); in an earlier turn only the models read it, and the crisis head does not score history. Questions ("what meds am I on")
    and other people's medication are not."""
    for sentence in re.split(r'(?<=[.!?])\s+',canonicalize(text)):
        if sentence.rstrip().endswith('?') or QUESTION.match(sentence) or PERSON.search(sentence) or not FIRST_PERSON.search(sentence):continue
        if any(_own_medication(sentence,med) for med in _MEDICATION.finditer(sentence)):return True
    return False


# A value asked for in a unit other than the stored one (spec v1.8 10g.36.3): "my weight in pounds", "convert it to mmol/L". CONTEXT['units'] maps a
# catalog unit to its spellings; a unit that is the native unit of a bound metric is a plain label ("fat mass in kg"), not a conversion.
UNIT_PHRASE=_any(_words('unit_phrases'))
_UNIT_NAMES=list(CONTEXT['units'])
_UNITS='|'.join('('+'|'.join(forms)+')' for forms in CONTEXT['units'].values())
UNIT_ASK=re.compile(r"\b(?:in|into|to|as|using)\s+(?:the |my |a |an )?(?:"+_UNITS+r")(?![\w/])")
UNIT_REJECT=re.compile(r"\b(?:instead of|rather than)\s+(?:the |my )?(?:"+_UNITS+r")(?![\w/])")     # "lbs instead of kg": the stored unit is what is refused


def research_restricted(text):
    """A literature question restricted with "only" ("studies on creatine only in adolescents"): the topic cannot be minimised without changing it
    (spec v1.8 10g.35.8, 10g.37); public_topic used to drop "only" and plan the unrestricted topic."""
    question=RESEARCH_QUESTION.fullmatch(text)
    parts=re.split(r'\b(?:research|studies|study|papers?|literature|evidence|science)\b',text,maxsplit=1)
    text=question['topic'] if question else parts[-1] if len(parts)==2 else ''
    public = re.sub(r'\b(zone|omega|type|stage|phase|grade|class|glp|covid|il) (\d{1,2}s?)\b',r'\1-\2',text)
    return bool(text and (re.search(r'\b(?:only|solely|exclusively|adults? over|aged?|published (?:after|before|since)|since \d)\b',text) or re.search(r'(?<![\w-])\d',public)))


def unit_conversion(request):
    """True if the request asks for a value in a unit other than the stored one; the direct template renders the stored unit."""
    text=canonicalize(request.state.current_request)
    if UNIT_PHRASE.search(text) or UNIT_REJECT.search(text):return True
    asked={_UNIT_NAMES[m.lastindex-1] for m in UNIT_ASK.finditer(text)}
    if not asked:return False
    native={(metric_definition(v) or {}).get('unit') for _,_,values in entities(enhance(text),request.available_metrics) for k,v in values if k=='metric'}
    return bool(asked-native)


PLAIN=_any(_words('pleasantry','not_ask'))
# Over-the-counter supplements and vitamins are plain context (spec v1.8 10g.36.4, 10g.37): the phrase is removed before the medical class is matched.
_SUPPLEMENT=re.compile(_words('not_medication')+r"(?: [a-e]\d*)?(?: (?:pills?|tablets?|capsules?|gummies|powder|shake|drink|supplements?))?")
WINDOW=_any(_words('window_words'))
# Periods the rule date parser leaves unresolved but the learned parser reads ("last night", "this morning", "on Tuesday", "3 days
# ago"): a sentence that states one states a date all the same (sentence_roles), whichever parser reads it.
TIME_PHRASE=_any(_words('time_phrases'))
# Windows relative to another window ("the week before", "the previous day", "2 weeks earlier", "the same period last month"; spec v1.8
# 10g.38 items 1, 5, c, 10g.25): windows for a comparison or a continuation of the read (window_spans), never dates for the sentence roles, so
# "What did I eat the day before?" stays another question (class 8, 10g.6). Anchored on an event ("the week before my birthday") a relative
# window is unresolved (10g.29 analogue), and "the same week" alone is the read's own window: neither is one.
RELATIVE_WINDOW=_any(_words('relative_windows'))
# Style, format and language instructions ("answer in Romanian", "put it in a table", "keep it short"): the direct template
# cannot honour them, so they hand off as a text transformation when the plan is a single read; a multi-read plan composes
# (spec v1.8 10g.28 class 3 [refined 10g.38]). Unit-system instructions ("imperial only", 10g.36.3) and creative rewrites, which may
# carry another request ("write me a haiku about autumn"), hand off in every plan.
STYLE=_any(_words('style'))
STYLE_ALWAYS=_any(_words('style_always'))
# A comparison of the read across windows in a side sentence ("vs last month", "how does that compare with the week before?") and
# the words that state a current value as the read's window ("current", "right now", "latest"): spec v1.8 10g.38.
COMPARED=_any(_words('comparison_window'))
POINT_IN_TIME=_any(_words('point_in_time'))
COMPARISON_RELATION=_any(_words('comparison_read'))
# The wording a factual comparative between two windows may carry besides its names and windows ("has my ... got worse ... than", "I wonder
# if", "tell me whether"): window words first, so a phrase of either list is removed whole (spec v1.8 10g.38 items 6, a, d).
COMPARISON_FRAME=_any(_words('window_words','comparison_frame'))
# A correction or replacement ("No, the week before.", "Sorry, I mean last month.", "the week before instead"): the new window replaces
# the read's, so the plan reads one window, never an added one.
CORRECTION=_any(_words('correction'))
_PERSON='(?:'+'|'.join(CONTEXT['other_person'])+')'
PERSON=_any(_words('other_person'))
_CLINICIAN='(?:'+'|'.join(CONTEXT['clinician'])+')'
# 2. A clinician's involvement is medical context: they ask, tell, want or will review; "for my appointment"; seeing them.
_INVOLVED=(r"(?:\b(?:my|the|our|a|his|her) (?:[\w-]+ )?)?\b"+_CLINICIAN+r"(?:'s)? (?:(?:has|had|is|was|will|would|just|also|keeps?|kept|has been|is going to) )*"
    r"(?:ask\w*|told|tells?|telling|wants?|wanted|said|says|saying|suggest\w*|recommend\w*|advis\w*|order\w*|prescribed|referred|mentioned|needs?|needed|request\w*|nag\w*"
    r"|review\w*|check\w*|look\w*|go(?:es|ing)? over|went over|flagged|noticed|thinks|thought|worried|concerned|monitor\w*|track\w*|put me|took me|started me|switched me|(?:had|has|got|set|made) me|likes?|loves?|hopes?|expects?|insists?|prefers?|(?:is |are )?(?:having|taking) a (?:nosy|look)|sign(?:s|ed|ing)? off|discharg\w*)\b"
    r"|\bfor (?:my|the|a|an|our) (?:[\w'-]+ )?(?:appointment|consultation|check-?up|visit|follow-?up|review|call|session)s?\b|\b(?:so|as|because|cos|since|before|until|ready to show|to show|show(?:ing)?(?: it| them)? to|ready for) (?:my|the|our|a) (?:[\w-]+ )?"+_CLINICIAN+r"\b|\b(?:my|the|our) (?:gp practice|health cent(?:er|re)|clinic|hospital|surgery|practice|care team)(?:'s office)? (?:requested|asked|wants|would like|ordered|is expecting|is after|needs|called|phoned|sent|expects|would appreciate)\b|\b(?:for|to|with|by|from|per) (?:my|the|our) (?:[\w-]+ )?"+_CLINICIAN+r"\b"
    r"|\b(?:see|saw|seen|seeing|visit\w*|went to|go to|going to|appointment with|booked in with|called|phoned|emailed|messaged) (?:my|the|a|an|our) (?:[\w-]+ )?"+_CLINICIAN+r"\b"
    r"|\b"+_CLINICIAN+r"'s (?:orders?|advice|instructions?|recommendations?|notes?)\b")
# In an extra sentence any clinician counts, but not the user's own job ("I'm a nurse").
_ANY_CLINICIAN=r"(?<!i'm a )(?<!i am a )(?<!as a )(?<!i'm an )(?<!i am an )(?<!as an )\b"+_CLINICIAN+r"\b"
# 2. Someone else reporting the user's symptom ("my wife says I snore a lot").
_REPORTED=(r"\b"+_PERSON+r" (?:says|said|thinks|thought|noticed|notices|complains|complained|reckons|swears|mentioned|told me|tells me|keeps saying) (?:that )?"
    r"(?:i|i'm|i've|i was|i am|my)\b.*?"+_words('symptom','symptom_or_metric'))
# 6. Another person only when the read or ask involves their data or situation: a possessive before their data (a bound name,
# blanked in the read sentence) or at the end ("my wife's HRV", "and my son's?"), "for my mom", comparing with them, a question
# about them, or "what about my wife?". A context-only mention ("my son's team made the final") is plain context.
_DATA=r"(?:data|numbers|stats|figures|statistics|readings?|results?|levels?|values?|scores?|metrics|labs?|trends?|sleep|steps|weight|vitals|heart rate|hrv|bp|blood pressure|cholesterol|glucose|age|health|week)"
_OTHER=(r"\b"+_PERSON+r"(?:'s|s')(?! (?:health|medical|family) history)(?: (?:latest|last|recent|current|average|avg|daily|resting|own))?(?:(?=\s{2,})|(?=\s*[,;:!?])|\s*(?:too|as well|also)?\W*$|(?: \w+){0,2} "+_DATA+r"\b)"
    r"|\bfor (?:my|our|the) (?:[\w-]+ )?"+_PERSON+r"\b|\bhow (?:is |are |was |were |did |does |do |has |have )?(?:my|our|the|his|her) (?:[\w-]+ )?"+_PERSON+r"\b"
    r"|\b(?:do|check|show|pull up|include|track|run|add) (?:it |that )?(?:for )?(?:him|her|hers|them)\s+(?:too|as well|also)\b|\b(?:him|her|them) (?:too|as well)\b"
    r"|\b(?:my|our) "+_PERSON+r"(?: \w+){0,2} "+_DATA+r" (?:too|as well|also|please|aswell|plz)\b|\b(?:my|our) "+_PERSON+r"\s{2,}(?:too|as well|also|please|aswell|plz)\b"
    r"|\b(?:do|check|show|pull up|include|track|run|add) (?:the |my |our )?(?:entire|whole|full|rest of the|all of (?:the |my |our )?|all (?:the |my |our )?|every|each) ?(?:[\w-]+ )?"+_PERSON+r"\b"
    r"|\b(?:run it across|across|include|cover|do|check|give me|show|pull up|track|for)\b[^.?!]{0,25}\b(?:all|every|each|whole|entire|everyone|everybody)\b[^.?!]{0,30}\b(?:clients?|team|players?|kids|class|office|group|department|staff|students|athletes|members|employees|family|household|squad|roster|people|patients?)\b|\b(?:everyone|everybody) in (?:my|our|the)\b"
    r"|\b(?:what|how|how much) (?:kind of \w+ )?(?:my|our|the) (?:[\w-]+ )?"+_PERSON+r" (?:scored|got|had|did|slept|ran|walked|weighs?|sleeps?|has had|have been up to|is doing|are doing|has been)\b"
    r"|\btheirs\b|\bthe pair of them\b|\b(?:for|to) the other (?:two|three|four|one|ones)\b|\binclude (?:them|him|her)\b|\b(?:asked|wants|wanted|told|needs|needed) me to (?:pull|check|look|get|show|find|run|do)\b"
    r"|\bbelong(?:s)? to (?:my|a|an|our|the) (?:[\w-]+ )?"+_PERSON+r"\b|\bfor (?:a |an )?(?:friend|colleague|client|patient|relative|neighbou?r|coworker)(?: of mine)?\b"
    r"|\b(?:wife|hubby|nan|nana|mum|mom|dad|son|daughter|sister|brother|partner|girlfriend|boyfriend|bf|gf)s\s+(?:\w+ ){0,2}\s*(?:too|aswell|as well|also|2|please|plz)\b"
    r"|\b(?:compar\w*|vs\.?|versus|than|same as|similar to|against) (?:\w+ ){0,3}?(?:(?:with|to|against) )?(?:my|our|the|his|her) "+_PERSON+r"\b"
    r"|\b(?:did|does|do|is|was|were|are|has|have|had|can|could|will|would|should) (?:my|our|the) "+_PERSON+r"\b|\b(?:his|her) "+_DATA+r"\b|\b(?:his|her)(?=\s{2,})"
    r"|^(?:(?:and|also|or|plus|what about|how about|same for|now|then)\W+)+(?:for )?(?:(?:my|our|the) "+_PERSON+r"|him|her|hers|his|them)\b|^(?:for )?(?:my|our) "+_PERSON+r"\W*$"
    r"|\b(?:someone|somebody|anyone|anybody|everyone|everybody) else'?s?\b|\bhers\b|\bon behalf of\b|\basking for (?:my|our|the) "+_PERSON+r"\b"
    r"|(?<![\w'])(?!(?:it|that|what|there|here|who|he|she|how|let|where|this|when|why|one|today|yesterday|tomorrow|tonight)'s\b)[a-z]+'s (?:too|as well|also)\b")
# 7. An action, write or external request in request form; a past-tense statement ("I uploaded my PDF") is context.
_REQUEST=(r"(?:^|[,;:] )(?:(?:ok|okay|so|and|also|then|now|please|pls|just|btw|anyway|oh|hey|hi|um|uh)[, ]+)*(?:(?:can|could|would|will) (?:you|u) (?:please |also |just )?"
    r"|i (?:want|need|would like|'d like|wanna) (?:you )?to |go ahead and |remember to |don'?t forget to |make sure (?:to |you ))?")
# A write joined to the read by "and" is a request too when it takes an object ("... next to last week and email it to me", "and
# export them as a PDF"); "pre and post workout" or "add them up" is not one.
_JOINED=r"\band (?:then |also |please |just )*(?:"+'|'.join(CONTEXT['write_verbs'])+r") (?:it|them|that|this|those|these|everything|all|the|my|a|an|me|last|today|yesterday|tomorrow)\b(?! up\b)"
_action=lambda verbs:_REQUEST+r'(?:'+'|'.join(CONTEXT[verbs])+r')\b|'+_JOINED+'|'+_words('action_phrases')
# The pattern each class is checked with in an extra sentence (nothing bound), in read wording or small talk, and in the
# sentence that carries the read; None: not checked there. Only in an extra sentence do a filter word such as "only", an
# address to the assistant, another question, and words that may also name a metric, a profile field or a lab group count.
CONTEXT_CLASSES={
    'symptom':(_any(_words('symptom','symptom_or_metric'),_REPORTED),_any(_words('symptom'),_REPORTED),_any(_words('symptom'),_REPORTED)),         # 2 (10g.17)
    'medical':(_any(_words('medical','medical_generic','care'),_ANY_CLINICIAN,_INVOLVED),_any(_words('medical','medical_generic'),_INVOLVED),
               _any(_words('medical'),_INVOLVED)),                                                                                             # 2
    'injection':(_any(_words('injection')),)*3,                                                                                                # 3 (10g.7)
    'instruction':(_any(_words('instruction','addressing')),_any(_words('instruction')),_any(_words('instruction_read'))),                    # 3
    'filter':(_any(_words('filter','filter_extra')),_any(_words('filter')),_any(_words('filter_read'))),                                      # 4
    'comparison':(_any(_words('comparison')),_any(_words('comparison')),_any(_words('comparison_read','comparison_change'))),                 # 5
    'other_person':(_any(_OTHER),)*3,                                                                                                          # 6
    'action':(_any(_action('action_verbs')),_any(_action('write_verbs')),_any(_action('write_verbs'))),                                       # 7
    'judgement':(_any(_words('judgement')),_any(_words('judgement')),_any(_words('judgement_read'))),                                          # 9
    'prediction':(_any(_words('prediction')),_any(_words('prediction')),_any(_words('prediction','prediction_read'))),                        # a forecast is not a read of recorded data
}
ROLES=('extra','wording','read')
# The forecast patterns of the read sentence run on the sentence without its bound names (a blank stands for the metric), so the cheap check is a loose cue.
PREDICTION_CUE=re.compile(r"\b(?:going to|gonna|will|wil|'ll|would|should|might|could|predict\w*|pred[a-z]{1,3}t\w*|fore?cast\w*|estimat\w*|project\w*|expect\w*|anticipat\w*|outlook|trajectory|guess|likely|from now|by (?:next|the end|january|february|march|april|may|june|july|august|september|october|november|december)|long run|plateau|down the line)\b")
# 8. Another question or request in a sentence with nothing bound (10g.6, all or nothing).
QUESTION=re.compile(r"(?:(?:ok|okay|so|and|also|then|now|hey|hi|btw|anyway|well|oh|but|um|uh|yo|random q|random question|off[- ]topic|unrelated|quick q|quick one|random one)[, ]+)*"
    r"(?:(?:what|wat|wot|when|where|why|how|who|whom|whose|which)(?:'s|s|'re|'d)\b|(?:what|wat|wot|when|where|why|how|who|whom|whose|which)\b(?! a\b| an\b)"
    r"|(?:is|are|am|was|were|do|does|did|can|could|would|will|should|shall|has|have|had)(?:n'?t)? (?:i|you|u|we|they|he|she|it|there|that|this|these|those)\b"   # only before a subject
    r"|(?:please |pls )?(?:gimme|tell|show|give|list|pull|get|fetch|check|look|find|display|bring|send|compare|calculate|explain|let me|work out|name|spell|translate)\b"
    r"|i (?:want|need|would like|'d like|wanna) (?:you|to know|to see|the|my|a|an|some|all|every|more|it|that|this|them)\b)")
# A second question after a conjunction ("unrelated but how do i reset my router"); relative clauses ("which is") are not.
QUESTION_CLAUSE=re.compile(r"(?:\b(?:but|and|also|btw|anyway|plus)|[,;:-])\s+(?:how|what|where|why) (?:do|does|did|can|could|should|would|will) (?:i|you|we|they)\b"
    # an indirect question ("no idea what the error code means"), and a change of subject with content after it ("on a completely different subject, my dishwasher ...")
    r"|\b(?:no idea|don'?t know|do not know|not sure|wondering|wonder|can'?t (?:tell|figure out)|unsure|arguing|debating)(?: \w+){0,3} (?:about )?(?:what|how|why|where|when|who|which|whether|if)\b|\b(?:zero|no) clue (?:how|what|why|where|when|who|which)\b|\b(?:unrelated|off (?:on )?a tangent|random)\b[^.?!]{0,15}\bbut\b"
    r"|(?:on )?(?:a )?(?:completely |totally |entirely )?(?:different|unrelated|separate|random) (?:subject|topic|note|matter)\s*[,:-]\s*\w+ \w+ \w+")
RESEARCH=re.compile(r"\b(?:research\w*|stud(?:y|ies)|trials?|evidence|literature|papers?|articles?|published|pubmed|scien(?:ce|tific)|meta-analys\w*|rcts?|what'?s known|what is known)\b")
FIRST_PERSON=re.compile(r"\b(?:i|i'?m|i'?ve|i'?d|i'?ll|me|my|mine|myself)\b")
PERSONAL=re.compile(r"\b(?:my|mine|myself|i'?m|i am|i'?ve|i have|i had|i was|i take|i took|i'?d)\b")


def sentence_roles(request):
    """(sentence, role) for each sentence of the current request.

    'read': it carries the read (a bound metric, record, profile field or research wording, or a date when no other sentence
    carries the read), or it is the only sentence; 'window': only a date, next to a sentence that carries the read ("Last week
    please.", "How does that compare with last year?"); 'wording': read wording or small talk ("just the latest reading is
    fine", "thanks!"); 'extra': nothing bound.
    """
    # Clauses joined by ";" or dashes are handled by the clause binder; only whole sentences are checked here.
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',request.state.current_request) if re.search(r'[A-Za-z]',s)]
    if len(sentences)<2:
        yield from ((sentence,'read') for sentence in sentences);return
    texts=[canonicalize(sentence) for sentence in sentences]
    # Read instructions and profile fields belong to the read ("just the latest reading is fine", "show my allergies").
    bound=[bool(entities(enhance(sentence),request.available_metrics) or re.search(r'\b(?:research|studies|literature|papers?|evidence)\b',text)
                or any(re.search(r'\b'+re.escape(alias)+r'\b',text) for alias in FIELDS)) for sentence,text in zip(sentences,texts)]
    for i,(sentence,text) in enumerate(zip(sentences,texts)):
        if bound[i]:yield sentence,'read'
        elif temporal(text,request.state.reference_date)[1] or TIME_PHRASE.search(text):yield sentence,('window' if any(bound[:i]+bound[i+1:]) else 'read')
        # Emoji and other symbols carry no request content.
        elif SMALL_TALK.fullmatch(re.sub(r'[^\w\s\',.!?-]','',text).strip()) or READ_WORDS.search(text):yield sentence,'wording'
        else:yield sentence,'extra'


def _outside_bound(sentence,text,request):
    """The read sentence without its bound metric, record and profile-field names ("fasting insulin") and without a literature
    question's public topic (10g.8) unless the user brings in their own situation there. Bound names are blanked, not removed."""
    enhanced=enhance(sentence)
    subject=PLAIN.sub(' ',erase(enhanced,[*entities(enhanced,request.available_metrics),*(m.span() for alias in FIELDS for m in re.finditer(r'\b'+re.escape(alias)+r'\b',enhanced))]))
    research=RESEARCH.search(subject)
    return (subject[:research.start()] if PERSONAL.search(text) else '') if research else subject


def window_spans(text,request,points=False):
    """The windows a sentence states (spec v1.8 10g.38), as spans: resolved periods, the periods the learned parser reads (time_phrases) and
    windows relative to another (relative_windows: "the week before"), adjacent pieces of one window merged ("the same period last month");
    with points, a current value too ("current", "right now", "latest": the read's window in 10g.38 item 7). The date parser reads the text
    without its relative windows, whose "before" it would take as an open bound ("last week and the week before")."""
    relative=[m.span() for m in RELATIVE_WINDOW.finditer(text)]
    spans=sorted([*temporal(erase(text,relative),request.state.reference_date)[1],*(m.span() for m in TIME_PHRASE.finditer(text)),
                  *relative,*((m.span() for m in POINT_IN_TIME.finditer(text)) if points else ())])
    merged=[]
    for start,end in spans:
        if merged and not text[merged[-1][1]:start].strip():merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def compared_windows(texts,request,points=False):
    """The distinct windows the texts state that a comparison can be across; an open window ("since last week") is not one: a change
    since then is a verdict (10g.38 item 6). Two texts naming the same window ("last week") name one window, and every current value
    ("latest", "right now") is one: now."""
    return {'now' if POINT_IN_TIME.fullmatch(text[a:b]) else text[a:b] for text in texts for a,b in window_spans(text,request,points)
            if not text[a:b].startswith('since')}


plain_window=lambda t:not re.search(r'[a-z0-9]',WINDOW.sub(' ',t))   # nothing but window words


def continuation(sentence,role,request):
    """How a side sentence continues the read, or None: 'window' for a plain window clause ("And last
    week?", "Same for last month."), 'windows' for a window relative to the read's or a comparison of the read across windows, whose plan
    reads each window, 'style' for a style or format instruction alone.

    A plain window clause ("And last week?", "Same for last month.", "Also show the week before.") carries nothing but its windows and
    window words: a clause-bound continuation, one read per window (spec v1.8 10g.38 item 5, 10g.25). So does a comparison of the read
    across windows ("vs last month", "how does that compare with the week before?", "was it higher than last week?", "put them side by
    side") when the read and this sentence state two windows, none of them open ("since"); against one window the reference is implicit,
    a verdict, and it stays class 5 (10g.38 item 6). The comparison words are found before the windows are erased, so a change over a window
    of its own ("has it improved this month?", "did it improve the week before?") is a verdict. A relative window does not make a sentence a
    window one: anything else next to it ("What did I eat the day before?") is checked as in an extra sentence (class 8). A sentence that is
    only a style or format instruction ("Put it in a table.") is class 3, which the style guard decides on the plan ([refined 10g.38]).
    A correction ("No, the week before.", "Sorry, I mean last month.", "Rather the week before.") replaces the read's window: never a
    second window, so at most the plain window clause it was before 10g.38."""
    text=PLAIN.sub(' ',canonicalize(sentence)).strip(' ,;:-')
    dated=erase(text,window_spans(text,request));corrected=CORRECTION.search(canonicalize(sentence))
    if not corrected and RELATIVE_WINDOW.search(text) and plain_window(dated):return 'windows'
    if role=='window' and plain_window(dated):return 'window'
    if STYLE.search(dated) and plain_window(STYLE.sub(' ',dated)):return 'style'
    if corrected:return None
    compared=[m.span() for m in COMPARED.finditer(text)]
    if compared and not re.search(r'\bsince\b',text) and plain_window(POINT_IN_TIME.sub(' ',erase(dated,compared))) and len(
            compared_windows([text,*(canonicalize(s) for s,r in sentence_roles(request) if r=='read')],request,points=True))>=2:return 'windows'
    return None


def sentence_classes(sentence,role,request):
    """The 10g.28 classes in one sentence (see CONTEXT_CLASSES); empty for plain context, which is ignored for planning."""
    text=PLAIN.sub(' ',canonicalize(sentence)).strip(' ,;:-')
    if not re.search(r'[a-z0-9]',text):return []   # only pleasantries ("How are you?")
    own=False
    if role in ('window','extra') and continuation(sentence,role,request):
        return ['acute'] if acute_or_crisis(sentence) else []
    if role=='window':
        # Any other sentence with a date is checked as an extra one ("Also, what's the weather tomorrow?"),
        # but a question about the user's own data may be the read itself, under a name the index lacks ("how was my
        # homa-ir in march?"), so it is not "another question".
        role,own='extra',bool(FIRST_PERSON.search(text))
    column=ROLES.index(role);subject=None
    found=['acute'] if acute_or_crisis(sentence) else []
    for name,patterns in CONTEXT_CLASSES.items():
        pattern=patterns[column]
        if pattern is None:continue
        if role=='read' and name not in ('injection','action'):
            # Matched outside the bound names and a literature question's public topic; the cheap check on the whole sentence comes first.
            if not (PERSON if name=='other_person' else PREDICTION_CUE if name=='prediction' else pattern).search(text):continue
            subject=_outside_bound(sentence,text,request) if subject is None else subject
            # A change or comparative verdict between two windows the sentence states ("has my sleep got worse this week than last week?",
            # "tell me if my HRV dropped from last week to this week") is a factual comparative, one read per window (10g.38 item 6, a);
            # with one window, an open one or an event ("since I started running") it stays class 5, and relationships always are. So does one
            # with anything but the wording of the comparison (comparison_frame) outside its names and windows, anywhere in the sentence, which
            # only the model may interpret: a reference ("worse than usual", "below my baseline", "than Anna's", "than for other runners"), a norm
            # ("my ideal weight", "normal for my age"), a manner ("worryingly"), a reason, a purpose or a judgement ("... vs last week is normal",
            # "... because I'm overtraining", "... so I know if the diet is working"; 10g.38 hand-off list, classes 5, 6, 9).
            # The comparison is of a metric the sentence names: made of frame words alone, a self-state or symptom ("have I got worse this
            # week than last week?", "am I getting better this week vs last week?") has no metric to read and stays class 5.
            if name=='comparison' and len(compared_windows([text],request))>=2 and not re.search(
                    r'\b(?:since|after|before|following)\b',erase(text,window_spans(text,request))) and not re.search(
                    r'[a-z0-9]',COMPARISON_FRAME.sub(' ',erase(subject,window_spans(subject,request)))) and any(
                    kind!='record' for _,_,values in entities(enhance(sentence),request.available_metrics) for kind,_ in values):pattern=COMPARISON_RELATION
            if pattern.search(_SUPPLEMENT.sub(' ',subject) if name=='medical' else subject):found.append(name)
        elif pattern.search(_SUPPLEMENT.sub(' ',text) if name=='medical' else text):found.append(name)
    if role=='extra' and not own and (re.sub(r'[^\w?]+$','',sentence).endswith('?') or QUESTION.match(text) or QUESTION_CLAUSE.search(text)):found.append('question')
    return found


def unexplained_sentence(request):
    """True if the request carries content outside its read that only the model may interpret (spec v1.8 10g.28).

    Plain context (lifestyle, training, travel, app use, device changes, pleasantries, curiosity) is not: it is ignored for
    planning, and the parser still sees the whole message.
    """
    return any(sentence_classes(sentence,role,request) for sentence,role in sentence_roles(request))


def explicit_overview(text):
    return bool(re.search(r"\b(?:analy[sz]e (?:me|my (?:health|data|everything))|(?:full|complete|whole) (?:health )?(?:summary|overview|report|analysis|picture of my health)"
                          r"|everything|all (?:of )?my (?:health )?data|health (?:overview|summary|analysis|assessment|report|rundown)|overall health"
                          r"|how am i doing|how'?s my health|assess (?:my|me)|evaluate my health)\b",canonicalize(text)))


TREND_WORDS=re.compile(r'\b(?:trends?|history|over|since|evolved|changed|changing|been|progress|graph|chart|average|avg|across|all|every|values|readings|results|levels|numbers|measurements)\b')
def single_value_question(text,request):
    """Latest unless the request asks for a series (spec §3.9, §3.10d).

    Any stated window (including today, spec v1.7 §3 item 9), a trend word, or
    show/list wording keeps the trend; "what's my X", a bare "X" or "X now" is one value.
    Metric names are removed first, so "daily average heart rate" is not "average".
    """
    text=canonicalize(text)
    rest=erase(text,[(a,b) for a,b,_ in entities(text,request.available_metrics)])
    # Scope instructions ("sleep only, no labs") restrict a default read; they are not a value question.
    if re.search(r'\b(?:show|pull up|display|graph|chart|plot|list|only|just|no)\b',rest) or TREND_WORDS.search(rest):
        return False
    _,spans,error=temporal(text,request.state.reference_date)
    if error:return False
    return all(re.fullmatch(r'(?:right )?now',text[start:end]) for start,end in spans)


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
    paths=[ROOT/p for p in ('query_plan.py','query_selector.py','query_execution.py',
        'metadata/reason-codes.v1.json','metadata/context-classes.v1.json')]
    if learned_parser_enabled():
        # The selector pin must change when the served decoder or its catalog
        # changes; model/threshold identity is separately exposed in /health.
        paths += [ROOT/'learned_parser'/p for p in ('parse.py','decode.py','verify.py')]
        paths += sorted((ROOT/'metadata/learned').glob('*.json'))
    mode=b'learned' if learned_parser_enabled() else b'rules'
    return hashlib.sha256(base_identity().encode()+mode+b''.join(p.read_bytes() for p in paths)+INTENT_SHA256.encode()).hexdigest()


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
    r"(?:research|studies|papers|trials|evidence|literature) (?:on|about|for|regarding) "
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
    text=re.sub(r"^(\W*how (?:did i|was my|'d i) sleep)\W*$",r'\1 last night',text)   # no window = the most recent night (spec v1.8 10g.13)
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
    text=re.sub(r'(?<!max )(?<!maximum )(?<!peak )(?<!min )(?<!minimum )\bactivity '+hr+r'(?: (?:avg|average|mean))?\b','activity hr average',text)
    text=re.sub(r'\bactivity hr average (max|maximum|peak)\b','activity hr max',text)
    text=re.sub(r'\bactivity hr average (min|minimum|lowest)\b','activity hr min',text)
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


# ---- learned catalog-driven parser (OPEN_JEV_PARSER=learned; model dir LEARNED_PARSER_DIR) ----
# The hard guards above (acute/crisis, read restrictions, transformations, unexplained sentences) always run first,
# and every learned plan passes the same QueryPlan contract, inventory and budget checks as the rule path.
_LEARNED=None
RECORD_BASIS={'profile':'current_snapshot','labs':'exam_date','workouts':'started_at','calendar':'next_due_date'}


def learned_parser_enabled():
    return os.environ.get('OPEN_JEV_PARSER')=='learned'


class _Agreement:
    """Two independently trained parsers: plan only when both return the identical plan (else handoff), and only when the
    weaker confidence clears LEARNED_PARSER_MIN_CONFIDENCE (cut-off chosen on the dev set, exp/final_select.py). With
    LEARNED_VERIFIER_DIR, a plan checker (cross-encoder over request + plan) must also score the health reads at least
    LEARNED_VERIFIER_MIN (threshold chosen on the dev set; 0 = no veto). Below the cut-off (tier 2 of the accept rule,
    exp/fixes/calibration/ACCEPT.md) a plan is still accepted when both models' decision confidence (the confidence of the
    decisions that reach the plan) and the checker clear the cut-off; without a checker tier 2 never accepts."""
    def __init__(self, parsers,cutoff,verifier=None,verifier_min=0.0):
        self.parsers=parsers;self.cutoff=cutoff;self.verifier=verifier;self.verifier_min=verifier_min
    def select(self,req):
        results=[p.select(req) for p in self.parsers]
        acute=next((r for r in results if 'acute_or_crisis_requires_model' in r.get('reason_codes', [])),None)
        if acute is not None:return acute
        first=results[0]
        if first['status']!='planned':return first
        confidence=min(r.get('confidence',1.0) for r in results)
        def accepted(tier,checker=None):
            return {**first,'diagnostics':{'acceptance_tier':tier,
                'confidence_bucket':int(confidence*10)/10,
                **({'checker_bucket':int(checker*10)/10} if checker is not None else {})}}
        key=lambda r:json.dumps(sorted(json.dumps(q,sort_keys=True) for q in r['queries']))
        if any(r['status']!='planned' or key(r)!=key(first) for r in results[1:]):
            return {'status':'handoff','reason_codes':['learned_models_disagree'],'queries':[]}
        health=[q for q in first['queries'] if q.get('kind')=='health']
        if min(r.get('confidence',1.0) for r in results)<self.cutoff:
            decision=[r.get('decision_confidence') for r in results]
            if self.verifier is None or not health or None in decision or min(decision)<self.cutoff:
                return {'status':'handoff','reason_codes':['low_joint_confidence'],'queries':[]}
            score=self.verifier.score(req,health)
            if score<self.cutoff:
                return {'status':'handoff','reason_codes':['learned_plan_unverified'],'queries':[]}
            return accepted('tier2',score)
        if self.verifier is not None and self.verifier_min>0 and health and self.verifier.score(req,health)<self.verifier_min:
            return {'status':'handoff','reason_codes':['learned_plan_unverified'],'queries':[]}
        return accepted('tier1')


def learned_parser():
    global _LEARNED
    if _LEARNED is None:
        from pathlib import Path
        from learned_parser.decode import Parser
        dirs=[d for d in os.environ['LEARNED_PARSER_DIR'].split(',') if d.strip()]
        verifier=None
        if os.environ.get('LEARNED_VERIFIER_DIR'):
            from learned_parser.verify import Verifier
            verifier=Verifier(Path(os.environ['LEARNED_VERIFIER_DIR']))
        _LEARNED=_Agreement([Parser(Path(d.strip())) for d in dirs],float(os.environ.get('LEARNED_PARSER_MIN_CONFIDENCE','0')),
                            verifier,float(os.environ.get('LEARNED_VERIFIER_MIN','0')))
    return _LEARNED


def learned_parser_identity():
    """SHA-256 over the learned model files and its catalog data, for /health attestation."""
    from pathlib import Path
    roots=[Path(d.strip()) for d in os.environ['LEARNED_PARSER_DIR'].split(',') if d.strip()];data=Path(__file__).resolve().parent/'metadata/learned'
    digest=hashlib.sha256();digest.update(os.environ.get('LEARNED_PARSER_MIN_CONFIDENCE','0').encode())
    checker=[Path(os.environ['LEARNED_VERIFIER_DIR'])] if os.environ.get('LEARNED_VERIFIER_DIR') else []
    digest.update(os.environ.get('LEARNED_VERIFIER_MIN','0').encode() if checker else b'')
    cut=os.environ.get('LEARNED_PARSER_MIN_CONFIDENCE','0')     # with a checker the accept rule has tier 2 at the cut-off (_Agreement)
    digest.update(f'tier2:decision_confidence>={cut},checker>={cut}'.encode() if checker else b'')
    for f in [*(root/n for root in roots for n in ('config.json','model.safetensors','heads.pt','items.json','crisis.json','calibration.json','tokenizer.json')),
              # the checker head's file name carries its text version from v3 on (learned_parser/verify.py head_file)
              *(f for root in checker for f in (root/'config.json',root/'model.safetensors',*sorted(root.glob('verifier_head*.pt')),root/'verifier.json',root/'tokenizer.json')),
              *sorted(data.glob('*.json'))]:
        digest.update(f.name.encode());digest.update(f.read_bytes())
    return digest.hexdigest()


def public_topic(topic):
    # "zone 2" -> "zone-2": a number after a naming word is part of the name; any other number is a value ("ldl 160") and is dropped
    topic=re.sub(r'\b(zone|omega|type|stage|phase|grade|class|glp|covid|il) (\d{1,2}s?)\b',r'\1-\2',topic.lower())
    # Contract normalization may change punctuation, never erase a constraint or
    # an unrepresentable name. Reject the whole topic rather than truncate it.
    if re.search(r'(?<![\w-])\d',topic):return ''
    words=re.sub(r"[^a-z0-9+' -]",' ',topic).split()
    if TOPIC_EXCLUDED & set(words) or not 1<=len(words)<=12:return ''
    if any(not w[0].isalpha() for w in words):return ''
    return ' '.join(words)



def learned_queries(request,diagnostics):
    result=learned_parser().select({'state':request.state.model_dump(),'available_metrics':list(request.available_metrics),
        'available_record_types':list(request.available_record_types),'literature_available':request.literature_available})
    diagnostics['method']='learned_parser'
    diagnostics.update(result.get('diagnostics',{}))
    if result['status']!='planned':raise ValueError(result['reason_codes'][0])
    queries=[]
    for q in result['queries']:
        if q.get('kind')=='research':
            topic=public_topic(q['topic'])
            if not topic:raise ValueError('unbound_research_topic')
            # A tagged span can omit an untagged qualifier. For complete,
            # explicit literature questions, require every subject word to
            # survive; a shorter public topic must not silently change the ask.
            text=canonicalize(request.state.current_request)
            context=RESEARCH_CONTEXT.fullmatch(text)
            question=RESEARCH_QUESTION.fullmatch(context['rest'] if context else text)
            parts=re.split(r'\b(?:research|studies|study|papers?|literature|evidence|science)\b',context['rest'] if context else text,maxsplit=1)
            subject=question['topic'] if question else parts[-1] if len(parts)==2 else ''
            if re.search(r"\b(?:i|me|my|mine|myself|we|us|our|ours|not|no|without|except|excluding|exclude)\b",subject):
                raise ValueError('unbound_research_topic')
            if question:
                subject=re.sub(r'[, ]+(?:please|pls|plz|ty|ta|cheers|thanks(?: a lot| in advance| very much)?|thank you(?: very much)?)[.!?]*$', '', subject)
                subject=re.sub(r'^(?:the )?effects? of ', '', subject)
                subject=public_topic(re.sub(r'^(?:an?|the) ', '', subject))
                if not subject or not (set(subject.split())-{'a','an','the'}) <= set(topic.split()):
                    raise ValueError('unbound_research_topic')
            queries.append(ResearchRead(topic=topic));continue
        if q['records']:
            record=q['records'][0]
            period={'kind':'all_history'} if record=='profile' else q['period']
            # calendar reads carry their own basis (due/done wording, spec v1.8 10g.19); other records use the record's basis
            basis=q.get('date_basis') if record=='calendar' and q.get('date_basis') else RECORD_BASIS[record]
            queries.append(HealthRead(records=[record],operation='latest',period=period,date_basis=basis,
                                      profile_fields=q.get('profile_fields') or []))
        else:
            queries.append(HealthRead(metrics=q['metrics'],operation=q['operation'],period=q['period'],source=q.get('source'),
                                      date_basis=q.get('date_basis','observed_at')))
    # A record quantity is part of the request, not a suggestion to the model.
    count=re.search(r'\b(?:latest|newest|most recent|last)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?=(?:lab |blood |health )?(?:reports|workouts|appointments|plans)\b)',canonicalize(request.state.current_request))
    if not count and request.state.recent_user_requests and not entities(enhance(request.state.current_request),request.available_metrics):
        previous=canonicalize(request.state.recent_user_requests[-1])
        count=re.search(r'\b(?:latest|newest|most recent|last)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?=(?:lab |blood |health )?(?:reports|workouts|appointments|plans)\b)',previous)
        if count:
            # Resolve the prior plan through the same path, then verify the new
            # request retained its record subject before carrying the count.
            prior_request=request.model_copy(update={'state':request.state.model_copy(update={'current_request':request.state.recent_user_requests[-1],'recent_user_requests':request.state.recent_user_requests[:-1]})})
            prior=learned_queries(prior_request,{})
            if len(prior)!=1 or len(queries)!=1 or not isinstance(prior[0],HealthRead) or not isinstance(queries[0],HealthRead) or prior[0].records!=queries[0].records:
                raise ValueError('unbound_request_constraint')
    if count:
        from proposal_binding import NUMBER_WORDS
        limit=int(count[1]) if count[1].isdigit() else {'zero':0,**NUMBER_WORDS}[count[1]]
        if not 1<=limit<=200:raise ValueError('record_limit_out_of_range')
        if len(queries)!=1 or not isinstance(queries[0],HealthRead) or not queries[0].records or queries[0].records==['profile']:
            raise ValueError('unbound_request_constraint')
        queries[0]=HealthRead.model_validate({**queries[0].model_dump(),'limit':limit})
    return queries


def read_window(query,request):
    """A read's window as (first day, last day) on the request's calendar: a calendar period to date, a relative one ending today (N days,
    or N months back to the day after), all history ending today."""
    period=query.period;today=date.fromisoformat(request.state.reference_date)
    if period['kind']=='between':return date.fromisoformat(period['start_at'][:10]),date.fromisoformat(period['end_at'][:10])
    if period['kind']=='calendar':return {'day':today,'week':today-timedelta(days=today.weekday()),'month':today.replace(day=1)}[period['period']],today
    if period['kind']=='relative' and period['unit']=='days':return today-timedelta(days=period['amount']-1),today
    if period['kind']=='relative':
        month=today.month-1-period['amount'];year=today.year+month//12;month=month%12+1
        return date(year,month,min(today.day,monthrange(year,month)[1]))+timedelta(days=1),today
    return date.min,today


def windows_unread(queries,request):
    """True if the plan does not read a comparison across windows as spec v1.8 10g.38 requires: one read per window, over windows that do
    not overlap. A single read where a side sentence continues a read across windows ("Show my steps last week. Also show the week before.",
    "Versus last month?") has dropped a window. Two trend reads of a shared metric and source (or one of them of every source) over
    overlapping or identical windows compare a window with part of itself ("this month vs the last 30 days", "past 14 days vs past 2 weeks");
    part vs whole is a comparison only against one day or the latest value (follow-up f). After a correction ("No, the week before.",
    "Sorry, I mean last month.", "this week instead of last week") the new window replaces the old one: two reads of a metric add a window."""
    if len(queries)<2:
        # Only a sentence with a relative window or comparison words can continue the read across windows: the cheap check comes first.
        if not any(RELATIVE_WINDOW.search(t) or COMPARED.search(t) for s in re.split(r'(?<=[.!?])\s+',request.state.current_request)
                   for t in [PLAIN.sub(' ',canonicalize(s.strip())).strip(' ,;:-')]):return False
        roles=list(sentence_roles(request))
        return any(r=='read' for _,r in roles) and any(r in ('window','extra') and continuation(s,r,request)=='windows'
                                                       for s,r in roles)
    metrics=[set(q.metrics) for q in queries if isinstance(q,HealthRead)]
    # The cue corrects a window only in a sentence that states one ("I take the stairs instead of the lift" is context).
    if CORRECTION.search(canonicalize(request.state.current_request)) and any(a&b for i,a in enumerate(metrics) for b in metrics[i+1:]) and any(
            CORRECTION.search(t) and window_spans(t,request) for t in (canonicalize(s) for s in re.split(r'(?<=[.!?])\s+',request.state.current_request))):return True
    spans=[(set(q.metrics),q.source,*read_window(q,request)) for q in queries if isinstance(q,HealthRead) and q.metrics and q.operation=='trend']
    spans=[span for span in spans if span[2]!=span[3]]
    return any(a[0]&b[0] and (a[1]==b[1] or not (a[1] and b[1])) and a[2]<=b[3] and b[2]<=a[3] for i,a in enumerate(spans) for b in spans[i+1:])


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
            if any(history_medication(s) for s in request.state.recent_user_requests):
                raise ValueError('unbound_request_constraint')
            if any(foreign_language(s) for s in [request.state.current_request,*request.state.recent_user_requests]):
                raise ValueError('language_outside_evaluated_contract')
            if any(read_restricted(s) for s in [request.state.current_request,*request.state.recent_user_requests]):
                raise ValueError('read_restriction_requires_native_context')
            # Text transformations, unit-system instructions and creative rewrites need the native conversation in every plan.
            if re.search(r'\b(?:translate|translation|rephrase|rewrite|paraphrase)\b',canonicalize(request.state.current_request)) or STYLE_ALWAYS.search(
                    canonicalize(request.state.current_request)):
                raise ValueError('text_transformation_requires_native_context')
            # Style, format or language instructions ("answer in Romanian", "put it in a table", "keep it short"; spec v1.8 10g.28 class 3):
            # the direct template cannot honour them, so a single-read plan hands off; a multi-read plan always composes and keeps the plan
            # ("side by side", "in a table"; [refined 10g.38]). A request that states fewer than two windows hands off here, before any
            # parser; one that states two is decided on the plan, and a handoff for any other reason keeps its own code.
            style=bool(STYLE.search(canonicalize(request.state.current_request)))
            if style and len(compared_windows([canonicalize(request.state.current_request)],request,points=True))<2:
                raise ValueError('text_transformation_requires_native_context')
            # A unit conversion ("my weight in pounds", "convert it to mg/dL"; spec v1.8 10g.36.3) has its own published code.
            if unit_conversion(request):
                raise ValueError('unit_conversion')
            if research_restricted(canonicalize(request.state.current_request)):
                raise ValueError('unbound_research_topic')
            # Content outside the read in a protected class (spec v1.8 10g.28: "i don't see the point anymore. what's my
            # sleep score", a symptom or a clinician next to the read, another person's data, a write request) is content
            # only the model may interpret, whatever the encoder decides. Plain context is not.
            if unexplained_sentence(request):
                raise ValueError('unbound_request_constraint')
            if learned_parser_enabled():
                text=canonicalize(request.state.current_request)
                if any(re.search(r'\b(?:morning|afternoon|evening|night)\b.*\b(?:between|from)\s+\d{1,2}\s+(?:and|to|until)\s+\d{1,2}\b|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b',part) and entities(enhance(part),request.available_metrics) for part in re.split(r'[.!?;]\s+',text)):
                    raise ValueError('clock_time_not_supported')
                if re.search(r'\b(?:upload(?:ed)? (?:date|time|order)|by upload|import(?:ed)? (?:date|time))\b',text):
                    raise ValueError('upload_order_not_available')
                if any(re.search(r'\b(?:reports?|records?|labs?)\b.{0,30}\b(?:uploaded|imported)\b|\b(?:uploaded|imported)\b.{0,30}\b(?:reports?|records?|labs?)\b',part) for part in re.split(r'[.!?;]\s+',text)):
                    raise ValueError('upload_order_not_available')
                if re.search(r'\b(?:very first|first ever|earliest|oldest)\b.{0,35}\b(?:result|reading|measurement|report)\b',text):
                    raise ValueError('unbound_request_constraint')
                if re.search(r'\b(?:latest|newest|most recent|last)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b.{0,35}\b(?:readings|results|measurements|values)\b',text):
                    raise ValueError('unbound_request_constraint')
                if 'profile' in text and re.search(r'\b(?:hide|except|excluding|omit|without|leave out|but not)\b',text):
                    raise ValueError('unbound_request_constraint')
                if re.search(r'\b(?:health (?:centre|center)|clinic|hospital) appointment\b|\b(?:man|woman|person|patient) i (?:care for|look after)\b',text):
                    raise ValueError('unbound_request_constraint')
                # Fail closed: any other error inside the learned path (a date overflow, a decoder bug) is a published handoff, never HTTP 500.
                try:queries=learned_queries(request,diagnostics)
                except (ValueError,KeyError):raise
                except Exception:raise ValueError('query_contract_unrepresentable') from None
                if style and len(queries)<2:raise ValueError('text_transformation_requires_native_context')
                if windows_unread(queries,request):raise ValueError('unbound_request_constraint')
                result=QueryPlan(status='planned',queries=queries,time_zone=request.state.time_zone,
                    selector_sha256=identity(),adapter_sha256=INTENT_SHA256,model_revision=MODEL_REVISION,
                    request_sha256=request_identity(request),diagnostics=diagnostics)
                validate_inventory(result,request)
                if sum(query_operation_count(q) for q in queries)>8:
                    raise ValueError('operation_budget_exceeded')
                return result.model_dump(mode='json')
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
            if style and len(queries)<2:raise ValueError('text_transformation_requires_native_context')
            if windows_unread(queries,request):raise ValueError('unbound_request_constraint')
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
        # Bare "calories" is calories_burned (spec v1.6 §3.10c); calories eaten are not a recorded metric.
        if re.search(r'\b(?:calories|kcal)\b',text) and re.search(r'\b(?:eat|eaten|ate|intake|consumed?|food|meals?|diet)\b',text):
            raise ValueError('unresolved_metric_or_record')
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
            # The night basis exists only for these metrics (query_plan.py HealthRead, Vita query_tool); any other
            # metric "last night" is a one-day window on the reference (wake-up) day, observed_at (spec v1.7 10f item 5).
            night=all(v in NIGHT_METRICS for _,_,vs in entities(text,request.available_metrics) for kind,v in vs if kind=='metric')
            text=re.sub(r'\blast night\b','today',text)
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
        # One analyte under several inventory ids reads all of them in one read (spec v1.8 10g.1, index data).
        groups=INDEX.get('analyte_groups',{})
        spelling={m:set(g) for g in groups.get('spelling',{}).values() for m in g}
        unit_split={g[0]:set(g) for g in groups.get('unit_split',{}).values()}
        sampling_split={g[0]:set(g) for g in groups.get('sampling_split',{}).values()}   # bare glucose reads fasting too (10g.27)
        percent=re.search(r'%|percent',original.lower())
        metrics=sorted({x for m in metrics for x in ({m}|spelling.get(m,set())|(set() if percent else unit_split.get(m,set()))|sampling_split.get(m,set()))
                        if x==m or x in request.available_metrics})
        operation=proposal['operation']
        if metrics:
            if limit:raise ValueError('metric_latest_n_not_supported')
            # Observations cannot exist in the future; future calendar records are handled below.
            if period.get('kind')=='between' and str(period['start_at'])[:10]>request.state.reference_date:
                raise ValueError('future_period_unavailable')
            if period.get('kind')=='between' and str(period['end_at'])[:10]>request.state.reference_date:
                period={**period,'end_at':request.state.reference_date}
            # "What's my X?", a bare "X" or "X today" asks for one value (spec §3.9). Applied
            # after acceptance, so it never changes which requests are accepted.
            so_far=bool(re.search(r'\bso far\b',original)) and proposal['period']['period_kind']=='unstated'
            if so_far:   # today for daily metrics (registry fresh_days <= 14), all history for sparse ones (spec v1.8 10g.14)
                operation='trend'
                period=({'kind':'calendar','period':'day'} if all((metric_definition(m) or {}).get('fresh_days',365)<=14 for m in metrics)
                        else {'kind':'all_history'})
            if (operation=='trend' and proposal['coverage']!='broad' and not night and not proposal['records'] and not so_far
                    and single_value_question(original,request)):
                operation='latest'
                if proposal['period']['period_kind']=='unstated':period={'kind':'all_history'}
            # An unstated trend window is explicit: frequent metrics (catalog fresh_days <= 14) read 30 days,
            # sparse ones (labs, body composition) all history (spec v1.7 10f item 4).
            frequent={m for m in metrics if (metric_definition(m) or {}).get('fresh_days',365)<=14}
            if proposal['period']['period_kind']=='unstated' and operation=='trend' and set(metrics)<=frequent and not so_far:
                period={'kind':'relative','amount':30,'unit':'days'}
            if proposal['coverage']=='broad' and proposal['period']['period_kind']=='unstated':
                for group,window in ((set(metrics)-frequent,{'kind':'all_history'}),
                                     (set(metrics)&frequent,{'kind':'relative','amount':30,'unit':'days'})):
                    if group:output.append(HealthRead(metrics=sorted(group),operation=operation,period=window,source=source))
            else:
                # One night or one named day: NIGHT_METRICS read on the sleep_end_day basis (one explicit day), every other
                # metric on a one-day observed_at window of the same day (spec v1.8 10g.4); longer windows stay one read.
                day=None
                if period.get('kind')=='between' and str(period['start_at'])==str(period['end_at']) and len(str(period['start_at']))==10:day=str(period['start_at'])
                elif period.get('kind')=='calendar' and period.get('period')=='day':
                    ref=date.fromisoformat(request.state.reference_date);day=str(ref-timedelta(days=1) if period.get('offset')=='previous' else ref)
                nightly=sorted(set(metrics)&set(NIGHT_METRICS)) if day else []
                rest=sorted(set(metrics)-set(nightly))
                if nightly:output.append(HealthRead(metrics=nightly,operation=operation,period={'kind':'between','start_at':day,'end_at':day},
                    source=source,date_basis='sleep_end_day'))
                if rest:output.append(HealthRead(metrics=rest,operation=operation,period=period,source=source,date_basis='observed_at'))
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
