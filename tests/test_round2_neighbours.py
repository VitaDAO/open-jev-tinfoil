"""Round 2 neighbour variants: fresh phrasings of each phrase the production path served (same construct, other words and positions), as a side sentence and
as an in-clause modifier. Each was a gap before round 2 and hands off now; the plain look-alikes still reach the parser."""
import pytest

from test_round2_guards import outcome, selector  # noqa: F401  (fixtures and helper)

HAND_OFF = [
    # gap-0018
    "My urine dipstick turned that deep maroon colour, the one the tube label calls large. What's my HRV last week?",
    "The little test stick I dunk for ketones is on the very top colour of the scale, the one the leaflet marks as large. What's my latest LDL?",
    "The ketone paper is now the deepest purple on the bottle's colour scale, right in large territory. Show my resting heart rate this month.",
    "What's my HRV last week because my dipstick hit the maroon block labelled large.",
    "What's my weight cos my keto pee stick went dark purple, large on the box.",
    # gap-0026
    'The machine at the chemist printed out 192 over 118 after squeezing my arm. Show my resting heart rate this month.',
    "I got 340 on the glucose meter. What's my HRV last week?",
    "The lab portal shows six point four for potassium. What's my HRV last week?",
    "What's my latest LDL with the fair's cuff saying one eighty over one twenty.",
    "What's my weight with the thermometer saying forty point five degrees.",
    # gap-0031
    "I quit that thing for my thyroid. What's my HRV last week?",
    "Honestly can't be bothered swallowing my evening ones anymore. What's my latest LDL?",
    'I stopped bothering with the heart one. Show my resting heart rate this month.',
    "What's my HRV last week as I've quit that thing for my thyroid.",
    'Show my resting heart rate this month as I stopped bothering with the heart one.',
    # gap-0039
    "My bloodwork came back a bit off, so I'm keeping an eye on my A1c. What's my weight?",
    'labs came back kinda rough tbh, ferritin especially. Show my resting heart rate this month.',
    "To be honest, my bloodwork didn't come back the way I was hoping, and triglycerides in particular have been playing on my mind. What's my HRV last week?",
    'Show my resting heart rate this month cos my labs came back kinda rough, ferritin especially.',
    "What's my HRV last week as my bloodwork didn't come back well and triglycerides bother me.",
    # gap-0102
    "Plot that for me. What's my latest LDL?",
    'Visualise this for me, numbers make my eyes glaze over. Show my resting heart rate this month.',
    "Line plot for that one, thanks. What's my weight?",
    "What's my latest LDL and plot that out for me.",
    "What's my latest LDL and sketch me a bar graph of this.",
    # gap-0118
    'Only the readings from between five and six in the morning. Show my resting heart rate this month.',
    "I'd like it limited to the stretch of the day when I'm normally still in bed, meaning anything prior to about half past six. Show my resting heart rate this month.",
    "Pre-dawn hours only. What's my weight?",
    "What's my latest LDL for daytime hours only.",
    "What's my HRV last week only for whatever's aftr 9pm.",
    # gap-0123
    "If it's not too much bother, take the two wedding nights out of the calculation altogether. What's my latest LDL?",
    "dont incldue the lazy sofa days ok. What's my weight?",
    'pls excl. the flight days from it. Show my resting heart rate this month.',
    'Show my resting heart rate this month w/o the late shifts.',
    'Show my resting heart rate this month excl. the flight days.',
    # gap-0125
    "Start the clock on the day I moved into the new flat. What's my weight?",
    'just use data since i got the new watch, nothing older pls. Show my resting heart rate this month.',
    'just do everything since i moved out yo. Show my resting heart rate this month.',
    "What's my weight starting from the day I moved flats.",
    "What's my HRV last week starting from the morning we brought the puppy home.",
    # gap-0136
    "Keep everything within the stretch that begins when the renovation finished and ends when the school term starts. What's my HRV last week?",
    "just the bit from when we got back from berlin til end of next month. What's my latest LDL?",
    "gimme the timeframe that kicks off when the builders left n ends when the lease is up. What's my latest LDL?",
    "What's my HRV last week within the stretch from when the renovation finished until the term starts.",
    "What's my weight for the stretch after the ski week to quarter end.",
    # gap-0147
    "as of jan 1 trust the scale not the app estimates ok. What's my weight?",
    "I'd like the lab-drawn numbers treated as the authority from New Year's Day forward, and the home test kits taken with a grain of salt. Show my resting heart rate this month.",
    "Starting June 3rd, the wrist device is the one I rely on. What's my latest LDL?",
    'Show my resting heart rate this month going only by the chest strap from the 15th on.',
    "Show my resting heart rate this month with lab results as the authority since New Year's Day.",
    # gap-0178
    "I'm curious whether my resting heart rate has come down since I began cycling to work. What's my latest LDL?",
    "wonder if my sleep got worse since we got the dog lol. What's my HRV last week?",
    "Ever since I overhauled my routine, with the strength training and the earlier bedtimes, I've been meaning to find out whether any of it actually moved the needle. What's my latest LDL?",
    "What's my latest LDL and whether it dropped since I began cycling to work.",
    "What's my weight and see if it improved after the holidays.",
    # gap-0181
    "I suspect the blue light from my phone is what's been dragging everything down. What's my weight?",
    "I keep thinking the new commute is behind the low readings. What's my latest LDL?",
    "For some time now I have had a nagging feeling that the change in my eating pattern, the late dinners especially, might be what lies behind the way the line has been drifting. What's my weight?",
    "What's my weight and whether my phone use at night is behind it.",
    'Show my resting heart rate this month and whether afternoon espresso influences the dips.',
    # gap-0187
    "Ever since the baby arrived in June I've been curious how the chart has changed. Show my resting heart rate this month.",
    "Would love a look at the stretch either side of when we relocated to Denver. What's my weight?",
    'Since I swapped to night shifts at the bakery in March, I want to see what that has done to my numbers. Show my resting heart rate this month.',
    "What's my HRV last week comparing the months before and after my new job.",
    "What's my HRV last week set against the years before I retired.",
    # gap-0194
    "Apparently getting to bed before eleven boosts your readiness score, and I'd love to find out whether that holds for me. Show my resting heart rate this month.",
    "I'm keen to see if it holds true for me, given the claim that cold showers shift resting heart rate. What's my weight?",
    'ppl say screen time before bed messes with recovery, wanna see if thats legit for me lol. Show my resting heart rate this month.',
    "What's my weight so I can check on myself whether cold showers shift it.",
    'Show my resting heart rate this month to find out first-hand if morning sun improves it.',
    # gap-0198
    "I have a hunch my sleep score has something to do with how late I have dinner. What's my latest LDL?",
    'Maybe my readiness depends on how many steps I did the day before. Show my resting heart rate this month.',
    "Wouldn't surprise me if my step totals rise and fall with how early I get up. Show my resting heart rate this month.",
    "Show my resting heart rate this month given it might be tied to the previous day's activity.",
    'Show my resting heart rate this month in case it rises and falls with when I get up.',
    # gap-0216
    "And let me know what my partner scored this week. What's my HRV last week?",
    "Long story short, I also need to know what the twins have been up to lately, because I've no idea. Show my resting heart rate this month.",
    "What's my weight and how my brother got on.",
    'Show my resting heart rate this month and how the kids are doing.',
    # gap-0218
    "Run that again for the other two. What's my latest LDL?",
    "Do the same for the pair of them, please. What's my weight?",
    "Include them in this as well. What's my latest LDL?",
    # gap-0223
    'gimme my sons sleep as well pls. Show my resting heart rate this month.',
    "and wifes weight aswell plz. What's my weight?",
    "hubbys resting hr too if u can. What's my HRV last week?",
    'Show my resting heart rate this month and my sons sleep aswell.',
    "What's my weight plus wifes weight aswell.",
    # gap-0229
    "I'm doing this for a friend of mine who asked me to look into it. What's my HRV last week?",
    "My girlfriend asked me to pull all of this up for her since she can't get into the app. Show my resting heart rate this month.",
    "Quick heads up: I'm the one typing, but the readings belong to my dad. What's my weight?",
    "What's my HRV last week as a favour for a friend of mine.",
    'Show my resting heart rate this month since my girlfriend asked me to look for her.',
    # gap-0239
    "Run it across all my clients, there are about thirty of them. What's my HRV last week?",
    "Give me every one of my players' sleep scores. What's my weight?",
    'Same request, but for everybody in our office. Show my resting heart rate this month.',
    "What's my HRV last week across all my clients.",
    'Show my resting heart rate this month for everybody in our office.',
    # gap-0311
    "quick one, who painted the mona lisa again. What's my HRV last week?",
    'wheres a decent place for a weekend away near lisbon. Show my resting heart rate this month.',
    "gimme the python one liner for reversing a list. What's my HRV last week?",
    "What's my latest LDL and any gift ideas for a coworker leaving.",
    # gap-0329
    "Off topic, but the washing machine at my flat smells awful and I'm not sure how to clean it. Show my resting heart rate this month.",
    "Anyway, unrelated, but I have zero clue how to change a flat tyre and my dad is away. What's my HRV last week?",
    "Sorry to go way off on a tangent but the garage door remote at my parents' house stopped working and nobody here knows how to reprogram it. Show my resting heart rate this month.",
    # prediction
    "What’s my A1c going to come out at in March? What's my latest LDL?",
    "What's the outlook for my LDL over the coming year? What's my latest LDL?",
    "I reckon my HRV will be 60 by June, am I right? What's my latest LDL?",
    "What's my latest LDL and what it is going to be by March.",
    "What's my latest LDL n forcast the next 3 months.",
]

PLAIN = [
    "I'm on a keto diet and it suits my routine. What's my latest LDL?",
    'The slip from the walk-in kiosk shows 118 slash 76 for my BP. Show my resting heart rate this month.',
    "The kiosk slip at the station says platform 12 slash 14 for my train. What's my weight?",
    "I take a multivitamin each morning and it's just part of my routine. Show my resting heart rate this month.",
    "The cholesterol-free recipes in my new cookbook are great, so I'm cooking them all week. What's my HRV last week?",
    "My running club posts the season's race chart on the noticeboard every month. What's my latest LDL?",
    "I'm a visual person so I've been enjoying the new watch face with its big colourful rings. What's my HRV last week?",
    "My flight boards at 7am tomorrow so I'll be up early. What's my HRV last week?",
    'I got the new ring recently and I love the titanium finish. Show my resting heart rate this month.',
    "The trip to Lisbon was fantastic, the pastries alone were worth it. What's my weight?",
    "The window in my office looks out over the river, which is nice. What's my weight?",
    "My cousin's wedding was on the 1st of last month and it was a great party. What's my weight?",
    "I started my evening walks around the lake and the sunsets are gorgeous. What's my HRV last week?",
    "After I started the new job I found a lovely lunch spot nearby. What's my latest LDL?",
    "The reason I'm up so late tonight is a family movie marathon. What's my latest LDL?",
    "Since we moved house in spring, the neighbours have been really welcoming. What's my latest LDL?",
    "Everyone on our street seems to be repainting their front doors this spring, which is quite a trend. What's my HRV last week?",
    "I've got my driving test on Thursday, so I'll be up early. Show my resting heart rate this month.",
    "I always eat pasta the night before a race, it's a habit from my school days. What's my latest LDL?",
    "My partner scored twice in the five-a-side game last night. What's my latest LDL?",
    "I did the Snowdon ridge last summer and I'd like to do the Lake District too. What's my latest LDL?",
    "my sons school trip is on friday so mornings are chaos. What's my weight?",
    "my boyfriends running club meets at six on tuesdays, so we're up early. Show my resting heart rate this month.",
    'All of this baking is for the school fete on Saturday, so the kitchen is a mess. Show my resting heart rate this month.',
    'I coach the under-12 squad on Tuesday evenings, so my week is fairly full. Show my resting heart rate this month.',
    "We did the entire coastal path in three days last summer. What's my weight?",
    'grey drizzly weather over here again, typical. Show my resting heart rate this month.',
    "By the way, thanks for all the help so far. What's my weight?",
    "will you show my HRV for last week? What's my latest LDL?",
    "could you check my LDL from my last blood test. What's my HRV last week?",
    "can you tell me my weight from this morning. What's my latest LDL?",
    "could you possibly fetch my latest HbA1c result. What's my HRV last week?",
    "would you kindly list my blood pressure readings from Tuesday. What's my latest LDL?",
    "would you be able to show me what my weight was on 3 March. What's my HRV last week?",
    "my HRV was going up all of last week. What's my HRV last week?",
    "going by my last reading I just want the raw numbers. What's my latest LDL?",
    "the trend going back six months please. What's my HRV last week?",
    "it will be a busy week. What's my weight?",
    "we're going to be late for the train. What's my weight?",
]


# A metric quantity cannot be represented by the v2 metric read contract.
HAND_OFF.append("Would you please look up my last three ferritin results? What's my latest LDL?")


@pytest.mark.parametrize('text', HAND_OFF)
def test_neighbour_variants_hand_off_before_any_parser(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'handoff', text


@pytest.mark.parametrize('text', PLAIN)
def test_neighbour_look_alikes_still_reach_the_parser(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'planned', text
