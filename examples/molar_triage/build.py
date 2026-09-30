# ruff: noqa: E501
"""Molar Triage: a small, hand-written dental-practice inbox for the calibrate-on-your-own-data walkthrough.

Every message is fictional and was written for this example; there are no real patients in
it. Each message yields three labeled rows (a noul, a choice, a score), split into a
calibration file and a test file by alternating messages.

    uv run python examples/molar_triage/build.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEPARTMENTS = {
    "hygiene": "Cleanings, checkups, whitening, sensitivity advice",
    "restorative": "Fillings, crowns, chipped or broken teeth, lost fillings",
    "orthodontics": "Braces, aligners, retainers, wires",
    "surgery": "Extractions, wisdom teeth, implants, post-operative care",
    "billing": "Insurance, invoices, payment plans, estimates",
    "front_desk": "Scheduling, forms, hours, directions, records requests",
}
URGENCY = [
    "Routine: can wait for the next regular visit",
    "Soon: book within a few weeks",
    "This week: schedule in the next few days",
    "Today: needs same-day attention",
    "Emergency: go to the practice or an emergency room now",
]

QUESTIONS = {
    "in_pain": {"type": "noul", "instructions": "The sender says they are in pain right now."},
    "department": {"type": "choice", "instructions": "Which department should take this message?",
                   "criteria": DEPARTMENTS},
    "urgency": {"type": "score", "instructions": "How urgently does this message need attention?",
                "criteria": URGENCY},
}

# text, in_pain, department, urgency level index
MESSAGES: list[tuple[str, bool, str, int]] = [
    ("Hi, I'd like to book my six-month cleaning sometime next month, mornings preferred.", False, "hygiene", 0),
    ("My crown came off while eating caramel. No pain, but the tooth underneath feels weird.", False, "restorative", 2),
    ("I knocked out my front tooth playing hockey twenty minutes ago and it's bleeding a lot.", True, "surgery", 4),
    ("Can you send me a copy of my last invoice for my HSA? I need it by Friday.", False, "billing", 0),
    ("The wire on my braces is poking my cheek and it's really sore.", True, "orthodontics", 3),
    ("What are your hours on Saturdays? I want to bring my kids in.", False, "front_desk", 0),
    ("I had a wisdom tooth out three days ago and now there's a horrible taste and throbbing pain.", True, "surgery", 3),
    ("My teeth are sensitive to cold lately. Not painful, just annoying. Any toothpaste you recommend?", False, "hygiene", 1),
    ("Does my insurance cover a night guard? The estimate said $450.", False, "billing", 0),
    ("I lost my retainer on vacation. How fast can I get a new one before my teeth shift?", False, "orthodontics", 2),
    ("A filling fell out this morning and the tooth is aching whenever I drink anything.", True, "restorative", 3),
    ("Need to reschedule Tuesday's appointment, something came up at work.", False, "front_desk", 1),
    ("There's a swelling on my gum above a molar and it hurts to bite. It's getting bigger since yesterday.", True, "surgery", 3),
    ("My son's aligner tray cracked. He has plenty of trays left, should he just move to the next one?", False, "orthodontics", 1),
    ("I chipped a back tooth on a popcorn kernel. Doesn't hurt but it's sharp against my tongue.", False, "restorative", 2),
    ("Are you accepting new patients? My family just moved to the area.", False, "front_desk", 0),
    ("Can I set up a payment plan for the implant? The full amount up front isn't possible for me.", False, "billing", 1),
    ("I'd like to ask about whitening before my wedding in October.", False, "hygiene", 1),
    ("My face is swollen on one side and I can barely open my mouth. Fever since last night.", True, "surgery", 4),
    ("You charged me twice for last week's visit. Please refund the duplicate.", False, "billing", 1),
    ("Just confirming: is it normal for the extraction site to still bleed a little after two hours?", False, "surgery", 3),
    ("My retainer feels tight after skipping a few nights. Should I keep wearing it?", False, "orthodontics", 1),
    ("I get a sharp pain in a top molar every time I chew on that side. Started about a week ago.", True, "restorative", 2),
    ("Please update my address and phone number on file.", False, "front_desk", 0),
    ("My daughter is due for a checkup and sealants before school starts.", False, "hygiene", 1),
    ("My bracket popped off but nothing is poking me. Next adjustment is in three weeks.", False, "orthodontics", 1),
    ("Toothache kept me up all night, over-the-counter painkillers aren't touching it.", True, "restorative", 3),
    ("Does the practice validate parking?", False, "front_desk", 0),
    ("I received a bill for a procedure I never had. Invoice number 88213.", False, "billing", 1),
    ("The temporary crown feels loose and rough. Not painful. Permanent one is due in ten days.", False, "restorative", 2),
    ("My gums bleed every time I floss. Is that something to worry about?", False, "hygiene", 1),
    ("I fell off my bike and two front teeth are loose and pushed backward. There's blood.", True, "surgery", 4),
    ("Can I get the pre-authorization form for my orthodontic coverage?", False, "billing", 0),
    ("Elastics for my braces ran out. Can I pick some up at the front desk?", False, "orthodontics", 1),
    ("I'm having an implant consult next week. Is there anything I should bring?", False, "surgery", 0),
    ("A sharp piece of my molar broke off and the edge is cutting my tongue with every word.", True, "restorative", 3),
    ("Do you have any evening appointments for a routine exam?", False, "front_desk", 0),
    ("My jaw clicks and aches when I wake up. Could I be grinding?", True, "hygiene", 1),
    ("Pain after the root canal was expected, but it's day five and getting worse, with a bump on the gum.", True, "restorative", 3),
    ("The aligner scan appointment: do I need to stop eating beforehand?", False, "orthodontics", 0),
    ("I swallowed a small piece of my broken retainer. No pain, just worried.", False, "orthodontics", 2),
    ("My insurance changed in January. Sending the new card, please update before my next visit.", False, "billing", 0),
    ("Bleeding won't stop from where the tooth was pulled this morning, gauze is soaked through every ten minutes.", False, "surgery", 4),
    ("Can you send my x-rays to my new dentist in Portland?", False, "front_desk", 0),
    ("Cold sensitivity on one tooth has turned into a dull constant ache over the last two days.", True, "restorative", 2),
    ("Booking a cleaning for me and my partner back to back, if possible.", False, "hygiene", 0),
    ("The metal band around my molar came loose and is spinning. It's not hurting.", False, "orthodontics", 2),
    ("I need a written estimate for a bridge to submit to my employer's benefits portal.", False, "billing", 0),
    ("Stitches from the wisdom tooth removal came out early. No pain or bleeding, just checking.", False, "surgery", 2),
    ("My whitening trays are giving me a zing of sensitivity. Should I take a break?", False, "hygiene", 1),
    ("There's a dark spot on my tooth that wasn't there before. No pain.", False, "restorative", 1),
    ("I'm at the ER after a car accident, they say my jaw may be fractured and want your records.", True, "surgery", 4),
    ("Your online form won't let me submit the medical history page.", False, "front_desk", 1),
    ("Can I pay my remaining balance over the phone?", False, "billing", 0),
    ("A tooth cracked down the middle and it throbs when I release my bite.", True, "restorative", 3),
    ("My retainer wire came unglued from behind my bottom teeth. No pain.", False, "orthodontics", 1),
    ("Getting a deep cleaning quote for my mom, she's on Medicare.", False, "billing", 0),
    ("Persistent bad breath despite brushing. Would a hygiene visit help?", False, "hygiene", 1),
    ("The numbness from this morning's filling still hasn't worn off after six hours and I bit my lip badly.", True, "restorative", 3),
    ("What's the earliest you open on weekdays? I'd like a 7am slot.", False, "front_desk", 0),
]


def rows(message: tuple[str, bool, str, int]) -> list[dict]:
    text, in_pain, department, urgency = message
    return [
        {"state": text, "question": QUESTIONS["in_pain"], "label": "yes" if in_pain else "no"},
        {"state": text, "question": QUESTIONS["department"], "label": department},
        {"state": text, "question": QUESTIONS["urgency"], "label": str(urgency)},
    ]


def main() -> int:
    splits = {"calibration": MESSAGES[0::2], "test": MESSAGES[1::2]}
    for name, messages in splits.items():
        path = HERE / f"molar_triage.{name}.jsonl"
        with path.open("w") as f:
            for message in messages:
                for row in rows(message):
                    f.write(json.dumps(row) + "\n")
        print(f"{path.name}: {len(messages)} messages, {3 * len(messages)} rows")
    (HERE / "request.json").write_text(json.dumps({
        "state": MESSAGES[2][0],
        "questions": QUESTIONS,
        "lars": {"explain": True, "abstain_margin": 0.15},
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
