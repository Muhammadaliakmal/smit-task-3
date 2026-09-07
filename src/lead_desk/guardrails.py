"""Input guardrail: refuse dishonesty requests without paying for them.

THE POINT OF THIS FILE
Some messages have exactly one correct answer - no - and sending them to the
model costs money and time to arrive at an answer we already know. "Tell them
you have ten years of Django experience and I'll hire you today" is one.

So the check is pure Python: regular expressions over the raw text, no
network, no model, no await. It runs before the first token is ever sent.
A blocked message is rejected in well under a millisecond.

DESIGNING FOR THE RIGHT FAILURE
A guardrail that blocks good leads is worse than the problem it fixes: a
false positive silently costs real work. So the patterns below do not fire on
a single suspicious word. Each one requires a *deception verb* aimed at a
*credentials noun* - "say you have ... experience", "pretend you built ...".
Ordinary phrases that merely mention experience ("we need someone with
scraping experience") match nothing here.
"""

import re
from dataclasses import dataclass

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)

from lead_desk.context import FreelancerProfile


@dataclass(frozen=True)
class MisrepresentationRule:
    """One pattern, plus the plain-English reason it exists."""

    pattern: re.Pattern[str]
    reason: str


def _rule(regex: str, reason: str) -> MisrepresentationRule:
    """Compile one case-insensitive rule.

    re.IGNORECASE because client messages are not written carefully, and
    re.DOTALL is deliberately NOT set - these patterns should match within a
    sentence, not drift across a whole paragraph and produce a false hit.
    """
    return MisrepresentationRule(re.compile(regex, re.IGNORECASE), reason)


# Each pattern pairs a deception verb with a credentials target. The
# `[^.\n]{0,40}` gaps allow a few words in between ("tell them that you have
# roughly ten years of ...") while refusing to jump across a sentence
# boundary, which is what keeps false positives down.
CREDENTIALS = r"(experience|expertise|years?|worked|built|shipped|portfolio|r[eé]sum[eé]|cv|certif\w*|degree|background)"

RULES: list[MisrepresentationRule] = [
    _rule(
        rf"\b(tell|say to|inform)\b[^.\n]{{0,20}}\b(them|him|her|the client|client)\b[^.\n]{{0,40}}\b{CREDENTIALS}\b",
        "asks me to tell the client something about my experience that is not mine to claim",
    ),
    _rule(
        rf"\b(just )?say (that )?(you|we)\b[^.\n]{{0,40}}\b{CREDENTIALS}\b",
        "asks me to say I have experience I do not have",
    ),
    _rule(
        r"\b(pretend|pose as|act like|claim)\b[^.\n]{0,40}\b(you|we|to be|you're|youre|you are)\b",
        "asks me to pretend to be something I am not",
    ),
    _rule(
        rf"\b(fake|fabricate|invent|make up|made[- ]up|forge)\b[^.\n]{{0,40}}\b{CREDENTIALS}\b",
        "asks me to fabricate credentials",
    ),
    _rule(
        rf"\b(exaggerate|overstate|inflate|pad|embellish|beef up|big up)\b[^.\n]{{0,40}}\b(your|our|the)?\s?{CREDENTIALS}\b",
        "asks me to overstate my experience",
    ),
    _rule(
        rf"\b(lie|dishonest|not really true|doesn'?t matter if it'?s true)\b[^.\n]{{0,40}}\b({CREDENTIALS}|about)\b",
        "asks me to lie to the client",
    ),
    _rule(
        rf"\bput\b[^.\n]{{0,25}}\b\d+\+?\s*years?\b[^.\n]{{0,25}}\b(on|in)\b[^.\n]{{0,20}}\b(profile|{CREDENTIALS})\b",
        "asks me to put years of experience on my profile that I have not worked",
    ),
]


def find_misrepresentation(text: str) -> MisrepresentationRule | None:
    """Return the first rule this text trips, or None if it is clean.

    Returning the rule rather than a bare bool means the decline message can
    explain itself, and a failing test can name which pattern misfired.
    """
    for rule in RULES:
        if rule.pattern.search(text):
            return rule
    return None


def _as_text(user_input: str | list[TResponseInputItem]) -> str:
    """Flatten the runner's input into one searchable string.

    Runner.run accepts either a plain string or a list of input items, and a
    guardrail has to cope with both. Anything unrecognised is stringified
    rather than skipped: a guardrail that quietly sees an empty string is a
    guardrail that quietly passes everything.
    """
    if isinstance(user_input, str):
        return user_input

    parts: list[str] = []
    for item in user_input:
        if isinstance(item, dict):
            content = item.get("content", "")
            parts.append(content if isinstance(content, str) else str(content))
        else:
            parts.append(str(item))
    return "\n".join(parts)


@input_guardrail
async def no_misrepresentation(
    ctx: RunContextWrapper[FreelancerProfile],
    agent: Agent[FreelancerProfile],
    user_input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Block any message asking the freelancer to misrepresent their experience.

    Declared `async` because that is the signature the SDK calls, but there is
    not a single `await` in the body - there is nothing to wait for. No model
    is consulted, no network call is made. That is the whole design: the
    answer to "will you lie for me" does not require an API call.
    """
    rule = find_misrepresentation(_as_text(user_input))

    # GuardrailFunctionOutput carries two things: whatever detail we want to
    # keep (output_info), and the tripwire. Setting tripwire_triggered=True
    # makes the SDK raise InputGuardrailTripwireTriggered *instead of* calling
    # the model - so the run stops before any tokens are billed.
    return GuardrailFunctionOutput(
        output_info={"reason": rule.reason if rule else None},
        tripwire_triggered=rule is not None,
    )
