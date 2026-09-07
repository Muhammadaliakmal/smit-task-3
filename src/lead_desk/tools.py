"""Tools the agent may call.

A tool's DOCSTRING is not for you - it is the description the model reads
when deciding whether to call the tool. The SDK turns the function signature
into a JSON schema and the docstring into the description, and ships both to
the model. So these docstrings are written as instructions to the model.

THE IMPORTANT DETAIL IN THIS FILE
Every tool's first parameter is `ctx: RunContextWrapper[FreelancerProfile]`.
The SDK recognises that annotation and strips the parameter out before it
builds the JSON schema - so it never appears in the schema, the model never
sees it, and the model cannot supply or influence it. Your own code fills it
in via Runner.run(..., context=profile).

Result: two kinds of parameter in one function.
  - `skill`, `week`  -> came FROM the model. Untrusted. In the schema.
  - `ctx`            -> came from YOUR PROGRAM. Trusted. Invisible.
The minimum rate lives on the second kind, which is why it cannot leak.
"""

import json
from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool

from lead_desk.config import PROJECT_ROOT
from lead_desk.context import FreelancerProfile
from lead_desk.schemas import LeadTriage


@function_tool
def lookup_rate_card(
    ctx: RunContextWrapper[FreelancerProfile], skill: str
) -> str:
    """Look up the freelancer's hourly rate, in PKR, for one named skill.

    You MUST call this before you mention, quote, estimate, compare or
    negotiate any price. Never state an hourly rate that did not come back
    from this tool. If this tool reports that a skill is not on the rate
    card, say plainly that you do not have a rate for it - do not substitute
    a similar skill's rate and do not guess a number.

    Args:
        skill: The single skill to price, lowercase, e.g. "python",
            "web scraping", "fastapi", "data analysis", "automation".
    """
    # ctx.context is the FreelancerProfile instance handed to Runner.run.
    profile = ctx.context

    rate = profile.rate_for(skill)
    if rate is None:
        return (
            f"NOT ON RATE CARD: there is no published rate for '{skill}'. "
            f"You do not know what this costs. Say so."
        )

    # Note what is NOT returned: profile.min_rate_pkr_hour. The tool can read
    # the floor, but it never reports it, so it never enters the conversation.
    return f"{skill}: {rate} PKR per hour."


@function_tool
def check_availability(
    ctx: RunContextWrapper[FreelancerProfile], week: str
) -> str:
    """Check how many working hours the freelancer has free in a given week.

    Call this before you promise a start date, agree to a deadline, or say
    whether a job fits. Never guess at availability.

    Args:
        week: Which week to check - "this week" or "next week".
    """
    hours = ctx.context.hours_free(week)
    if hours is None:
        return f"NO AVAILABILITY DATA for '{week}'. Ask for this week or next week."
    return f"{week}: {hours} hours free."


# ---------------------------------------------------------------------------
# save_lead is a real tool - it has a schema, it takes run context, and it can
# be invoked exactly like the two above. It is deliberately NOT handed to the
# agent. Task 3 requires that the decision to save is taken by Python reading
# a typed field, not by the model choosing to call a function. Giving the
# agent this tool would move the guesswork rather than remove it.
# main.py invokes it directly via save_lead.on_invoke_tool(...).
# ---------------------------------------------------------------------------
@function_tool
async def save_lead(
    ctx: RunContextWrapper[FreelancerProfile],
    lead_id: str,
    platform: str,
    triage: LeadTriage,
) -> str:
    """Append one triaged lead to saved.json, the freelancer's worth-answering pile.

    Args:
        lead_id: The id of the lead being saved, e.g. "L1".
        platform: Where the message arrived from, e.g. "Upwork".
        triage: The structured verdict for this lead.
    """
    saved_path = PROJECT_ROOT / "saved.json"

    # Read-modify-write. saved.json is a JSON array, so we cannot simply
    # append text to the end of the file - it has to be parsed, extended and
    # re-serialised or the file stops being valid JSON.
    try:
        existing = json.loads(saved_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing.append(
        {
            "lead_id": lead_id,
            "platform": platform,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "saved_for": ctx.context.name,
            "triage": triage.model_dump(),
        }
    )

    saved_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return f"Saved {lead_id} to saved.json ({len(existing)} leads on file)."


# ===========================================================================
# TASK 5, OPTION B - a tool that only exists for a verified profile.
# ===========================================================================
def profile_is_verified(
    ctx: RunContextWrapper[FreelancerProfile], agent
) -> bool:
    """Gate for send_proposal: is this run's profile a verified account?

    The SDK calls this once per run, while assembling the tool list. If it
    returns False the tool is dropped BEFORE the request is built, so its
    name, description and schema never reach the model. That is different in
    kind from telling the model "do not use send_proposal unless verified":
    an instruction can be argued with, an absent tool cannot be called.
    """
    return ctx.context.verified


@function_tool(is_enabled=profile_is_verified)
async def send_proposal(
    ctx: RunContextWrapper[FreelancerProfile],
    lead_id: str,
    message: str,
) -> str:
    """Send a written proposal to a client. Only available to verified accounts.

    Args:
        lead_id: The id of the lead the proposal answers, e.g. "L1".
        message: The proposal text to send to the client.
    """
    proposals_path = PROJECT_ROOT / "proposals.json"

    try:
        existing = json.loads(proposals_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing.append(
        {
            "lead_id": lead_id,
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sent_by": ctx.context.name,
            "message": message,
        }
    )
    proposals_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return f"Proposal for {lead_id} sent and logged."
