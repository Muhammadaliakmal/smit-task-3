"""Entry point - and, importantly, the place where the decisions are taken.

The Agents SDK's runner is asynchronous (Runner.run is a coroutine), so the
real entry point is `async def main_async`. `main` exists only because a
console-script cannot be a coroutine - it is the sync shim that starts the
event loop exactly once.

Usage:
    uv run lead-desk                     # triage every lead in leads.json
    uv run lead-desk L2                  # triage one lead by id
    uv run lead-desk L1 L4               # triage several
    uv run lead-desk --message "..."     # triage a message not in the file
    uv run lead-desk --bonus             # Task 5B demonstration (no API calls)
    uv run lead-desk --self-test         # offline checks       (no API calls)
"""

import asyncio
import json
import sys

from agents import InputGuardrailTripwireTriggered, Runner
from agents.tool_context import ToolContext
from openai import RateLimitError

from lead_desk.agent_setup import build_research_agent, build_triage_agent
from lead_desk.bonus import run_bonus_demo
from lead_desk.config import PROJECT_ROOT
from lead_desk.context import FreelancerProfile, default_profile
from lead_desk.retry import with_rate_limit_retry
from lead_desk.schemas import LeadTriage
from lead_desk.tools import save_lead

LEADS_PATH = PROJECT_ROOT / "leads.json"

# The single rule that decides what is worth the freelancer's time. It lives
# here, in Python, as a value this file can test - not as a sentence in a
# prompt that the model may or may not honour on any given run.
SAVE_WHEN_PRIORITY_IS = "high"

DECLINE_MESSAGE = (
    "Thanks for reaching out, but I have to pass on this one. I only "
    "represent experience I actually have, so I am not able to describe my "
    "background the way this request asks. If you would like a proposal "
    "based on my real track record, I am glad to send one."
)


# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------
def load_leads() -> list[dict]:
    """Read the fixture messages from disk."""
    return json.loads(LEADS_PATH.read_text(encoding="utf-8"))


def select_leads(argv: list[str]) -> list[dict]:
    """Work out which messages to triage from the command line.

    Three shapes, all returning the same {id, platform, message} dict so the
    rest of the program does not care where a message came from:
      --message "..."  -> one ad-hoc message (used at the demo)
      L1 L4            -> those fixture ids
      (nothing)        -> every fixture lead
    """
    if argv and argv[0] in ("--bonus", "--self-test"):
        # Handled by the caller; nothing to triage.
        return []

    if argv and argv[0] == "--message":
        if len(argv) < 2:
            raise SystemExit('--message needs text, e.g. --message "hi..."')
        return [
            {"id": "AD-HOC", "platform": "cli", "message": " ".join(argv[1:])}
        ]

    leads = load_leads()
    if not argv:
        return leads

    wanted = {a.upper() for a in argv}
    chosen = [lead for lead in leads if lead["id"].upper() in wanted]
    if not chosen:
        known = ", ".join(lead["id"] for lead in leads)
        raise SystemExit(f"No lead matched {argv}. Known ids: {known}")
    return chosen


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def banner(lead: dict, triage: LeadTriage, profile: FreelancerProfile) -> str:
    """Build the one-line summary printed before the save decision is taken.

    The arithmetic here is the proof that budget_pkr is a number and not a
    string: had the model handed back "80,000 PKR" as text, the division
    below would raise a TypeError instead of printing hours.
    """
    if triage.budget_pkr is None:
        money = "budget: none stated"
    else:
        # Real arithmetic on a real int. // is integer division.
        # min_rate_pkr_hour is read here, in Python. It is never printed and
        # never sent anywhere - only the number of hours it implies is shown.
        hours_at_floor = triage.budget_pkr // profile.min_rate_pkr_hour
        money = (
            f"budget: {triage.budget_pkr:,} PKR "
            f"(~{hours_at_floor}h at my floor)"
        )

    return f"[{lead['id']}] priority: {triage.priority.upper()} | {money}"


def print_verdict(triage: LeadTriage) -> None:
    """Show the typed fields, and the fact that they really are typed."""
    print(f"  type of final_output : {type(triage).__name__}")
    print(f"  intent               : {triage.intent}")
    print(
        f"  budget_pkr           : {triage.budget_pkr!r} "
        f"({type(triage.budget_pkr).__name__})"
    )
    print(f"  red_flags            : {triage.red_flags}")
    print(f"  priority             : {triage.priority}")
    print(f"  suggested_reply      : {triage.suggested_reply}")
    print()


# ---------------------------------------------------------------------------
# THE DECISION
#
# This is the function the paper is really asking about, so it is its own
# function: one place, no model involved, branching on a typed field. It takes
# no agent and makes no API call, which is also what makes it testable
# offline - see `uv run lead-desk --self-test`.
# ---------------------------------------------------------------------------
async def decide_and_save(
    lead: dict, triage: LeadTriage, profile: FreelancerProfile
) -> bool:
    """Print the banner, then save the lead only if Python says so.

    Returns True if the lead was saved.
    """
    print(banner(lead, triage, profile))

    # The agent was never given save_lead and cannot call it. Moving this rule
    # into the prompt would turn a guarantee into a suggestion.
    if triage.priority != SAVE_WHEN_PRIORITY_IS:
        print(
            f"  -> not saved (only '{SAVE_WHEN_PRIORITY_IS}' priority is kept)"
        )
        print()
        return False

    payload = json.dumps(
        {
            "lead_id": lead["id"],
            "platform": lead["platform"],
            "triage": triage.model_dump(),
        }
    )
    # ToolContext is what the SDK itself hands a tool during a real agent
    # loop: the run context plus the identity of this particular call. We
    # build one by hand because there is no model call behind this
    # invocation - the caller is the if-statement above.
    tool_ctx = ToolContext(
        context=profile,
        tool_name=save_lead.name,
        tool_call_id=f"python-decision-{lead['id']}",
        tool_arguments=payload,
    )
    outcome = await save_lead.on_invoke_tool(tool_ctx, payload)
    print(f"  -> SAVED. {outcome}")
    print()
    return True


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
async def triage_one(
    research_agent, triage_agent, lead: dict, profile: FreelancerProfile
) -> LeadTriage | None:
    """Run one client message through both stages and act on the verdict.

    Returns the verdict, or None if the guardrail refused the message.
    """
    print("=" * 70)
    print(f"{lead['id']}  [{lead['platform']}]")
    print("-" * 70)
    print(lead["message"])
    print("-" * 70)

    try:
        # STAGE 1 - research. The input guardrail is attached to this agent,
        # so it runs here, before the first request leaves the machine. If it
        # trips, no model is called and nothing is billed.
        #
        # context=profile is the injection point: the profile travels with the
        # run and is handed to every tool, but is never serialised into any
        # message sent to the model.
        # lambda, not a coroutine: with_rate_limit_retry may need to build
        # a fresh call, and a coroutine object can only be awaited once.
        notes = await with_rate_limit_retry(
            lambda: Runner.run(research_agent, lead["message"], context=profile)
        )
    except InputGuardrailTripwireTriggered as exc:
        # Catching this is what turns a raised exception into a refusal. An
        # uncaught tripwire would be a crash, not a decline.
        reason = exc.guardrail_result.output.output_info.get("reason")
        print(f"  BLOCKED BEFORE ANY MODEL CALL - {reason}")
        print()
        print(DECLINE_MESSAGE)
        print()
        return None

    # STAGE 2 - structure. The stage-1 prose is folded into the input so the
    # triage agent works from looked-up facts rather than from guesses. Note
    # that it receives only what stage 1 chose to write down - never the
    # profile's private fields.
    stage2_input = (
        f"CLIENT MESSAGE:\n{lead['message']}\n\n"
        f"DESK NOTES:\n{notes.final_output}"
    )
    result = await with_rate_limit_retry(
        lambda: Runner.run(triage_agent, stage2_input, context=profile)
    )

    # Because the triage agent declares output_type=LeadTriage, this is a
    # validated LeadTriage instance - not a string that happens to look like
    # JSON.
    triage: LeadTriage = result.final_output
    print_verdict(triage)

    await decide_and_save(lead, triage, profile)
    return triage


async def main_async() -> None:
    # The bonus demonstration makes no model call at all, so it short-circuits
    # before any agent is built.
    if sys.argv[1:2] == ["--bonus"]:
        await run_bonus_demo()
        return

    if sys.argv[1:2] == ["--self-test"]:
        # Imported here rather than at module scope because self_test imports
        # back from this module - a top-level import would be circular.
        from lead_desk.self_test import run_self_test

        await run_self_test()
        return

    # Built once and reused. An Agent is a plain description of a
    # configuration - it holds no per-run state, so the same objects safely
    # handle every lead.
    research_agent = build_research_agent()
    triage_agent = build_triage_agent()
    profile = default_profile()

    for lead in select_leads(sys.argv[1:]):
        try:
            await triage_one(research_agent, triage_agent, lead, profile)
        except RateLimitError as exc:
            # The Gemini free tier is a small daily quota and a full run costs
            # two requests per lead. Turning this into a readable line rather
            # than a 60-line traceback matters most in a live demo.
            print(f"  RATE LIMITED by the Gemini API - stopping here.\n  {exc}")
            break


def main() -> None:
    """Sync wrapper referenced by [project.scripts] in pyproject.toml."""
    asyncio.run(main_async())
