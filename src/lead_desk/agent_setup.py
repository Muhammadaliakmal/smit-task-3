"""The Lead Desk agents.

WHY THERE ARE TWO AGENTS AND NOT ONE
Gemini rejects any request that carries both function-calling tools and a
forced JSON response format:

    400 - Function calling with a response mime type: 'application/json'
          is unsupported

The Agents SDK implements output_type by setting response_format, so
tools=[...] together with output_type=... is an illegal combination on this
provider (it is legal on OpenAI's own endpoint). Rather than give up one of
the two requirements, the work is split into two runs:

    1. RESEARCH agent - has the tools, has the private context, returns prose.
    2. TRIAGE agent   - has no tools, returns a LeadTriage object.

That is also cleaner than one agent doing both: gathering facts and
committing to a shape are different jobs, and each is testable on its own.
"""

from agents import Agent, ModelSettings

from lead_desk.config import build_model
from lead_desk.context import FreelancerProfile
from lead_desk.guardrails import no_misrepresentation
from lead_desk.schemas import LeadTriage
from lead_desk.tools import check_availability, lookup_rate_card, send_proposal

# NOTE ON THE INSTRUCTIONS BELOW:
# There is no money in either of these strings. No hourly rate, no minimum,
# no "don't go below X". Anything written here is part of the conversation
# and can be talked out of the model by a client who writes "ignore your
# instructions". The numbers live in tools + context instead (Tasks 1 and 2).

RESEARCH_INSTRUCTIONS = """
You are the research step of a triage desk for a freelance software
developer who takes work from Upwork and Fiverr.

You are given one raw client message. Establish the facts a decision needs:
what the client actually wants, what (if anything) they have offered to pay,
what skill the job needs, and whether the message carries warning signs -
unpaid work, equity or revenue share instead of cash, no scope at all, an
impossible deadline, or pressure and hostility.

You have tools for money and for time. Use them:
- Before you mention ANY hourly rate or price, call lookup_rate_card.
- Before you comment on whether the work fits, call check_availability.

Never state a rate or an amount of free time that did not come back from a
tool. If a tool reports that a skill is not on the rate card, write that the
rate is unknown - do not estimate one.

Answer as short bullet points. These notes are read by another program, not
by the client.
""".strip()

TRIAGE_INSTRUCTIONS = """
You turn a client message and a set of desk notes into one structured
verdict. The notes were produced by a colleague who already looked up the
rate card and the calendar; treat the figures in them as authoritative and
do not invent any others.

Rules:
- budget_pkr is only ever a figure the CLIENT stated. If the client named no
  budget, use null. Never copy an hourly rate into it, and never estimate.
- red_flags must be non-empty whenever the message offers equity, revenue
  share, exposure or anything other than money; gives no scope; sets an
  impossible deadline; or is hostile.
- priority is high only when the scope is clear AND real money is on the
  table AND the message is worth answering today. Unpaid work, equity
  instead of cash, no scope, or a hostile tone is never high.
- suggested_reply is two to four professional sentences addressed to the
  client. Do not put any hourly rate in it that is not in the desk notes.
""".strip()


def build_research_agent() -> Agent[FreelancerProfile]:
    """Stage 1: the agent that may call tools.

    Generic over FreelancerProfile so a type checker verifies that every tool
    attached here expects RunContextWrapper[FreelancerProfile] - a mismatch
    is caught before the program runs, not at the first model call.
    """
    return Agent[FreelancerProfile](
        name="Lead Desk Research",
        instructions=RESEARCH_INSTRUCTIONS,
        model=build_model(),
        # Passing the tools here is what puts their names, descriptions and
        # JSON schemas into the request. A tool the agent does not carry is a
        # tool the model has never heard of and cannot call. save_lead is
        # deliberately absent - see the comment on it in tools.py.
        #
        # send_proposal is listed but carries is_enabled=profile_is_verified,
        # so the SDK filters it out of this list at run time whenever the
        # profile is unverified (Task 5, option B).
        tools=[lookup_rate_card, check_availability, send_proposal],
        # The guardrail hangs off the FIRST agent to run, because that is the
        # only place it can save money: it fires before this agent's request
        # is built, so a blocked message costs nothing at all.
        input_guardrails=[no_misrepresentation],
        model_settings=ModelSettings(temperature=0.0),
    )


def build_triage_agent() -> Agent[FreelancerProfile]:
    """Stage 2: the agent that returns the type.

    No tools, so output_type is legal on Gemini here.
    """
    return Agent[FreelancerProfile](
        name="Lead Desk Triage",
        instructions=TRIAGE_INSTRUCTIONS,
        model=build_model(),
        # output_type is what turns result.final_output from a str into a
        # LeadTriage. The SDK derives a JSON schema from the class, sends it
        # as the required response format, then validates the reply back into
        # a real Python object.
        output_type=LeadTriage,
        # temperature=0: triage is a classification, not creative writing.
        # Sampling randomness here only buys flaky output.
        model_settings=ModelSettings(temperature=0.0),
    )
