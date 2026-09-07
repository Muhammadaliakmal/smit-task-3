"""Task 5, option B - demonstration that an unverified run never sees the tool.

Run with:  uv run lead-desk --bonus

WHAT THIS PROVES
Agent.get_all_tools(run_context) is the exact call the SDK makes internally
while building a request: it walks the agent's tool list, evaluates each
tool's `is_enabled` against this run's context, and returns the survivors.
Whatever comes back from it is precisely what gets serialised into the
request's `tools` array.

So printing it for an unverified profile and again for a verified one is not
an approximation of the difference - it is the difference. In the first run
`send_proposal` is not in the list, which means its name, description and
schema are never sent, which means the model cannot call a tool it has never
been told exists.
"""

import asyncio

from agents import RunContextWrapper

from lead_desk.agent_setup import build_research_agent
from lead_desk.config import ensure_offline_credentials
from lead_desk.context import FreelancerProfile, default_profile


async def tools_offered_for(profile: FreelancerProfile) -> list[str]:
    """Return the names of the tools this profile's run would actually send."""
    agent = build_research_agent()

    # get_all_tools needs a run context because is_enabled is a function OF
    # the context. Building one by hand lets us inspect the tool list without
    # spending a model call.
    ctx = RunContextWrapper(context=profile)
    tools = await agent.get_all_tools(ctx)
    return [tool.name for tool in tools]


async def run_bonus_demo() -> None:
    # This demo inspects the tool list without calling the API, so it runs
    # even on a machine with no key configured.
    ensure_offline_credentials()

    unverified = default_profile(verified=False)
    verified = default_profile(verified=True)

    print("=" * 70)
    print("TASK 5B - a tool that only exists when profile.verified is True")
    print("=" * 70)

    for label, profile in (
        ("UNVERIFIED", unverified),
        ("VERIFIED", verified),
    ):
        names = await tools_offered_for(profile)
        print()
        print(f"  profile.verified = {profile.verified}  ({label})")
        print(f"  tools actually sent to the model : {names}")
        print(f"  is 'send_proposal' among them?   : {'send_proposal' in names}")

    print()
    print("-" * 70)
    print(
        "The unverified run does not merely decline to use send_proposal.\n"
        "The tool is not in the request at all - the model is never told it\n"
        "exists, so there is nothing for it to be talked into calling."
    )
    print("-" * 70)


def main() -> None:
    asyncio.run(run_bonus_demo())
