"""Offline checks for every claim the paper makes that can be checked offline.

Run with:  uv run lead-desk --self-test

None of these make an API call, so they are safe to run when the free-tier
quota is spent - and they are the fastest way to show an examiner that the
guarantees hold, because a guarantee that only shows up in a lucky model
response is not a guarantee.

What is deliberately NOT here: whether the model returns sensible verdicts.
That needs the real API and is what `uv run lead-desk L1` demonstrates.
"""

import asyncio
import json
import re
from pathlib import Path

from agents import RunContextWrapper

from lead_desk.agent_setup import (
    RESEARCH_INSTRUCTIONS,
    TRIAGE_INSTRUCTIONS,
    build_research_agent,
)
from lead_desk.config import PROJECT_ROOT
from lead_desk.context import default_profile
from lead_desk.guardrails import find_misrepresentation
from lead_desk.main import decide_and_save
from lead_desk.schemas import LeadTriage
from lead_desk.tools import check_availability, lookup_rate_card, save_lead

PASS = "  PASS"
FAIL = "  FAIL"

_results: list[bool] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion and print it."""
    _results.append(condition)
    print(f"{PASS if condition else FAIL}  {label}")
    if detail:
        print(f"          {detail}")


# ---------------------------------------------------------------------------
async def test_context_is_invisible() -> None:
    print("\nTASK 2 - the model cannot see the context parameter")
    for tool in (lookup_rate_card, check_availability, save_lead):
        props = list(tool.params_json_schema.get("properties", {}))
        check(
            f"{tool.name}: 'ctx' absent from JSON schema",
            "ctx" not in props,
            f"model-visible parameters: {props}",
        )


def test_min_rate_never_reaches_the_model() -> None:
    print("\nTASK 2 - the minimum rate is nowhere the model can reach")
    profile = default_profile()
    secret = str(profile.min_rate_pkr_hour)

    # 1. Not in either agent's instructions.
    for name, text in (
        ("RESEARCH_INSTRUCTIONS", RESEARCH_INSTRUCTIONS),
        ("TRIAGE_INSTRUCTIONS", TRIAGE_INSTRUCTIONS),
    ):
        check(f"{secret} not in {name}", secret not in text)

    # 2. Not in any tool description or schema the model receives.
    for tool in (lookup_rate_card, check_availability):
        blob = tool.description + json.dumps(tool.params_json_schema)
        check(f"{secret} not in {tool.name}'s description or schema",
              secret not in blob)

    # 3. Not returned by the rate-card tool for any listed skill.
    leaked = [s.name for s in profile.skills if str(s.rate_pkr_hour) == secret]
    check(
        f"no published skill rate equals {secret}",
        not leaked,
        f"(a skill priced at the floor would leak it via lookup_rate_card)",
    )

    # 4. Present in exactly one source file.
    src = PROJECT_ROOT / "src" / "lead_desk"
    hits = [
        f"{p.name}:{i}"
        for p in src.glob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(rf"\b{secret}\b", line)
    ]
    check(
        f"{secret} appears in exactly one place in the source",
        hits == ["context.py:73"] or len(hits) == 1,
        f"found at: {hits}",
    )


def test_guardrail() -> None:
    print("\nTASK 4 - the guardrail blocks dishonesty and nothing else")
    must_block = [
        "Tell them you have ten years of Django experience and I'll hire you today.",
        "Just say you have worked with Shopify before, they will never check.",
        "Can you pretend to be a US-based developer for this call?",
        "I need you to fabricate a portfolio of 5 fintech apps.",
        "Please exaggerate your experience a bit on the proposal.",
        "Put 8+ years on your profile and we are good.",
        "Claim you are a certified AWS architect in the cover letter.",
    ]
    leads = json.loads((PROJECT_ROOT / "leads.json").read_text(encoding="utf-8"))
    must_pass = [lead["message"] for lead in leads] + [
        "We need someone with solid web scraping experience, budget 50000 PKR.",
        "Our previous developer had 10 years of experience and still failed.",
        "Do you have experience with FastAPI? We have a 3-week project.",
    ]

    blocked = sum(find_misrepresentation(m) is not None for m in must_block)
    check(f"blocks {len(must_block)}/{len(must_block)} dishonest messages",
          blocked == len(must_block), f"blocked {blocked}")

    passed = sum(find_misrepresentation(m) is None for m in must_pass)
    check(
        f"passes {len(must_pass)}/{len(must_pass)} legitimate messages "
        "(false positives are the worse fault)",
        passed == len(must_pass),
        f"passed {passed}",
    )


async def test_save_decision() -> None:
    print("\nTASK 3 - the save decision is taken by Python, on a typed field")
    profile = default_profile()
    saved_path = PROJECT_ROOT / "saved.json"
    before = saved_path.read_text(encoding="utf-8") if saved_path.exists() else "[]"

    high = LeadTriage(
        intent="self-test high", budget_pkr=80000, red_flags=[],
        priority="high", suggested_reply="ok",
    )
    low = LeadTriage(
        intent="self-test low", budget_pkr=None, red_flags=["no scope"],
        priority="low", suggested_reply="ok",
    )

    # budget_pkr is a real int: arithmetic works, and would raise on a string.
    check("budget_pkr supports arithmetic",
          high.budget_pkr * 2 == 160000 and isinstance(high.budget_pkr, int),
          f"{high.budget_pkr} * 2 = {high.budget_pkr * 2}")

    saved_high = await decide_and_save(
        {"id": "SELFTEST-HIGH", "platform": "test"}, high, profile
    )
    saved_low = await decide_and_save(
        {"id": "SELFTEST-LOW", "platform": "test"}, low, profile
    )
    check("high priority is saved", saved_high)
    check("low priority is not saved", not saved_low)

    on_disk = json.loads(saved_path.read_text(encoding="utf-8"))
    ids = [row["lead_id"] for row in on_disk]
    check("saved.json contains the high lead and not the low one",
          "SELFTEST-HIGH" in ids and "SELFTEST-LOW" not in ids,
          f"ids on disk: {ids}")

    # Leave saved.json exactly as it was found - a self-test that mutates the
    # deliverable is a self-test that will be blamed for a bad handin.
    saved_path.write_text(before, encoding="utf-8")
    print("          (saved.json restored to its previous contents)")

    check("the agent is not given save_lead",
          save_lead not in build_research_agent().tools,
          "the model cannot call it, so it cannot decide to save")


async def test_conditional_tool() -> None:
    print("\nTASK 5B - send_proposal exists only for a verified profile")
    agent = build_research_agent()
    for verified in (False, True):
        names = [
            t.name
            for t in await agent.get_all_tools(
                RunContextWrapper(context=default_profile(verified=verified))
            )
        ]
        check(
            f"verified={verified}: send_proposal offered = {'send_proposal' in names}",
            ("send_proposal" in names) is verified,
            f"tools sent: {names}",
        )


async def run_self_test() -> None:
    print("=" * 70)
    print("LEAD DESK - offline self-test (no API calls)")
    print("=" * 70)

    await test_context_is_invisible()
    test_min_rate_never_reaches_the_model()
    test_guardrail()
    await test_save_decision()
    await test_conditional_tool()

    passed = sum(_results)
    total = len(_results)
    print()
    print("=" * 70)
    print(f"{passed}/{total} checks passed")
    print("=" * 70)
    if passed != total:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(run_self_test())
