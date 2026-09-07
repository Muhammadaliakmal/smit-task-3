"""The typed verdict the agent must return.

WHY A TYPE AND NOT A PARAGRAPH
"This looks like a promising lead" cannot be sorted, filtered, counted or
branched on. `triage.priority == "high"` can. Handing the model a Pydantic
model as its output_type makes the SDK do three things:
  1. derive a JSON schema from this class,
  2. send it to the model as a required response format,
  3. validate and parse the reply back into a real Python object.
So result.final_output is a LeadTriage instance, not a str - and
result.final_output.budget_pkr is a real int you can do arithmetic on.
"""

from typing import Literal

from pydantic import BaseModel, Field


class LeadTriage(BaseModel):
    """One structured verdict about one client message."""

    intent: str = Field(
        description="What the client actually wants, in a short phrase. "
        "For example 'build an e-commerce scraper' or 'debug a production "
        "FastAPI outage'."
    )

    # `| None` matters: plenty of real messages state no budget at all.
    # Returning None is honest; returning 0 would silently poison any
    # arithmetic done downstream.
    budget_pkr: int | None = Field(
        default=None,
        description="The budget the client stated, as a whole number of PKR. "
        "Use null if the client did not state a budget. Never estimate or "
        "invent one. If a range is given, use the lower end.",
    )

    red_flags: list[str] = Field(
        default_factory=list,
        description="Short warning phrases about this message: unpaid work, "
        "equity or revenue share instead of cash, no scope, impossible "
        "deadline, hostile tone, budget withheld. Empty list if none.",
    )

    # Literal, not str: the SDK turns this into an enum in the JSON schema, so
    # the model is constrained to these three values. Without it you would get
    # "High", "urgent", "very high" and your if-statements would quietly fail.
    priority: Literal["high", "medium", "low"] = Field(
        description="high = clear scope AND real money AND worth answering "
        "now. medium = plausible but needs clarification. low = vague, "
        "underpaid, unpaid, or hostile."
    )

    suggested_reply: str = Field(
        description="A short, professional reply to send this client. Two to "
        "four sentences. Do not include any hourly rate that did not come "
        "from the lookup_rate_card tool."
    )
