# Lead Desk

Triage agent for freelance client messages. Takes one raw message, works out
what the client wants, checks it against a private rate card and calendar
through tools, and returns a typed verdict a program can branch on.

Built on the OpenAI Agents SDK running against Gemini `gemini-2.5-flash`.

## Run it

```bash
uv run lead-desk                     # triage all six leads in leads.json
uv run lead-desk L1                  # triage one lead
uv run lead-desk L1 L4               # triage several
uv run lead-desk --message "..."     # triage a message not in the file
uv run lead-desk --bonus             # Task 5B demo (makes no API calls)
```

Requires `GEMINI_API_KEY` in `.env` (git-ignored).

## How it fits together

```
leads.json ──> [ guardrail ]  pure Python, no model call
                    │  trips? -> polite decline, exit 0
                    ▼
              STAGE 1  Research agent
                  tools: lookup_rate_card, check_availability
                         (+ send_proposal only if profile.verified)
                  context: FreelancerProfile  <- never sent to the model
                    │  prose desk notes
                    ▼
              STAGE 2  Triage agent
                  no tools, output_type=LeadTriage
                    │  LeadTriage object
                    ▼
              main.py:  if triage.priority == "high":  save_lead(...)
                    ▼
              saved.json
```

## Where each requirement lives

| Requirement | File |
|---|---|
| Async entry point, single command | `main.py`, `pyproject.toml` `[project.scripts]` |
| Gemini wiring, `.env` loading | `config.py` |
| Six sample leads | `leads.json` |
| `lookup_rate_card`, `check_availability` | `tools.py` |
| `FreelancerProfile` context, private minimum rate | `context.py` |
| `LeadTriage` typed output | `schemas.py` |
| Two agents and why there are two | `agent_setup.py` |
| `save_lead`, and the decision to call it | `tools.py`, `main.py` |
| Input guardrail, no model call | `guardrails.py` |
| Task 5B conditional tool + proof | `tools.py`, `bonus.py` |

## The two design claims worth defending

**The minimum rate is absent, not protected.** `min_rate_pkr_hour` lives on
the run context. It reaches the tools and never reaches the model — not in the
instructions, not in a message, not in a tool schema. `grep -rn "3500" src/`
returns exactly one line, in `context.py`. An instruction like "never reveal
your minimum rate" would be a refusal the model can be argued out of; this is
absence, which cannot be argued with.

**The save decision is an `if` statement.** `save_lead` is a real tool with a
real schema, and it is deliberately not in the agent's tool list. `main.py`
reads `triage.priority` and invokes the tool itself. Letting the model choose
to call it would move the guesswork, not remove it.

## Two provider constraints found the hard way

1. Gemini rejects `tools` + a forced JSON `response_format` in one request
   (`400 ... 'application/json' is unsupported`). Hence two agents rather than
   one — see the module docstring in `agent_setup.py`.
2. The free tier allows 20 requests. A full six-lead run costs 12 (two per
   lead). `RateLimitError` is caught and reported in one line.
