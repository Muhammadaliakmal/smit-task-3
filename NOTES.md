# NOTES

One line per task: what came back wrong the first time, and what changed.

**Task 0 — Project and connection.** The connection itself was fine, but the
first run died on `GEMINI_API_KEY is not set` even though a `.env` existed —
the key I copied across from an earlier project was present but *empty*
(`GEMINI_API_KEY=`), so `load_dotenv` succeeded and the value was still `""`.
Changed `build_model()` to raise a named error on a falsy key rather than
letting an empty string travel down to a confusing 401 several layers deeper.

**Task 1 — Sample data and lookup tools.** The tools worked first time, but I
had put `RATE_CARD_PKR_PER_HOUR` and `HOURS_FREE_PER_WEEK` at module level,
which is exactly the thing Task 2 then had to undo. Also learned the SDK
parses the docstring's `Args:` block into *per-parameter* descriptions, not
just one blob — so rewriting the docstrings as instructions aimed at the model
(rather than notes for a maintainer) is what actually made tool selection
reliable.

**Task 2 — Data the model is never given.** First attempt still had the tools
reading module globals with the profile merely passed alongside, which proves
nothing. Changed both tools to take `ctx: RunContextWrapper[FreelancerProfile]`
as their first parameter and read `ctx.context`; verified with
`tool.params_json_schema` that `ctx` is absent from the schema, and with
`grep -rn "3500" src/` that the minimum rate exists at exactly one line in
`context.py`. Three prompt-injection attempts returned no number.

**Task 3 — A verdict the program can act on.** Two real failures. First,
Gemini returned `400 - Function calling with a response mime type:
'application/json' is unsupported`: this provider will not accept `tools` and
a forced JSON `response_format` in the same request, so `tools=[...]` plus
`output_type=...` on one agent is impossible here. Split into a two-stage
pipeline — a research agent with the tools, then a tool-less triage agent that
returns `LeadTriage`. Second, invoking `save_lead` from Python with a bare
`RunContextWrapper` raised `AttributeError: 'RunContextWrapper' object has no
attribute 'run_config'`; the correct object is `ToolContext`, which carries the
run context *plus* the identity of the call.

**Task 4 — Refusing before you pay.** The first pattern set was a flat keyword
list (`experience`, `years`, `claim`) and it blocked lead L1 and a legitimate
"we need someone with scraping experience" message — a false positive, which
the paper rightly calls the worse fault. Rewrote each rule to require a
*deception verb* aimed at a *credentials noun* within one sentence
(`[^.\n]{0,40}` gaps, no `re.DOTALL`). Now 7/7 dishonest messages block and
9/9 legitimate ones pass, in 0.61 ms total. In-process the tripwire raises in
~1.8 ms against 1500–4000 ms for a real round-trip.

**Task 5 — Bonus B, conditional tools.** Worked as expected once I found that
`is_enabled` accepts a callable of `(RunContextWrapper, Agent)`. The thing I
had to check rather than assume was how to *prove* the claim: the evidence is
`await agent.get_all_tools(ctx)`, which is the same call the SDK makes when
assembling a request, so its output is literally the `tools` array that gets
sent. Unverified → `['lookup_rate_card', 'check_availability']`.

---

## Operational note discovered while testing

The Gemini free tier on this key allows **20 `generate_content` requests**
before returning `429 RESOURCE_EXHAUSTED`. A full six-lead run costs **12
requests** (two per lead, because of the two-stage pipeline). `main_async`
catches `RateLimitError` and stops with one readable line instead of a
traceback, and `retry.py` honours the server's own `retryDelay`. When the
daily quota is genuinely spent the suggested delay *grows* (30s, then 58s)
rather than shrinking, which is how you tell an exhausted day from a
momentary burst limit.

`config.py` also accepts `OPENAI_API_KEY` as a fallback provider so a demo is
not hostage to one quota. Gemini stays the default and the required target;
`MODEL_PROVIDER=openai` forces the fallback. For a live demo, run two or three leads plus `--bonus` and the
guardrail (which cost zero requests between them).
