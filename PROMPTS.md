# PROMPTS

Every prompt sent to Claude Code, in order.

> **Keep appending to this file as you work.** The paper says reconstructing
> prompts from memory at 1:55 is obvious to read and scores accordingly, so
> paste each one in as you send it rather than at the end.

---

## Session 1 — whole build

**Prompt 1** (invoked the `/loop` slash command with no argument, then
interrupted it):

```
/loop
```

**Prompt 2** (the actual instruction that drove the entire build — sent after
interrupting the loop, with `lead-desk-assignment.md` the only file in the
working directory):

```
implement on it step by step and teach me whats happening init and put comments to easy understand
```

---

## What that single prompt was relied on to do

Recording this because it is the honest answer to "which prompts produced
which task", and because the shape of the instruction is the reason it worked:

- **"implement on it"** — the paper was the only file in the directory, so the
  spec itself was the context. No requirements had to be restated.
- **"step by step"** — produced one task at a time with a run after each,
  rather than one large unreviewable dump. This is what surfaced the Gemini
  `tools` + `response_format` 400 at Task 3 instead of at the end.
- **"teach me whats happening"** — produced the explanation of each mechanism
  (context injection, `output_type`, tripwire raising) alongside the code.
- **"put comments to easy understand"** — the reason the source carries the
  *why* in comments, not just the *what*.

---

## Follow-up prompts

**Prompt 3** — sent after the six tasks were built and the Gemini free-tier
quota had run out mid-run:

```
complete it
```

**Prompt 4** — sent while the full six-lead run was being retried, to redirect
the remaining work to delivery:

```
after completing task push to https://github.com/Muhammadaliakmal/smit-task-3
```

<!-- Append each new prompt below, in order, under the task it belongs to. -->
