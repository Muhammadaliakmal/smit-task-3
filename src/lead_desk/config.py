"""Environment loading and model wiring.

Everything that knows about API keys, base URLs and model names lives here,
so the rest of the program never touches os.environ.

WHY GEMINI THROUGH AN "OPENAI" CLASS?
Google publishes an OpenAI-compatible Chat Completions endpoint. The Agents
SDK does not talk to "OpenAI the company" - it talks to any object that
behaves like an OpenAI client. So we point that client at Google's URL and
hand it the Gemini key. Nothing else in the codebase has to change.

WHY THERE IS A SECOND PROVIDER
Gemini is the required target and is always preferred. But its free tier
allows 20 requests, and a six-lead run costs 12, so a demo can be stopped by
quota rather than by anything being wrong. If GEMINI_API_KEY is missing or
its quota is spent, OPENAI_API_KEY is used instead.

That fallback costs almost nothing to support, which is itself the point:
only base_url, api_key and the model name differ between the two branches
below. Everything downstream - tools, run context, guardrails, output_type -
is written against the SDK, not against a vendor.
"""

import os
from pathlib import Path

from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

# __file__ is .../lead-desk/src/lead_desk/config.py
#   parents[0] = src/lead_desk , parents[1] = src , parents[2] = project root
# Resolving from the file (not the cwd) means the program finds .env no matter
# which directory you launch it from. load_dotenv() with no argument searches
# upward from the working directory instead, which breaks as soon as you run
# from a subfolder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL_DEFAULT = "gpt-4.1-mini"

# The SDK's default tracing exporter uploads run data to OpenAI's platform.
# We are not authenticated there, so those uploads would only produce noisy
# warnings. Turn tracing off once, globally, at import time.
set_tracing_disabled(True)


def load_environment() -> None:
    """Read .env into os.environ. Safe to call more than once."""
    load_dotenv(PROJECT_ROOT / ".env")


def _clean(name: str) -> str:
    """Read an environment variable, treating blank as missing.

    A line like `GEMINI_API_KEY=` parses successfully and yields "", which is
    falsy but not absent - and an empty key produces a confusing 401 several
    layers deeper instead of an obvious error here.
    """
    return (os.environ.get(name) or "").strip()


def active_provider() -> str:
    """Return which provider this run will use: 'gemini' or 'openai'.

    Gemini wins whenever a key is present, because it is the required target.
    OPENAI_PROVIDER=openai forces the fallback, which is how you demo while
    the Gemini free-tier quota is spent.
    """
    load_environment()

    forced = _clean("MODEL_PROVIDER").lower()
    if forced in ("gemini", "openai"):
        return forced

    if _clean("GEMINI_API_KEY"):
        return "gemini"
    if _clean("OPENAI_API_KEY"):
        return "openai"

    raise RuntimeError(
        "No API key set. Put GEMINI_API_KEY (preferred) or OPENAI_API_KEY "
        "in lead-desk/.env - that file is git-ignored."
    )


def ensure_offline_credentials() -> bool:
    """Let the no-API-call commands run on a machine with no key configured.

    --self-test and --bonus never reach the network, but they do build Agent
    objects, and an Agent needs a model, which needs a client, which needs a
    key *string*. So supply a placeholder when nothing is configured.

    This is deliberately not done inside build_model(): a real run must still
    fail loudly on a missing key rather than quietly get a placeholder and a
    401 fifty lines later. Returns True if a placeholder was installed.
    """
    load_environment()

    if _clean("GEMINI_API_KEY") or _clean("OPENAI_API_KEY"):
        return False

    os.environ["GEMINI_API_KEY"] = "offline-placeholder-never-sent"
    return True


def build_model() -> OpenAIChatCompletionsModel:
    """Return the chat model every agent in this project runs on."""
    provider = active_provider()

    if provider == "gemini":
        api_key = _clean("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MODEL_PROVIDER=gemini but GEMINI_API_KEY is empty in .env."
            )
        # AsyncOpenAI, not OpenAI: the Agents SDK's runner is asynchronous, so
        # the transport underneath it has to be awaitable too.
        client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
        model_name = GEMINI_MODEL
    else:
        api_key = _clean("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MODEL_PROVIDER=openai but OPENAI_API_KEY is empty in .env."
            )
        # No base_url: the OpenAI client already points at OpenAI by default.
        client = AsyncOpenAI(api_key=api_key)
        model_name = _clean("OPENAI_MODEL") or OPENAI_MODEL_DEFAULT

    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def describe_provider() -> str:
    """One line naming the provider and model, printed at startup.

    Worth showing explicitly: at a demo, "which model is this actually
    talking to" should never be something anyone has to guess.
    """
    provider = active_provider()
    model = (
        GEMINI_MODEL
        if provider == "gemini"
        else (_clean("OPENAI_MODEL") or OPENAI_MODEL_DEFAULT)
    )
    return f"provider: {provider}  |  model: {model}"
