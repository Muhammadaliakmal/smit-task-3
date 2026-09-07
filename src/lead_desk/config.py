"""Environment loading and model wiring.

Everything that knows about API keys, base URLs and model names lives here,
so the rest of the program never touches os.environ.

WHY GEMINI THROUGH AN "OPENAI" CLASS?
Google publishes an OpenAI-compatible Chat Completions endpoint. The Agents
SDK does not talk to "OpenAI the company" - it talks to any object that
behaves like an OpenAI client. So we point that client at Google's URL and
hand it the Gemini key. Nothing else in the codebase has to change.
"""

import os
from pathlib import Path

from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

# __file__ is .../lead-desk/src/lead_desk/config.py
#   parents[0] = src/lead_desk , parents[1] = src , parents[2] = project root
# Resolving from the file (not the cwd) means the program finds .env no matter
# which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

# The SDK's default tracing exporter uploads run data to OpenAI's platform.
# We are not authenticated there, so those uploads would only produce noisy
# warnings. Turn tracing off once, globally, at import time.
set_tracing_disabled(True)


def load_environment() -> None:
    """Read .env into os.environ. Safe to call more than once."""
    load_dotenv(PROJECT_ROOT / ".env")


def build_model() -> OpenAIChatCompletionsModel:
    """Return the chat model every agent in this project runs on.

    Raises a clear error rather than letting an empty key turn into a
    confusing 401 several layers deeper.
    """
    load_environment()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Put it in lead-desk/.env "
            "(that file is git-ignored)."
        )

    # AsyncOpenAI, not OpenAI: the Agents SDK's runner is asynchronous, so the
    # transport underneath it has to be awaitable too.
    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    return OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=client)
