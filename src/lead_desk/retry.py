"""Retrying around the Gemini free-tier rate limit.

WHY THIS EXISTS
The free tier returns 429 RESOURCE_EXHAUSTED on a rolling window, and a full
six-lead run costs two requests per lead. Without this, a demo stops halfway
through for a reason that has nothing to do with the agent being wrong.

Google puts the wait it wants in the error body, twice:
  - "Please retry in 11.427513434s."      (in the message)
  - {"@type": ".../RetryInfo", "retryDelay": "11s"}   (in details)
Honouring the server's own number is better than a fixed backoff: it is
neither too eager (which just burns another request) nor too patient.
"""

import asyncio
import re

from openai import RateLimitError

# Matches "Please retry in 11.427513434s." and "retryDelay': '11s'".
_DELAY_PATTERNS = (
    re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE),
    re.compile(r"retryDelay['\"]?:\s*['\"]?([0-9.]+)s", re.IGNORECASE),
)

DEFAULT_WAIT_SECONDS = 20.0
MAX_WAIT_SECONDS = 65.0


def suggested_wait(exc: RateLimitError) -> float:
    """Read the server's own retry hint out of the error, in seconds.

    Falls back to a fixed wait when the hint is missing, and caps the result
    so a bad parse cannot stall a live demo for minutes.
    """
    text = str(exc)
    for pattern in _DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            # +1s of slack: retrying at exactly the boundary tends to 429 again.
            return min(float(match.group(1)) + 1.0, MAX_WAIT_SECONDS)
    return DEFAULT_WAIT_SECONDS


async def with_rate_limit_retry(coro_factory, attempts: int = 3):
    """Await coro_factory(), retrying on 429 using the server's suggested wait.

    Takes a *factory* rather than a coroutine because a coroutine object can
    only be awaited once - retrying needs a fresh one each time. Any exception
    other than RateLimitError propagates untouched: this is a throttling
    helper, not a blanket try/except that would hide real bugs.
    """
    last: RateLimitError | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except RateLimitError as exc:
            last = exc
            if attempt == attempts:
                break
            wait = suggested_wait(exc)
            print(
                f"  rate limited (attempt {attempt}/{attempts}) - "
                f"waiting {wait:.0f}s as the API asked, then retrying"
            )
            await asyncio.sleep(wait)

    assert last is not None
    raise last
