"""Use OpenAI to pick the best word from EMG class sequence candidates given page context."""

import asyncio

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL


async def pick_best_word(
    candidates: list[str],
    page_context: str,
    previously_added_words: list[str],
    api_key: str | None = None,
) -> str | None:
    """Pick the best word from candidates given webpage context and previously added words.

    Uses a low-cost model (gpt-4o-mini) for cheap inference.
    Returns the best match, or None if no candidates.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
    prev = " ".join(previously_added_words) if previously_added_words else "(none yet)"
    prompt = f"""You are helping a user who is typing via thought-controlled EMG input. They selected a sequence of mouth-shape classes (OPEN, CLOSE, TIP, BACK, LIPS) which maps to multiple possible English words.

Page context (what's visible on the webpage):
{page_context[:1500]}

Previously added words in the prompt so far: {prev}

Possible words from the EMG sequence: {', '.join(candidates)}

Pick the single most likely word the user intended. Consider:
- The webpage context (search queries, forms, links, etc.)
- The previously added words (the full prompt they're building)
- Common English usage

Reply with ONLY the word, nothing else. Lowercase."""

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Validate it's one of our candidates
        for c in candidates:
            if c.lower() == raw:
                return c
        return candidates[0]
    except Exception:
        return candidates[0]
