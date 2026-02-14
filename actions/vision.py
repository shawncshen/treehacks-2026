"""OpenAI API integration for analyzing page elements and suggesting actions."""

import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL, NUM_SUGGESTIONS


@dataclass
class Suggestion:
    """A single suggested browser action."""

    id: int
    label: str
    action_type: str  # click, scroll, type, navigate, press_key
    action_detail: dict = field(default_factory=dict)
    description: str = ""


SYSTEM_PROMPT = f"""Return JSON array of {NUM_SUGGESTIONS} browser actions. No markdown.
Format: [{{"id":0,"label":"short","action_type":"click|scroll|type|navigate|press_key","action_detail":{{}},"description":"brief"}}]
action_detail: click={{"element_id":N}}, scroll={{"direction":"up|down"}}, type={{"element_id":N,"text":"..."}}, navigate={{"url":"..."}}, press_key={{"key":"..."}}
Use element_id from list. Prefer navigate+href. 1 scroll. Useful first."""


class PageAnalyzer:
    """Sends element list to OpenAI and gets suggested actions."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self._client = AsyncOpenAI(api_key=api_key)

    async def analyze(
        self,
        current_url: str,
        elements: list[dict],
    ) -> list[Suggestion]:
        """Analyze page elements and return suggested actions."""
        # Compact element list — top 20 only, minimal text
        el_parts = []
        for el in elements[:20]:
            p = f"[{el['id']}]{el['tag']}:\"{el['text']}\""
            if el.get("href"):
                p += f" h={el['href']}"
            el_parts.append(p)
        el_text = "\n".join(el_parts) if el_parts else "(none)"

        response = await self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{current_url}\n{el_text}"},
            ],
            max_tokens=600,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return [Suggestion(0, "Scroll down", "scroll", {"direction": "down"}, "Scroll")]

        return [
            Suggestion(
                id=item.get("id", i),
                label=item.get("label", "Unknown"),
                action_type=item.get("action_type", "scroll"),
                action_detail=item.get("action_detail", {}),
                description=item.get("description", ""),
            )
            for i, item in enumerate(items[:NUM_SUGGESTIONS])
        ]
