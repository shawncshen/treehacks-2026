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


SYSTEM_PROMPT = f"""You suggest browser actions. Given a URL and interactive elements list, return a JSON array of {NUM_SUGGESTIONS} actions. No markdown fences.
Each: {{"id":N,"label":"short","action_type":"click|scroll|type|navigate|press_key","action_detail":{{...}},"description":"brief"}}
action_detail: click={{"element_id":N}}, scroll={{"direction":"up|down"}}, type={{"element_id":N,"text":"..."}}, navigate={{"url":"..."}}, press_key={{"key":"..."}}
Use element_id from the list. Prefer navigate with href when available. Include 1 scroll option. Most useful first."""


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
        # Build compact element list — only send top 30 most relevant
        el_lines = []
        for el in elements[:30]:
            line = f"[{el['id']}] <{el['tag']}> \"{el['text']}\""
            if el.get("href"):
                line += f" href={el['href']}"
            if el.get("type"):
                line += f" type={el['type']}"
            el_lines.append(line)
        elements_text = "\n".join(el_lines) if el_lines else "(no elements)"

        response = await self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"URL: {current_url}\n\n{elements_text}"},
            ],
            max_tokens=800,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return [
                Suggestion(0, "Scroll down", "scroll", {"direction": "down"}, "Scroll down")
            ]

        suggestions = []
        for item in items[:NUM_SUGGESTIONS]:
            suggestions.append(
                Suggestion(
                    id=item.get("id", len(suggestions)),
                    label=item.get("label", "Unknown"),
                    action_type=item.get("action_type", "scroll"),
                    action_detail=item.get("action_detail", {}),
                    description=item.get("description", ""),
                )
            )
        return suggestions
