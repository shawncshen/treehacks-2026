"""OpenAI Vision API integration for analyzing screenshots and suggesting actions."""

import base64
import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL, NUM_SUGGESTIONS, VIEWPORT_WIDTH, VIEWPORT_HEIGHT


@dataclass
class Suggestion:
    """A single suggested browser action."""

    id: int
    label: str
    action_type: str  # click, scroll, type, navigate, press_key
    action_detail: dict = field(default_factory=dict)
    description: str = ""


SYSTEM_PROMPT = f"""\
You are a browser automation assistant. The user will send you a screenshot of a web page and the current URL.

Analyze the page and return exactly {NUM_SUGGESTIONS} suggested next actions the user might want to take.

Return ONLY a JSON array (no markdown, no code fences) where each element has:
- "id": integer 0-{NUM_SUGGESTIONS - 1}
- "label": short human-readable label (max 60 chars)
- "action_type": one of "click", "scroll", "type", "navigate", "press_key"
- "action_detail": object with keys depending on action_type:
  - click: ALWAYS use pixel coordinates {{"x": pixel_x, "y": pixel_y}} based on where the element appears in the screenshot. The screenshot is {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT} pixels.
  - scroll: {{"direction": "up" or "down"}}
  - type: {{"x": pixel_x, "y": pixel_y, "text": "text to type"}} — coordinates of the input field to click first, then text to type
  - navigate: {{"url": "full URL"}}
  - press_key: {{"key": "Enter" or other key name}}
- "description": brief explanation of what this action does

IMPORTANT: For click and type actions, ALWAYS use pixel coordinates (x, y) estimated from the screenshot. Do NOT use CSS selectors.
Prioritize the most useful and common actions first. Include a mix of action types when appropriate."""


class PageAnalyzer:
    """Sends screenshots to OpenAI Vision API and parses suggested actions."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self._client = AsyncOpenAI(api_key=api_key)

    async def analyze(
        self, screenshot_bytes: bytes, current_url: str
    ) -> list[Suggestion]:
        """Analyze a screenshot and return a list of suggested actions."""
        b64_image = base64.b64encode(screenshot_bytes).decode("utf-8")

        response = await self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Current URL: {current_url}\n\nHere is a screenshot of the page. Suggest {NUM_SUGGESTIONS} actions.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return [
                Suggestion(
                    id=0,
                    label="Scroll down",
                    action_type="scroll",
                    action_detail={"direction": "down"},
                    description="Scroll down the page (fallback)",
                )
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
