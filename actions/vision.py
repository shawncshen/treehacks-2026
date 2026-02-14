"""LLM-powered agent planner for autonomous browser automation."""

import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class Suggestion:
    """A single browser action."""

    id: int
    label: str
    action_type: str  # click, scroll, type, navigate, press_key, done
    action_detail: dict = field(default_factory=dict)
    description: str = ""


SYSTEM_PROMPT = """You are an autonomous browser automation agent. You control a browser to accomplish user goals.

Given:
- The user's goal
- Current page URL and title
- Interactive elements on the page (with IDs, text, types, positions)
- History of actions you've already taken

Decide the SINGLE next action. Return ONLY a JSON object (no markdown, no extra text):
{"action_type": "click|type|scroll|navigate|press_key|done", "action_detail": {}, "reasoning": "brief explanation"}

action_detail formats:
- click: {"element_id": N}
- type: {"element_id": N, "text": "what to type"}  (clicks the element first, then types)
- scroll: {"direction": "up|down"}
- navigate: {"url": "https://..."}
- press_key: {"key": "Enter|Tab|Escape|..."}
- done: {"summary": "what was accomplished"}

Rules:
1. Take exactly ONE action per step.
2. Use element_id from the provided element list — pick the best match.
3. For search workflows: first type into the search box, then in the next step press_key Enter.
4. Return "done" when the goal is fully accomplished or truly cannot be completed.
5. Be efficient — minimize unnecessary steps.
6. If the page doesn't have what you need, navigate to the right URL directly.
7. When you need to type into a field, always include the element_id of the input/textarea.
8. If you get stuck or see the same page repeatedly, try a different approach."""


class AgentPlanner:
    """Decides the next action to take given goal, page state, and history."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self._client = AsyncOpenAI(api_key=api_key)
        self._messages: list[dict] = []

    def reset(self):
        """Clear conversation history for a new goal."""
        self._messages = []

    async def decide_next_action(
        self,
        goal: str,
        current_url: str,
        page_title: str,
        elements: list[dict],
        step_number: int,
    ) -> Suggestion | None:
        """Decide the next action based on current page state.

        Returns a Suggestion to execute, or a Suggestion with action_type="done"
        when the goal is complete. Returns None on LLM failure.
        """
        # Build compact element list
        el_parts = []
        for el in elements[:40]:
            p = f"[{el['id']}] {el['tag']}: \"{el['text']}\""
            if el.get("href"):
                p += f" href={el['href']}"
            if el.get("type"):
                p += f" type={el['type']}"
            el_parts.append(p)
        el_text = "\n".join(el_parts) if el_parts else "(no interactive elements found)"

        user_content = f"""Goal: {goal}

Step: {step_number}
URL: {current_url}
Page Title: {page_title}

Interactive elements:
{el_text}

What is the single next action?"""

        # Build messages — system + conversation history + current state
        if not self._messages:
            self._messages.append({"role": "system", "content": SYSTEM_PROMPT})

        self._messages.append({"role": "user", "content": user_content})

        try:
            response = await self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self._messages,
                max_tokens=300,
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()

            # Save assistant response to history
            self._messages.append({"role": "assistant", "content": raw})

            # Trim history to avoid context overflow (keep system + last 20 turns)
            if len(self._messages) > 41:
                self._messages = [self._messages[0]] + self._messages[-40:]

            # Parse JSON
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]

            action = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return None

        action_type = action.get("action_type", "done")
        action_detail = action.get("action_detail", {})
        reasoning = action.get("reasoning", "")

        return Suggestion(
            id=0,
            label=reasoning,
            action_type=action_type,
            action_detail=action_detail,
            description=reasoning,
        )
