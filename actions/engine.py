"""Main action engine — instant DOM elements + background GPT suggestions."""

import asyncio

from actions.browser import BrowserController
from actions.vision import PageAnalyzer, Suggestion
from actions.overlay import Overlay
from actions.config import OPENAI_API_KEY


class ActionEngine:
    """Shows DOM elements instantly, then upgrades to GPT suggestions in background."""

    def __init__(self, overlay=None):
        self.browser = BrowserController()
        self.analyzer = PageAnalyzer(api_key=OPENAI_API_KEY)
        self.overlay = overlay if overlay is not None else Overlay()
        self._suggestions: list[Suggestion] = []
        self._elements: list[dict] = []
        self._selected: int = 0
        self._running: bool = False
        self._gpt_task: asyncio.Task | None = None

    async def start(self, url: str):
        self.overlay.set_status(False)
        await self.browser.launch()
        await self.browser.goto(url)
        self._running = True

    def _elements_to_suggestions(self, elements: list[dict]) -> list[Suggestion]:
        """Convert raw DOM elements into suggestions (instant, no API)."""
        suggestions = []
        for el in elements[:8]:
            if el.get("href"):
                action_type = "navigate"
                detail = {"url": el["href"], "element_id": el["id"]}
            elif el["tag"] in ("input", "textarea", "select") or el.get("type") in ("text", "search", "email", "password", "url"):
                action_type = "type"
                detail = {"element_id": el["id"], "text": ""}
            else:
                action_type = "click"
                detail = {"element_id": el["id"]}

            suggestions.append(Suggestion(
                id=len(suggestions),
                label=el["text"][:50],
                action_type=action_type,
                action_detail=detail,
                description=f"{action_type} <{el['tag']}>",
            ))

        suggestions.append(Suggestion(
            id=len(suggestions), label="Scroll Down", action_type="scroll",
            action_detail={"direction": "down"}, description="Scroll down",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Scroll Up", action_type="scroll",
            action_detail={"direction": "up"}, description="Scroll up",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Go Back", action_type="history",
            action_detail={"direction": "back"}, description="Previous page",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Go Forward", action_type="history",
            action_detail={"direction": "forward"}, description="Next page",
        ))
        return suggestions

    async def _fetch_gpt_suggestions(self, url: str, elements: list[dict]):
        """Background task: get GPT suggestions and update display."""
        try:
            gpt_suggestions = await self.analyzer.analyze(url, elements)
            if gpt_suggestions and self._running:
                self._suggestions = gpt_suggestions
                self._selected = min(self._selected, len(self._suggestions) - 1)
                self.overlay.show(self._suggestions, self._selected, smart=True)
        except Exception:
            pass  # Keep showing DOM elements if GPT fails

    async def run_cycle(self) -> list[Suggestion]:
        """Show DOM elements instantly, then upgrade with GPT in background."""
        # Cancel any pending GPT task
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()

        # RED — extracting elements
        self.overlay.set_status(False)

        # Step 1: Instant — extract elements and show immediately
        current_url = await self.browser.get_url()
        self._elements = await self.browser.get_interactive_elements()
        self._suggestions = self._elements_to_suggestions(self._elements)
        self._selected = 0

        # GREEN — user can interact now
        self.overlay.set_status(True)
        self.overlay.show(self._suggestions, self._selected, smart=False)

        # Step 2: Background — kick off GPT for smarter suggestions
        self._gpt_task = asyncio.ensure_future(
            self._fetch_gpt_suggestions(current_url, self._elements)
        )

        return self._suggestions

    def move_selection(self, direction: str):
        if not self._suggestions:
            return
        if direction == "up":
            self._selected = max(0, self._selected - 1)
        else:
            self._selected = min(len(self._suggestions) - 1, self._selected + 1)
        self.overlay.show(self._suggestions, self._selected)

    def select_index(self, index: int):
        if self._suggestions and 0 <= index < len(self._suggestions):
            self._selected = index
            self.overlay.show(self._suggestions, self._selected)

    async def execute_selected(self):
        if not self._suggestions:
            return
        # Cancel GPT if still running — we're moving on
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()

        # RED — executing action
        self.overlay.set_status(False)

        suggestion = self._suggestions[self._selected]
        await self.execute(suggestion)

    def _find_element(self, element_id: int) -> dict | None:
        for el in self._elements:
            if el["id"] == element_id:
                return el
        return None

    async def execute(self, suggestion: Suggestion):
        detail = suggestion.action_detail
        try:
            if suggestion.action_type == "click":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])

            elif suggestion.action_type == "type":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])
                text = detail.get("text", "")
                if text:
                    await self.browser.page_type(text)

            elif suggestion.action_type == "navigate":
                await self.browser.goto(detail.get("url", ""))

            elif suggestion.action_type == "scroll":
                await self.browser.scroll(detail.get("direction", "down"))

            elif suggestion.action_type == "press_key":
                await self.browser.press_key(detail["key"])

            elif suggestion.action_type == "history":
                if detail.get("direction") == "back":
                    await self.browser.go_back()
                else:
                    await self.browser.go_forward()

        except Exception as e:
            print(f"\n  Action failed: {e}", flush=True)

    async def stop(self):
        self._running = False
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()
        await self.browser.close()
        self.overlay.clear()
