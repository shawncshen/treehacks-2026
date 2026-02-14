"""Main action engine that wires browser, vision, and overlay together."""

import asyncio

from actions.browser import BrowserController
from actions.vision import PageAnalyzer, Suggestion
from actions.overlay import Overlay
from actions.config import OPENAI_API_KEY


class ActionEngine:
    """Core loop: screenshot -> vision API -> display suggestions -> execute."""

    def __init__(self):
        self.browser = BrowserController()
        self.analyzer = PageAnalyzer(api_key=OPENAI_API_KEY)
        self.overlay = Overlay()
        self._suggestions: list[Suggestion] = []
        self._selected: int = 0
        self._running: bool = False

    async def start(self, url: str):
        """Launch browser, navigate to URL, and enter the main loop."""
        await self.browser.launch()
        await self.browser.goto(url)
        self._running = True

    async def run_cycle(self) -> list[Suggestion]:
        """Run one screenshot -> analyze -> display cycle. Returns suggestions."""
        print("\n  Analyzing page...", flush=True)
        screenshot = await self.browser.screenshot()
        # Log screenshot size for debugging coordinate issues
        import struct
        if screenshot[:8] == b'\x89PNG\r\n\x1a\n':
            w = struct.unpack('>I', screenshot[16:20])[0]
            h = struct.unpack('>I', screenshot[20:24])[0]
            print(f"  Screenshot: {w}x{h}px ({len(screenshot)} bytes)", flush=True)
        current_url = await self.browser.get_url()
        self._suggestions = await self.analyzer.analyze(screenshot, current_url)
        self._selected = 0
        self.overlay.show(self._suggestions, self._selected)
        return self._suggestions

    def move_selection(self, direction: str):
        """Move the highlight up or down and redisplay."""
        if not self._suggestions:
            return
        if direction == "up":
            self._selected = max(0, self._selected - 1)
        else:
            self._selected = min(len(self._suggestions) - 1, self._selected + 1)
        self.overlay.show(self._suggestions, self._selected)

    def select_index(self, index: int):
        """Jump selection to a specific index."""
        if self._suggestions and 0 <= index < len(self._suggestions):
            self._selected = index
            self.overlay.show(self._suggestions, self._selected)

    async def execute_selected(self):
        """Execute the currently highlighted suggestion."""
        if not self._suggestions:
            return
        suggestion = self._suggestions[self._selected]
        await self.execute(suggestion)

    async def execute(self, suggestion: Suggestion):
        """Dispatch a suggestion to the browser."""
        detail = suggestion.action_detail
        print(f"\n  Executing: [{suggestion.action_type}] {suggestion.label}", flush=True)
        print(f"  Detail: {detail}", flush=True)
        try:
            if suggestion.action_type == "click":
                if "x" in detail and "y" in detail:
                    print(f"  Clicking at ({detail['x']}, {detail['y']})", flush=True)
                    await self.browser.click_coords(int(detail["x"]), int(detail["y"]))
                elif "selector" in detail:
                    await self.browser.click(detail["selector"])
            elif suggestion.action_type == "type":
                if "x" in detail and "y" in detail:
                    print(f"  Clicking input at ({detail['x']}, {detail['y']})", flush=True)
                    await self.browser.click_coords(int(detail["x"]), int(detail["y"]))
                    await asyncio.sleep(0.3)
                if "selector" in detail:
                    await self.browser.type_text(detail["selector"], detail["text"])
                else:
                    await self.browser.page_type(detail["text"])
            elif suggestion.action_type == "navigate":
                print(f"  Navigating to: {detail['url']}", flush=True)
                await self.browser.goto(detail["url"])
            elif suggestion.action_type == "scroll":
                await self.browser.scroll(detail.get("direction", "down"))
            elif suggestion.action_type == "press_key":
                await self.browser.press_key(detail["key"])
            print("  Done!", flush=True)
        except Exception as e:
            print(f"\n  Action failed: {e}", flush=True)
        # Brief pause to let the page update
        await asyncio.sleep(0.5)

    async def stop(self):
        """Clean up browser resources."""
        self._running = False
        await self.browser.close()
        self.overlay.clear()
