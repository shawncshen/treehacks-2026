"""Pyppeteer browser controller for taking screenshots and executing actions."""

import asyncio
from pyppeteer import launch

from actions.config import BROWSER_HEADLESS, VIEWPORT_WIDTH, VIEWPORT_HEIGHT


class BrowserController:
    """Controls a Chromium browser via Pyppeteer."""

    def __init__(self):
        self._browser = None
        self._page = None

    async def launch(self):
        """Launch Chromium browser."""
        self._browser = await launch(
            headless=BROWSER_HEADLESS,
            args=[
                f"--window-size={VIEWPORT_WIDTH},{VIEWPORT_HEIGHT}",
                "--no-sandbox",
            ],
        )
        self._page = (await self._browser.pages())[0]
        await self._page.setViewport(
            {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        )

    async def screenshot(self) -> bytes:
        """Take a full-page screenshot, returned as PNG bytes."""
        return await self._page.screenshot({"type": "png"})

    async def click(self, selector: str):
        """Click an element by CSS selector."""
        await self._page.click(selector)

    async def click_coords(self, x: int, y: int):
        """Click at specific pixel coordinates."""
        await self._page.mouse.click(x, y)

    async def scroll(self, direction: str):
        """Scroll the page up or down."""
        delta = -300 if direction == "up" else 300
        await self._page.evaluate(f"window.scrollBy(0, {delta})")

    async def goto(self, url: str):
        """Navigate to a URL."""
        try:
            await self._page.goto(url, {"waitUntil": "domcontentloaded", "timeout": 15000})
        except Exception:
            # Some pages never fully settle; continue with whatever loaded
            pass
        # Brief wait for rendering
        await asyncio.sleep(1)

    async def type_text(self, selector: str, text: str):
        """Type text into an input element."""
        await self._page.type(selector, text)

    async def page_type(self, text: str):
        """Type text into whatever element is currently focused."""
        await self._page.keyboard.type(text)

    async def press_key(self, key: str):
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        await self._page.keyboard.press(key)

    async def get_url(self) -> str:
        """Get the current page URL."""
        return self._page.url

    async def close(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
