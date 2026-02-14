"""Automated tests for the animated cursor and browser actions.

Launches Chromium (non-headless, since pyppeteer headless is unreliable on macOS)
against a local HTML test page, then verifies:
  - Cursor injection & centering
  - Smooth movement to coordinates
  - Cursor persistence after DOM removal
  - Click sequencing (click only fires AFTER cursor arrives)
  - Hide / show
  - Navigation persistence
  - Scroll with cursor
  - Callback firing

Run:  python -m actions.test_cursor
"""

import asyncio
import http.server
import threading

from actions.browser import BrowserController
from actions.cursor import PageCursor

# ---------- tiny test page served locally ----------
TEST_HTML = """<!DOCTYPE html>
<html><head><title>initial</title></head>
<body style="margin:0; height:3000px; font-family:sans-serif; background:#f0f0f0;">
  <h2 style="text-align:center;padding:20px;">Cursor Test Page</h2>
  <button id="btn1" style="position:absolute;left:200px;top:150px;padding:12px 24px;font-size:16px;"
          onclick="document.title='btn1_clicked'">Button 1</button>
  <button id="btn2" style="position:absolute;left:600px;top:400px;padding:12px 24px;font-size:16px;"
          onclick="document.title='btn2_clicked'">Button 2</button>
  <a id="link1" href="javascript:void(0)" style="position:absolute;left:400px;top:300px;font-size:16px;"
     onclick="document.title='link1_clicked'">Test Link</a>
</body></html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(TEST_HTML.encode())

    def log_message(self, *_):
        pass


def _start_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


# ---------- helpers ----------
async def _cursor_state(page):
    return await page.evaluate("""() => {
        const c = document.getElementById('__sp_cursor');
        if (!c) return {exists: false};
        return {
            exists: true,
            left: parseFloat(c.style.left) || 0,
            top: parseFloat(c.style.top) || 0,
            display: c.style.display || '',
        };
    }""")


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, condition):
    ok = bool(condition)
    print(f"  {PASS if ok else FAIL}  {name}")
    results.append((name, ok))


# ---------- tests ----------
async def run_tests():
    srv, port = _start_server()
    url = f"http://127.0.0.1:{port}"

    # Use a fresh BrowserController (same as prod, non-headless)
    callback_log = []
    bc = BrowserController(on_cursor_move=lambda x, y, a: callback_log.append((x, y, a)))

    try:
        print("\n  Launching browser for tests...", flush=True)
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)
        page = bc._page
        cursor = bc._cursor
        print("  Browser ready.\n", flush=True)

        # ===== INJECTION =====
        print("=== Cursor Injection Tests ===")
        state = await _cursor_state(page)
        check("Cursor element exists", state["exists"])

        ripple = await page.evaluate("() => !!document.getElementById('__sp_ripple')")
        check("Ripple element exists", ripple)

        vw, vh = await bc.get_viewport_size()
        check("Cursor near center X", abs(state["left"] - vw // 2) < 50)
        check("Cursor near center Y", abs(state["top"] - vh // 2) < 50)

        # No duplicates
        await cursor._inject_cursor()
        count = await page.evaluate("() => document.querySelectorAll('#__sp_cursor').length")
        check("No duplicate cursor on re-inject", count == 1)

        # ===== MOVEMENT =====
        print("\n=== Cursor Movement Tests ===")
        await cursor.move_to(200, 150, duration_ms=150)
        state = await _cursor_state(page)
        check("Moved to x=200", abs(state["left"] - 200) < 5)
        check("Moved to y=150", abs(state["top"] - 150) < 5)

        await cursor.move_to(600, 400, duration_ms=150)
        state = await _cursor_state(page)
        check("Moved to x=600", abs(state["left"] - 600) < 5)
        check("Moved to y=400", abs(state["top"] - 400) < 5)
        check("Internal _x=600", cursor._x == 600)
        check("Internal _y=400", cursor._y == 400)

        # ===== PERSISTENCE =====
        print("\n=== Cursor Persistence Tests ===")
        await page.evaluate("() => { document.getElementById('__sp_cursor')?.remove(); }")
        check("Cursor removed from DOM", not (await _cursor_state(page))["exists"])

        await cursor.ensure_alive()
        state = await _cursor_state(page)
        check("ensure_alive re-injected cursor", state["exists"])
        check("Re-injected at last pos x=600", abs(state["left"] - 600) < 5)

        # move_to also re-injects
        await page.evaluate("() => { document.getElementById('__sp_cursor')?.remove(); }")
        await cursor.move_to(300, 250, duration_ms=50)
        state = await _cursor_state(page)
        check("move_to re-injects if missing", state["exists"])

        # ===== CLICK SEQUENCING =====
        print("\n=== Click Sequencing Tests ===")
        await page.evaluate("() => { document.title = 'initial'; }")

        # Move cursor to btn1 area — should NOT click yet
        await cursor.move_to(212, 162, duration_ms=200)
        title = await page.evaluate("() => document.title")
        check("No click during cursor move", title == "initial")

        # Click effect — visual only, still no real click
        await cursor.click_effect()
        title = await page.evaluate("() => document.title")
        check("click_effect is visual only", title == "initial")

        # Real click via BrowserController
        await page.mouse.click(212, 162)
        await asyncio.sleep(0.1)
        title = await page.evaluate("() => document.title")
        check("Real click fires btn1", title == "btn1_clicked")

        # Full sequence via click_coords (cursor + effect + click)
        await page.evaluate("() => { document.title = 'reset'; }")
        await bc.click_coords(612, 412)
        await asyncio.sleep(0.1)
        title = await page.evaluate("() => document.title")
        check("click_coords full sequence fires btn2", title == "btn2_clicked")

        # Verify cursor is at btn2 position after click_coords
        state = await _cursor_state(page)
        check("Cursor at btn2 after click_coords", abs(state["left"] - 612) < 5)

        # ===== HIDE / SHOW =====
        print("\n=== Hide / Show Tests ===")
        await cursor.hide()
        state = await _cursor_state(page)
        check("Cursor hidden", state["display"] == "none")

        await cursor.show()
        state = await _cursor_state(page)
        check("Cursor shown", state["display"] == "")

        # ===== SCROLL =====
        print("\n=== Scroll Tests ===")
        scroll_before = await page.evaluate("() => window.scrollY")
        await bc.scroll("down")
        await asyncio.sleep(1)
        scroll_after = await page.evaluate("() => window.scrollY")
        check("Scroll down moved page", scroll_after > scroll_before)

        # Cursor should have moved to center before scroll
        state = await _cursor_state(page)
        vw, vh = await bc.get_viewport_size()
        check("Cursor near center X after scroll", abs(state["left"] - vw // 2) < 50)
        check("Cursor near center Y after scroll", abs(state["top"] - vh // 2) < 50)

        # ===== NAVIGATION =====
        print("\n=== Navigation Tests ===")
        await bc.goto(url)
        await asyncio.sleep(0.5)
        state = await _cursor_state(page)
        check("Cursor exists after navigation", state["exists"])
        check("Cursor re-centered after nav", abs(state["left"] - 640) < 50)

        # ===== CALLBACKS =====
        print("\n=== Callback Tests ===")
        check("Callbacks fired", len(callback_log) > 0)
        check("Has move callback", any(a == "move" for _, _, a in callback_log))
        check("Has click callback", any(a == "click" for _, _, a in callback_log))

    finally:
        await bc.close()
        srv.shutdown()

    # ---------- summary ----------
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n{'='*44}")
    if failed:
        print(f"  {passed}/{total} passed  ({failed} FAILED)")
        for name, ok in results:
            if not ok:
                print(f"    FAILED: {name}")
    else:
        print(f"  {passed}/{total} passed — ALL PASSED!")
    print()
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    raise SystemExit(0 if ok else 1)
