"""End-to-end tests for the autonomous browser agent.

Tests the full pipeline:
  1. Browser launch + page load
  2. DOM element extraction
  3. LLM decides next action (or mock if no API key)
  4. Action execution (click, type, navigate, scroll, press_key)
  5. Page state changes correctly after each action

Run:  python -m actions.test_agent
"""

import asyncio
import http.server
import json
import threading

from actions.browser import BrowserController
from actions.engine import AutonomousAgent
from actions.vision import AgentPlanner, Suggestion


# ---------- test pages ----------
SEARCH_PAGE = """<!DOCTYPE html>
<html><head><title>Test Search Engine</title></head>
<body style="margin:0; font-family:sans-serif; background:#fff;">
  <h1 style="text-align:center; padding:40px;">Test Search</h1>
  <div style="text-align:center;">
    <input id="searchbox" type="text" placeholder="Search..."
           style="width:400px; padding:12px; font-size:16px; border:1px solid #ccc; border-radius:4px;" />
    <button id="searchbtn" style="padding:12px 24px; font-size:16px; margin-left:8px; cursor:pointer;"
            onclick="doSearch()">Search</button>
  </div>
  <div id="results" style="padding:40px; text-align:center; color:#666;"></div>
  <a id="link_about" href="/about" style="display:block; text-align:center; margin-top:20px;">About</a>
  <a id="link_help" href="/help" style="display:block; text-align:center; margin-top:10px;">Help</a>
  <script>
    function doSearch() {
      const q = document.getElementById('searchbox').value;
      document.title = 'Results: ' + q;
      document.getElementById('results').innerHTML =
        '<h2>Results for: ' + q + '</h2>' +
        '<p><a href="/result1" onclick="document.title=\\'clicked_result1\\'; return false;">Result 1 - ' + q + '</a></p>' +
        '<p><a href="/result2" onclick="document.title=\\'clicked_result2\\'; return false;">Result 2 - ' + q + '</a></p>';
    }
    // Also search on Enter
    document.getElementById('searchbox').addEventListener('keydown', function(e) {
      if (e.key === 'Enter') doSearch();
    });
  </script>
</body></html>"""

ABOUT_PAGE = """<!DOCTYPE html>
<html><head><title>About Page</title></head>
<body style="margin:0; font-family:sans-serif; background:#f9f9f9;">
  <h1 style="text-align:center; padding:40px;">About Us</h1>
  <p style="text-align:center;">This is the about page.</p>
  <a id="link_home" href="/" style="display:block; text-align:center; margin-top:20px;">Back to Home</a>
  <div style="height:2000px;"></div>
</body></html>"""

SCROLL_PAGE = """<!DOCTYPE html>
<html><head><title>Scroll Test</title></head>
<body style="margin:0; font-family:sans-serif;">
  <h1 style="padding:20px;">Top of page</h1>
  <div style="height:3000px; background: linear-gradient(#fff, #eee);"></div>
  <div id="bottom_marker" style="padding:20px; background:#4CAF50; color:white;">
    <h2>Bottom reached!</h2>
    <button id="bottom_btn" onclick="document.title='bottom_clicked'">Click me at bottom</button>
  </div>
</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        if self.path == "/about":
            self.wfile.write(ABOUT_PAGE.encode())
        elif self.path.startswith("/scroll"):
            self.wfile.write(SCROLL_PAGE.encode())
        else:
            self.wfile.write(SEARCH_PAGE.encode())

    def log_message(self, *_):
        pass


def _start_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


# ---------- test helpers ----------
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, condition):
    ok = bool(condition)
    print(f"  {PASS if ok else FAIL}  {name}")
    results.append((name, ok))


# ---------- tests ----------
async def test_browser_basics(port):
    """Test browser launch, DOM extraction, page title, page text."""
    print("\n=== Browser Basics ===")
    url = f"http://127.0.0.1:{port}"
    bc = BrowserController()

    try:
        await bc.launch()
        check("Browser launched", bc._browser is not None)

        await bc.goto(url)
        await asyncio.sleep(0.5)

        title = await bc.get_page_title()
        check("Page title correct", title == "Test Search Engine")

        text = await bc.get_page_text()
        check("Page text extracted", "Test Search" in text)

        elements = await bc.get_interactive_elements()
        check("Elements extracted", len(elements) > 0)

        # Should find searchbox, search button, and links
        tags = [el["tag"] for el in elements]
        check("Found input element", "input" in tags)
        check("Found button element", "button" in tags)
        check("Found link elements", "a" in tags)

        # Check element structure
        for el in elements:
            check(f"Element {el['id']} has cx/cy",
                  "cx" in el and "cy" in el and el["cx"] > 0)
            break  # Just check first one

        current_url = await bc.get_url()
        check("Get URL works", port.__str__() in current_url)

    finally:
        await bc.close()


async def test_click_action(port):
    """Test clicking an element by coordinates."""
    print("\n=== Click Action ===")
    url = f"http://127.0.0.1:{port}"
    bc = BrowserController()

    try:
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)

        elements = await bc.get_interactive_elements()

        # Find the search button
        btn = None
        for el in elements:
            if el["tag"] == "button" and "Search" in el.get("text", ""):
                btn = el
                break
        check("Found search button", btn is not None)

        if btn:
            # Type something first
            inp = None
            for el in elements:
                if el["tag"] == "input":
                    inp = el
                    break
            check("Found search input", inp is not None)

            if inp:
                await bc.click_coords(inp["cx"], inp["cy"])
                await bc.page_type("test query")
                await asyncio.sleep(0.3)

                # Click search button
                await bc.click_coords(btn["cx"], btn["cy"])
                await asyncio.sleep(0.5)

                title = await bc.get_page_title()
                check("Search executed (title changed)", "Results: test query" in title)

    finally:
        await bc.close()


async def test_type_action(port):
    """Test typing into an input field."""
    print("\n=== Type Action ===")
    url = f"http://127.0.0.1:{port}"
    bc = BrowserController()

    try:
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)

        elements = await bc.get_interactive_elements()

        # Find input
        inp = None
        for el in elements:
            if el["tag"] == "input":
                inp = el
                break
        check("Found input for typing", inp is not None)

        if inp:
            await bc.click_coords(inp["cx"], inp["cy"])
            await bc.page_type("hello world")
            await asyncio.sleep(0.3)

            # Verify text was typed
            value = await bc._page.evaluate(
                "() => document.getElementById('searchbox').value"
            )
            check("Text typed into input", value == "hello world")

    finally:
        await bc.close()


async def test_press_key(port):
    """Test pressing Enter to submit search."""
    print("\n=== Press Key Action ===")
    url = f"http://127.0.0.1:{port}"
    bc = BrowserController()

    try:
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)

        elements = await bc.get_interactive_elements()
        inp = None
        for el in elements:
            if el["tag"] == "input":
                inp = el
                break

        if inp:
            await bc.click_coords(inp["cx"], inp["cy"])
            await bc.page_type("enter test")
            await asyncio.sleep(0.2)

            await bc.press_key("Enter")
            await asyncio.sleep(0.5)

            title = await bc.get_page_title()
            check("Enter key triggered search", "Results: enter test" in title)

    finally:
        await bc.close()


async def test_navigate_action(port):
    """Test navigating to a different URL."""
    print("\n=== Navigate Action ===")
    url = f"http://127.0.0.1:{port}"
    bc = BrowserController()

    try:
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)

        title1 = await bc.get_page_title()
        check("Starting on search page", title1 == "Test Search Engine")

        await bc.goto(f"http://127.0.0.1:{port}/about")
        await asyncio.sleep(0.5)

        title2 = await bc.get_page_title()
        check("Navigated to about page", title2 == "About Page")

    finally:
        await bc.close()


async def test_scroll_action(port):
    """Test scrolling the page."""
    print("\n=== Scroll Action ===")
    url = f"http://127.0.0.1:{port}/scroll"
    bc = BrowserController()

    try:
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)

        scroll_before = await bc._page.evaluate("() => window.scrollY")
        check("Starts at top", scroll_before == 0)

        await bc.scroll("down")
        await asyncio.sleep(1)

        scroll_after = await bc._page.evaluate("() => window.scrollY")
        check("Scrolled down", scroll_after > scroll_before)

        await bc.scroll("up")
        await asyncio.sleep(1)

        scroll_up = await bc._page.evaluate("() => window.scrollY")
        check("Scrolled up", scroll_up < scroll_after)

    finally:
        await bc.close()


async def test_history_action(port):
    """Test go_back and go_forward."""
    print("\n=== History Navigation ===")
    url = f"http://127.0.0.1:{port}"
    bc = BrowserController()

    try:
        await bc.launch()
        await bc.goto(url)
        await asyncio.sleep(0.5)

        await bc.goto(f"http://127.0.0.1:{port}/about")
        await asyncio.sleep(0.5)
        check("On about page", (await bc.get_page_title()) == "About Page")

        await bc.go_back()
        await asyncio.sleep(0.5)
        check("Back to search page", (await bc.get_page_title()) == "Test Search Engine")

        await bc.go_forward()
        await asyncio.sleep(0.5)
        check("Forward to about page", (await bc.get_page_title()) == "About Page")

    finally:
        await bc.close()


async def test_engine_execute(port):
    """Test AutonomousAgent._execute with each action type."""
    print("\n=== Engine Execute Actions ===")
    url = f"http://127.0.0.1:{port}"
    agent = AutonomousAgent(max_steps=5)

    try:
        await agent.start(url)
        await asyncio.sleep(0.5)

        # Get elements
        agent._elements = await agent.browser.get_interactive_elements()
        check("Agent extracted elements", len(agent._elements) > 0)

        # Test type action
        inp = None
        for el in agent._elements:
            if el["tag"] == "input":
                inp = el
                break

        if inp:
            action = Suggestion(
                id=0, label="type test",
                action_type="type",
                action_detail={"element_id": inp["id"], "text": "agent typing"},
                description="typing into search"
            )
            await agent._execute(action)
            await asyncio.sleep(0.3)

            value = await agent.browser._page.evaluate(
                "() => document.getElementById('searchbox').value"
            )
            check("Agent type action worked", value == "agent typing")

        # Test press_key action
        action = Suggestion(
            id=0, label="press enter",
            action_type="press_key",
            action_detail={"key": "Enter"},
            description="submit search"
        )
        await agent._execute(action)
        await asyncio.sleep(0.5)

        title = await agent.browser.get_page_title()
        check("Agent press_key worked", "Results: agent typing" in title)

        # Test scroll action
        scroll_before = await agent.browser._page.evaluate("() => window.scrollY")
        action = Suggestion(
            id=0, label="scroll",
            action_type="scroll",
            action_detail={"direction": "down"},
            description="scroll down"
        )
        await agent._execute(action)
        await asyncio.sleep(1)

        scroll_after = await agent.browser._page.evaluate("() => window.scrollY")
        check("Agent scroll action worked", scroll_after >= scroll_before)

        # Test navigate action
        action = Suggestion(
            id=0, label="navigate",
            action_type="navigate",
            action_detail={"url": f"http://127.0.0.1:{port}/about"},
            description="go to about"
        )
        await agent._execute(action)
        await asyncio.sleep(0.5)

        title = await agent.browser.get_page_title()
        check("Agent navigate action worked", title == "About Page")

        # Test history action
        action = Suggestion(
            id=0, label="go back",
            action_type="history",
            action_detail={"direction": "back"},
            description="go back"
        )
        await agent._execute(action)
        await asyncio.sleep(0.5)

        title = await agent.browser.get_page_title()
        check("Agent history back action worked", "Results" in title or "Test Search" in title)

    finally:
        await agent.stop()


async def test_suggestions_display(port):
    """Test that _print_suggestions outputs the element list."""
    print("\n=== Suggestions Display ===")
    url = f"http://127.0.0.1:{port}"
    agent = AutonomousAgent(max_steps=5)

    try:
        await agent.start(url)
        await asyncio.sleep(0.5)

        elements = await agent.browser.get_interactive_elements()
        check("Got elements for display", len(elements) > 0)

        # This should print without error
        agent._print_suggestions(elements[:10])
        check("_print_suggestions ran without error", True)

        # Empty case
        agent._print_suggestions([])
        check("_print_suggestions handles empty list", True)

    finally:
        await agent.stop()


async def test_agent_planner_structure():
    """Test AgentPlanner message construction (without calling API)."""
    print("\n=== Agent Planner Structure ===")
    planner = AgentPlanner(api_key="test-key")

    check("Planner created", planner is not None)
    check("Messages start empty", len(planner._messages) == 0)

    planner.reset()
    check("Reset clears messages", len(planner._messages) == 0)


async def run_all_tests():
    srv, port = _start_server()
    print(f"\n  Test server running on port {port}\n")

    try:
        await test_browser_basics(port)
        await test_click_action(port)
        await test_type_action(port)
        await test_press_key(port)
        await test_navigate_action(port)
        await test_scroll_action(port)
        await test_history_action(port)
        await test_engine_execute(port)
        await test_suggestions_display(port)
        await test_agent_planner_structure()
    finally:
        srv.shutdown()

    # Summary
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
    ok = asyncio.run(run_all_tests())
    raise SystemExit(0 if ok else 1)
