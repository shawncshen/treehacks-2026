/**
 * Playwright-backed browser tools for the MCP server.
 *
 * 8 tools intentionally kept small for model reliability:
 * - browser_goto: navigate to URL
 * - browser_click: click an element by selector
 * - browser_type: type text into an element
 * - browser_press: press a keyboard key
 * - browser_scroll: scroll the page
 * - browser_screenshot: capture viewport as base64 PNG
 * - browser_extract_text: get visible text content
 * - browser_wait: wait for a duration
 */

import { chromium, Browser, Page } from "playwright";
import { PageCursor } from "./cursor.js";

let browser: Browser | null = null;
let page: Page | null = null;
let cursor: PageCursor | null = null;

const VIEWPORT = { width: 1280, height: 800 };

export async function ensureBrowser(): Promise<Page> {
  if (!page || !browser?.isConnected()) {
    if (browser) {
      try { await browser.close(); } catch {}
    }

    const headless = process.env.BROWSER_HEADLESS !== "false";

    browser = await chromium.launch({
      headless,
      args: [
        "--disable-extensions",
        "--disable-file-system",
        "--no-sandbox",
      ],
    });

    const context = await browser.newContext({
      viewport: VIEWPORT,
    });
    page = await context.newPage();
    cursor = new PageCursor(page);
    await cursor.attach();
  } else {
    // Ensure cursor is still in DOM (handles navigations)
    if (cursor) await cursor.ensureAlive();
  }
  return page;
}

export async function closeBrowser(): Promise<void> {
  if (browser) {
    await browser.close();
    browser = null;
    page = null;
  }
}

// --- Tool Implementations ---

export async function browserGoto(url: string): Promise<string> {
  const p = await ensureBrowser();
  if (cursor) await cursor.hide();
  await p.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
  if (cursor) {
    await cursor.attach();
    await cursor.show();
  }
  return `Navigated to ${p.url()}`;
}

export async function browserClick(selector: string): Promise<string> {
  const p = await ensureBrowser();
  try {
    const loc = p.locator(selector);
    const box = await loc.boundingBox();
    if (box && cursor) {
      await cursor.moveTo(box.x + box.width / 2, box.y + box.height / 2);
      await cursor.clickEffect();
    }
    await loc.click({ timeout: 5000 });
    return `Clicked: ${selector}`;
  } catch (e) {
    return `Failed to click '${selector}': ${e}`;
  }
}

export async function browserType(
  selector: string,
  text: string
): Promise<string> {
  const p = await ensureBrowser();
  try {
    await p.fill(selector, text, { timeout: 5000 });
    return `Typed into ${selector}: "${text}"`;
  } catch (e) {
    return `Failed to type into '${selector}': ${e}`;
  }
}

export async function browserPress(key: string): Promise<string> {
  const p = await ensureBrowser();
  await p.keyboard.press(key);
  return `Pressed key: ${key}`;
}

export async function browserScroll(
  direction: "up" | "down",
  amount: number = 300
): Promise<string> {
  const p = await ensureBrowser();
  if (cursor) {
    await cursor.moveTo(VIEWPORT.width / 2, VIEWPORT.height / 2);
  }
  const delta = direction === "down" ? amount : -amount;
  await p.mouse.wheel(0, delta);
  return `Scrolled ${direction} by ${amount}px`;
}

export async function browserScreenshot(): Promise<string> {
  const p = await ensureBrowser();
  const buffer = await p.screenshot({ type: "png" });
  return buffer.toString("base64");
}

export async function browserExtractText(): Promise<string> {
  const p = await ensureBrowser();
  const text = await p.evaluate(() => {
    return document.body?.innerText?.substring(0, 5000) || "";
  });
  return text;
}

export async function browserWait(ms: number): Promise<string> {
  const p = await ensureBrowser();
  await p.waitForTimeout(Math.min(ms, 10000)); // cap at 10s
  return `Waited ${ms}ms`;
}
