
import { PageCursor } from "./tools/cursor.js";

// Mock Page
const mockPage = {
    evaluate: async (fn: any, args?: any) => {
        if (typeof fn === 'string') {
            console.log("Evaluating string JS:", fn.substring(0, 100) + "...");
        } else {
            console.log("Evaluating function JS with args:", args);
        }
        return { x: 640, y: 400 }; // mock center
    }
} as any;

async function test() {
    console.log("--- Starting PageCursor Test ---");
    const cursor = new PageCursor(mockPage);

    console.log("\n1. Testing attach()");
    await cursor.attach();

    console.log("\n2. Testing moveTo(100, 200)");
    await cursor.moveTo(100, 200);

    console.log("\n3. Testing clickEffect()");
    await cursor.clickEffect();

    console.log("\n4. Testing hide()");
    await cursor.hide();

    console.log("\n--- PageCursor Test Complete ---");
}

test().catch(console.error);
