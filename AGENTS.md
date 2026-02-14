# Instructions for AI Agents

Welcome to the SilentPilot repository. This file provides critical context for AI agents working on this codebase.

## Repository Map

- `/silentpilot/` - Main Project
  - `emg_core/` - Python DSP/ML server (FastAPI)
    - `dsp/` - Feature extraction (TD0/TD10)
    - `ml/` - Training and Inference engines
    - `api/` - REST and WebSocket endpoints
  - `app_ui/` - React/Next.js dashboard
  - `agent/` - Agent orchestrator (OpenAI + MCP)
  - `mcp_server/` - MCP server with Playwright browser tools
- `/actions/` - Legacy Python browser controller (Pyppeteer)
- `/EMG-UKA-Trial-Corpus/` - Reference dataset for training (on `datasets` branch)

## Technical Constraints

- **EMG Sampling Rate**: 250 Hz (default) or 600 Hz (EMG-UKA).
- **Feature Pipeline**: The system uses a sliding window (TD10) to aggregate 5 features per channel across 21 frames, resulting in 420 dimensions per frame (for 4 channels), aggregated to 840 dimensions per segment.
- **MCP Server**: Uses SSE transport on port 3333.
- **Visual Cursor**: All browser actions performed by an agent SHOULD be visible via the `PageCursor` (red arrow injected into the page).

## Rules of Engagement

1. **Verify DSP Changes**: Use `silentpilot/scripts/e2e_test.py` to verify that changes to the DSP pipeline don't break classification accuracy.
2. **Prefer MCP**: For browser automation, prefer extending the TypeScript MCP server over the legacy Python `actions` folder.
3. **Respect the Cursor**: Do not remove or hide the `PageCursor` unless specifically instructed for a "stealth" mode. It is vital for user trust.
