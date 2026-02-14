# Codebase Agentability Analysis

This report evaluates the "agentability" of the SilentPilot codebase, specifically focusing on how well an AI agent like Cursor can understand, navigate, and contribute to the project.

## Current State

### Strengths
- **Modular Structure**: The separation of `emg_core`, `app_ui`, `agent`, and `mcp_server` is clear and logical.
- **Modern Stack**: Use of FastAPI, TypeScript, and MCP (Model Context Protocol) aligns with current AI agent capabilities.
- **Documentation**: Existing `README.md` and `CLAUDE_SESSION_NOTES.md` provide valuable context.
- **Internal Agent logic**: The project already implements its own agent orchestrator, showing a high level of "agentic" maturity.

### Weaknesses
- **Monorepo Complexity**: For an agent, the presence of multiple languages (Python, TS) and two different browser control systems (`actions/` vs `mcp_server/`) can be confusing without explicit guidance.
- **Implicit Knowledge**: Some technical details (e.g., how the DSP pipeline relates to the high-level commands) are documented in session notes but not in a way that an agent can automatically use as a "system prompt".
- **Lack of Visual Feedback**: While the internal `actions/` library has an "animated cursor", the modern `mcp_server/` does not, making it harder for an agent to "show" its actions.
- **Missing Codebase Rules**: There are no `.cursorrules` or `AGENTS.md` files to provide global context and constraints.

## Identified Areas for Improvement

1. **Unified Agent Guidance**: Add `.cursorrules` and `AGENTS.md` to the root to provide a single source of truth for AI agents.
2. **Visual Action Traceability**: Port the "Animated Cursor" from the legacy Python code to the TypeScript MCP server. This allows the agent to visually demonstrate its actions (clicks, moves) on the screen.
3. **Explicit Component Mapping**: Better document the relationship between the components, especially the two browser control systems.
4. **Agent-friendly Testing**: Ensure tests are easy to run and verify for an agent, with clear output.

## Recommendation

To improve "agent ability of cursor", we should:
- Implement a `.cursorrules` file.
- Port the `PageCursor` visual feedback to the MCP server.
- Standardize agent instructions in an `AGENTS.md` file.
