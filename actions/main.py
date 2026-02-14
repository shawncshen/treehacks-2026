"""Entry point for the autonomous browser agent."""

import argparse
import asyncio
import traceback

from actions.engine import AutonomousAgent
from actions.config import OPENAI_API_KEY, MAX_AGENT_STEPS


async def run_agent(goal: str, url: str, max_steps: int):
    """Launch browser and run autonomous agent with the given goal."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    agent = AutonomousAgent(max_steps=max_steps)
    try:
        print("  Launching browser...", flush=True)
        await agent.start(url)
        print("  Browser ready!\n", flush=True)
        await agent.run(goal)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.", flush=True)
    except Exception as e:
        print(f"\n  Fatal error: {e}", flush=True)
        traceback.print_exc()
    finally:
        await agent.stop()


def main():
    parser = argparse.ArgumentParser(description="SilentPilot Autonomous Browser Agent")
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help='Goal prompt, e.g. "go to amazon and find the cheapest basketball"',
    )
    parser.add_argument(
        "--url",
        default="https://www.google.com",
        help="Starting URL (default: https://www.google.com)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Maximum agent steps (default: {MAX_AGENT_STEPS})",
    )
    args = parser.parse_args()

    goal = args.goal
    if not goal:
        goal = input("  Enter your goal: ").strip()
        if not goal:
            print("  No goal provided. Exiting.", flush=True)
            return

    asyncio.run(run_agent(goal, args.url, args.max_steps))


if __name__ == "__main__":
    main()
