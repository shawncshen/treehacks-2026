"""Terminal-based overlay for displaying action suggestions."""

import os


class Overlay:
    """Displays numbered suggestions in the terminal with a selection marker."""

    def show(self, suggestions: list, selected_index: int):
        """Clear terminal and display suggestions with selection highlight."""
        self.clear()
        print("=" * 60)
        print("  SilentPilot — Suggested Actions")
        print("=" * 60)
        for i, s in enumerate(suggestions):
            marker = ">>" if i == selected_index else "  "
            print(f"  {marker} [{i}] {s.label}")
            print(f"       {s.description}")
        print("=" * 60)
        print("  0-9: pick | UP/DOWN: navigate | ENTER: execute | q: quit")
        print("=" * 60)

    def clear(self):
        """Clear the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")
