"""Modern GUI overlay — clean 3-button interface for MindOS.

Shows the current suggested word and three actions: Retry, Yes, No.
Uses customtkinter for a polished, React-app-like aesthetic.
"""

import queue
import tkinter as tk
from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:
    from actions.vision import Suggestion

# EMG thought classes (kept for external imports)
EMG_CLASSES = ("OPEN", "CLOSE", "TIP", "BACK", "LIPS")

# ── Modern dark color palette ───────────────────────────────────
BG           = "#0c0c14"
BG_CARD      = "#14141f"
BG_ELEVATED  = "#1a1a2a"
FG           = "#e8e8f0"
FG_DIM       = "#8888a8"
FG_MUTED     = "#555570"
ACCENT       = "#6366f1"
ACCENT_HOVER = "#818cf8"
GREEN        = "#22c55e"
GREEN_BG     = "#0f2918"
GREEN_HOVER  = "#153520"
RED          = "#ef4444"
RED_BG       = "#2a1010"
RED_HOVER    = "#3a1818"
BLUE         = "#60a5fa"
BLUE_BG      = "#101a30"
BLUE_HOVER   = "#182440"
BORDER       = "#222235"


class GuiOverlay:
    """Minimal 3-button overlay: Retry / Yes / No for word confirmation."""

    def __init__(self, command_queue: queue.Queue, root: tk.Tk):
        self._queue = command_queue
        self._root = root
        self._content_frame: ctk.CTkFrame | None = None
        self._status_label: ctk.CTkLabel | None = None
        self._status_dot: tk.Label | None = None
        self._word_label: ctk.CTkLabel | None = None
        self._prompt_label: ctk.CTkLabel | None = None
        self._retry_btn: ctk.CTkButton | None = None
        self._yes_btn: ctk.CTkButton | None = None
        self._no_btn: ctk.CTkButton | None = None
        self._action_frame: ctk.CTkFrame | None = None
        self._sequence_label: ctk.CTkLabel | None = None
        self._is_ready: bool = False
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────
    def _build_ui(self):
        root = self._root
        root.title("MindOS")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.94)
        except tk.TclError:
            pass
        root.configure(bg=BG)
        root.geometry("280x520")
        root.minsize(260, 480)
        root.resizable(False, False)

        # Position near right edge of screen
        sw = root.winfo_screenwidth()
        root.geometry(f"+{int(sw) - 310}+60")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Main container ──────────────────────────────────────────
        main = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Header ──────────────────────────────────────────────────
        header = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=14)
        header.pack(fill=tk.X, padx=14, pady=(14, 0))

        header_inner = ctk.CTkFrame(header, fg_color="transparent")
        header_inner.pack(fill=tk.X, padx=16, pady=12)

        ctk.CTkLabel(
            header_inner, text="MindOS",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=FG,
        ).pack(side=tk.LEFT)

        # Status pill
        self._status_badge = ctk.CTkFrame(
            header_inner, fg_color=GREEN_BG, corner_radius=10,
            width=82, height=24)
        self._status_badge.pack(side=tk.RIGHT)
        self._status_badge.pack_propagate(False)

        pill_inner = ctk.CTkFrame(self._status_badge, fg_color="transparent")
        pill_inner.place(relx=0.5, rely=0.5, anchor="center")

        self._status_dot = tk.Label(
            pill_inner, text="\u25cf", font=("", 7), fg=GREEN,
            bg=self._status_badge._apply_appearance_mode(
                self._status_badge.cget("fg_color")))
        self._status_dot.pack(side=tk.LEFT, padx=(0, 4))

        self._status_label = ctk.CTkLabel(
            pill_inner, text="Ready",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=FG_DIM,
        )
        self._status_label.pack(side=tk.LEFT)

        # ── Suggested word display ──────────────────────────────────
        word_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=14)
        word_card.pack(fill=tk.X, padx=14, pady=(10, 0))

        word_inner = ctk.CTkFrame(word_card, fg_color="transparent")
        word_inner.pack(fill=tk.X, padx=16, pady=14)

        ctk.CTkLabel(
            word_inner, text="Suggestion",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=FG_MUTED,
        ).pack(anchor=tk.W, pady=(0, 8))

        self._word_label = ctk.CTkLabel(
            word_inner, text="--",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=FG_DIM, anchor=tk.W,
        )
        self._word_label.pack(anchor=tk.W)

        # Sequence (small, below the word)
        self._sequence_label = ctk.CTkLabel(
            word_inner, text="",
            font=ctk.CTkFont(family="SF Mono", size=10),
            text_color=FG_MUTED, anchor=tk.W,
        )
        self._sequence_label.pack(anchor=tk.W, pady=(6, 0))

        # ── Action buttons ──────────────────────────────────────────
        self._action_frame = ctk.CTkFrame(main, fg_color="transparent")
        self._action_frame.pack(fill=tk.X, padx=14, pady=(10, 0))

        # Retry — top, full width (starts disabled until a word is suggested)
        self._retry_btn = ctk.CTkButton(
            self._action_frame, text="Retry",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=BORDER, hover_color=BLUE_HOVER,
            text_color=FG_MUTED, corner_radius=10,
            height=44, cursor="hand2", state="disabled",
            command=lambda: self._queue.put("word_retry"),
        )
        self._retry_btn.pack(fill=tk.X, pady=(0, 8))

        # Yes / No — side by side
        yn_row = ctk.CTkFrame(self._action_frame, fg_color="transparent")
        yn_row.pack(fill=tk.X)

        self._yes_btn = ctk.CTkButton(
            yn_row, text="Yes",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=BORDER, hover_color=GREEN_HOVER,
            text_color=FG_MUTED, corner_radius=10,
            height=44, cursor="hand2", state="disabled",
            command=lambda: self._queue.put("word_yes"),
        )
        self._yes_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self._no_btn = ctk.CTkButton(
            yn_row, text="No",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=BORDER, hover_color=RED_HOVER,
            text_color=FG_MUTED, corner_radius=10,
            height=44, cursor="hand2", state="disabled",
            command=lambda: self._queue.put("word_no"),
        )
        self._no_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ── Composed prompt + Go ────────────────────────────────────
        prompt_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=14)
        prompt_card.pack(fill=tk.X, padx=14, pady=(10, 0))

        prompt_inner = ctk.CTkFrame(prompt_card, fg_color="transparent")
        prompt_inner.pack(fill=tk.X, padx=16, pady=12)

        ctk.CTkLabel(
            prompt_inner, text="Prompt",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=FG_MUTED,
        ).pack(anchor=tk.W, pady=(0, 4))

        prompt_row = ctk.CTkFrame(prompt_inner, fg_color="transparent")
        prompt_row.pack(fill=tk.X)

        self._prompt_label = ctk.CTkLabel(
            prompt_row, text="Waiting for input...",
            font=ctk.CTkFont(size=12),
            text_color=FG_DIM, anchor=tk.W, wraplength=160,
        )
        self._prompt_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._goal_btn = ctk.CTkButton(
            prompt_row, text="Go",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=BORDER, hover_color=ACCENT_HOVER,
            text_color=FG_MUTED, corner_radius=8,
            width=52, height=30, cursor="hand2",
            state="disabled",
            command=lambda: self._queue.put("goal:run"),
        )
        self._goal_btn.pack(side=tk.RIGHT)

        # ── Stop agent button (hidden by default) ──────────────────
        self._stop_frame = ctk.CTkFrame(main, fg_color="transparent")

        self._stop_btn = ctk.CTkButton(
            self._stop_frame, text="Stop Agent",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=RED_BG, hover_color=RED_HOVER,
            text_color=RED, corner_radius=10,
            height=40, cursor="hand2",
            command=lambda: self._queue.put("stop_agent"),
        )
        self._stop_btn.pack(fill=tk.X)

        # ── Content area (for dynamic inserts like word confirmation) ──
        self._content_frame = ctk.CTkFrame(main, fg_color="transparent", corner_radius=0)
        self._content_frame.pack(fill=tk.X, padx=14, pady=(6, 0))
        # Alias for backward compat
        self._list_frame = self._content_frame

        # ── EMG class input buttons (for testing/demo without hardware) ──
        emg_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=14)
        emg_card.pack(fill=tk.X, padx=14, pady=(10, 0))

        emg_inner = ctk.CTkFrame(emg_card, fg_color="transparent")
        emg_inner.pack(fill=tk.X, padx=10, pady=8)

        ctk.CTkLabel(
            emg_inner, text="EMG Input",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=FG_MUTED,
        ).pack(anchor=tk.W, pady=(0, 6))

        # Row 1: OPEN, CLOSE, TIP
        emg_row1 = ctk.CTkFrame(emg_inner, fg_color="transparent")
        emg_row1.pack(fill=tk.X, pady=(0, 4))
        for cls in ("OPEN", "CLOSE", "TIP"):
            ctk.CTkButton(
                emg_row1, text=cls,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=BG_ELEVATED, hover_color=BORDER,
                text_color=FG_DIM, corner_radius=8,
                height=28, cursor="hand2",
                command=lambda c=cls: self._queue.put(f"emg:{c}"),
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Row 2: BACK, LIPS, REST
        emg_row2 = ctk.CTkFrame(emg_inner, fg_color="transparent")
        emg_row2.pack(fill=tk.X)
        for cls in ("BACK", "LIPS"):
            ctk.CTkButton(
                emg_row2, text=cls,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=BG_ELEVATED, hover_color=BORDER,
                text_color=FG_DIM, corner_radius=8,
                height=28, cursor="hand2",
                command=lambda c=cls: self._queue.put(f"emg:{c}"),
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        # REST button — distinct color to stand out
        ctk.CTkButton(
            emg_row2, text="REST",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="#ffffff", corner_radius=8,
            height=28, cursor="hand2",
            command=lambda: self._queue.put("emg:REST"),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # ── Footer ──────────────────────────────────────────────────
        ctk.CTkLabel(
            main, text="TreeHacks 2026",
            font=ctk.CTkFont(size=10),
            text_color=FG_MUTED,
        ).pack(side=tk.BOTTOM, pady=(0, 10))

        root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # ── Internal callbacks ─────────────────────────────────────────
    def _on_quit(self):
        self._queue.put("quit")
        self._root.quit()

    # ── Public API ─────────────────────────────────────────────────
    def set_status(self, ready: bool):
        """Set the status indicator: green (ready) or red (processing). Thread-safe."""
        self._root.after(0, self._update_status, ready)

    def _update_status(self, ready: bool):
        self._is_ready = ready
        if self._status_dot:
            self._status_dot.config(fg=GREEN if ready else RED)
        if self._status_label:
            self._status_label.configure(text="Ready" if ready else "Processing...")
        badge_bg = GREEN_BG if ready else RED_BG
        self._status_badge.configure(fg_color=badge_bg)
        try:
            resolved = self._status_badge._apply_appearance_mode(badge_bg)
            self._status_dot.config(bg=resolved)
        except Exception:
            pass

    def update_sequence(self, sequence: list[str]):
        """Update the displayed EMG sequence (thread-safe)."""
        def _do():
            if self._sequence_label:
                if sequence:
                    s = "  \u2192  ".join(sequence)
                    self._sequence_label.configure(text=s, text_color=ACCENT)
                else:
                    self._sequence_label.configure(text="", text_color=FG_MUTED)
        self._root.after(0, _do)

    def update_prompt_display(self, words: list[str]):
        """Update the displayed prompt and Go button state (thread-safe)."""
        def _do():
            if self._prompt_label:
                if words:
                    self._prompt_label.configure(text=" ".join(words), text_color=FG)
                    self._goal_btn.configure(
                        state="normal", fg_color=ACCENT, text_color="#ffffff")
                else:
                    self._prompt_label.configure(text="Waiting for input...", text_color=FG_DIM)
                    self._goal_btn.configure(
                        state="disabled", fg_color=BORDER, text_color=FG_MUTED)
        self._root.after(0, _do)

    def show_word_confirmation(self, suggested_word: str, has_alternatives: bool):
        """Update the suggestion display and enable/disable buttons. Thread-safe."""
        self._root.after(0, self._show_word_confirmation_ui, suggested_word, has_alternatives)

    def _show_word_confirmation_ui(self, suggested_word: str, has_alternatives: bool):
        no_matches = suggested_word == "(no matches)"

        if no_matches:
            self._word_label.configure(text="No matches", text_color=FG_MUTED)
            self._retry_btn.configure(state="disabled", fg_color=BORDER, text_color=FG_MUTED)
            self._yes_btn.configure(state="disabled", fg_color=BORDER, text_color=FG_MUTED)
            # No is always active when a suggestion was attempted (user can dismiss)
            self._no_btn.configure(state="normal", fg_color=RED_BG, text_color=RED)
        else:
            self._word_label.configure(text=suggested_word, text_color=FG)
            # Retry enabled only when there are alternatives
            if has_alternatives:
                self._retry_btn.configure(state="normal", fg_color=BLUE_BG, text_color=BLUE)
            else:
                self._retry_btn.configure(state="disabled", fg_color=BORDER, text_color=FG_MUTED)
            self._yes_btn.configure(state="normal", fg_color=GREEN_BG, text_color=GREEN)
            self._no_btn.configure(state="normal", fg_color=RED_BG, text_color=RED)

    def clear_content_area(self):
        """Reset the suggestion display back to idle (thread-safe)."""
        self._root.after(0, self._clear_content_ui)

    def _clear_content_ui(self):
        self._word_label.configure(text="--", text_color=FG_DIM)
        if self._sequence_label:
            self._sequence_label.configure(text="", text_color=FG_MUTED)
        # Idle state: disable action buttons since no word is being suggested
        self._retry_btn.configure(state="disabled", fg_color=BORDER, text_color=FG_MUTED)
        self._yes_btn.configure(state="disabled", fg_color=BORDER, text_color=FG_MUTED)
        self._no_btn.configure(state="disabled", fg_color=BORDER, text_color=FG_MUTED)
        # Also clear any dynamic children in content frame
        if self._content_frame:
            for w in self._content_frame.winfo_children():
                w.destroy()

    def show(self, suggestions: list["Suggestion"], selected_index: int, smart: bool = False):
        pass

    def prompt_type_text(self):
        pass

    def show_agent_question(self, question: str):
        pass

    def update_agent_status(self, text: str):
        """Show agent step info in the status label (thread-safe)."""
        def _do(t=text):
            if self._status_label:
                display = t[:28] + "\u2026" if len(t) > 28 else t
                self._status_label.configure(text=display)
        self._root.after(0, _do)

    def update_cursor_info(self, x, y, action):
        pass

    def show_stop_button(self):
        """Show the stop agent button (thread-safe)."""
        self._root.after(0, lambda: self._stop_frame.pack(fill=tk.X, padx=14, pady=(8, 0)))

    def hide_stop_button(self):
        """Hide the stop agent button (thread-safe)."""
        self._root.after(0, self._stop_frame.pack_forget)

    def clear(self):
        """Clear the list (thread-safe)."""
        self._root.after(0, self._clear_content_ui)
