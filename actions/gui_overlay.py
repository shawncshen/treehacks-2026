"""Tkinter GUI overlay — EMG thought-class buttons + free entry for browser control."""

import queue
import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actions.vision import Suggestion

# EMG thought classes (from EMG-UKA / SilentSpeech corpus)
EMG_CLASSES = ("OPEN", "CLOSE", "TIP", "BACK", "LIPS")

# ── Color palette — translucent sky-blue ──────────────────────
BG          = "#e8f4fc"   # very light sky blue
BG_LIGHT    = "#dceef8"   # slightly deeper sky
BG_CARD     = "#d0e8f4"   # card / bubble background
FG          = "#050510"   # near-black text
FG_DIM      = "#1a2840"   # dark navy secondary
ACCENT      = "#0077cc"   # deep sky accent
GREEN       = "#00a843"   # status green
RED         = "#e03030"   # status red
BORDER      = "#b8d8ec"   # subtle border
SELECTED_BG = "#c8e2f8"   # selected row highlight
TITLE_BG    = "#d4eaf6"   # title bar


class GuiOverlay:
    """EMG thought-class buttons + free entry. Simulates EMG sensor output for browser control."""

    def __init__(self, command_queue: queue.Queue, root: tk.Tk):
        self._queue = command_queue
        self._root = root
        self._list_frame: tk.Frame | None = None
        self._status_label: tk.Label | None = None
        self._status_dot: tk.Label | None = None
        self._sequence_label: tk.Label | None = None
        self._is_ready: bool = False
        self._build_ui()

    # ── UI construction ────────────────────────────────────────
    def _build_ui(self):
        root = self._root
        root.title("SilentPilot")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.82)
        except tk.TclError:
            pass
        root.configure(bg=BG)
        root.geometry("260x280")
        root.minsize(240, 220)
        root.resizable(False, False)

        # Position near right edge of screen
        sw = root.winfo_screenwidth()
        root.geometry(f"+{int(sw) - 290}+80")

        # ── Main container ────────────────────────────────────
        main = tk.Frame(root, bg=BG, padx=8, pady=6)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Status row ────────────────────────────────────────
        status_frame = tk.Frame(main, bg=BG)
        status_frame.pack(fill=tk.X, pady=(0, 4))

        self._status_dot = tk.Label(status_frame, text="●", font=("", 9), fg=RED, bg=BG)
        self._status_dot.pack(side=tk.LEFT, padx=(0, 4))
        self._status_label = tk.Label(status_frame, text="Ready",
                                      font=("", 9, "bold"), fg=FG, bg=BG, anchor=tk.W)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── EMG thought-class buttons ─────────────────────────
        btn_cfg = dict(font=("", 9, "bold"), bd=0, padx=4, pady=3, cursor="hand2")

        emg_row1 = tk.Frame(main, bg=BG)
        emg_row1.pack(fill=tk.X, pady=(0, 2))
        for cls in ("OPEN", "CLOSE", "TIP"):
            tk.Button(emg_row1, text=cls, bg=BG_CARD, fg=FG,
                      activebackground=SELECTED_BG, activeforeground=FG,
                      command=lambda c=cls: self._queue.put(f"emg:{c}"),
                      **btn_cfg).pack(side=tk.LEFT, padx=1)

        emg_row2 = tk.Frame(main, bg=BG)
        emg_row2.pack(fill=tk.X, pady=(0, 4))
        for cls in ("BACK", "LIPS"):
            tk.Button(emg_row2, text=cls, bg=BG_CARD, fg=FG,
                      activebackground=SELECTED_BG, activeforeground=FG,
                      command=lambda c=cls: self._queue.put(f"emg:{c}"),
                      **btn_cfg).pack(side=tk.LEFT, padx=1)
        tk.Button(emg_row2, text="REST", bg="#ffd4a0", fg=FG,
                  activebackground="#ffb366", activeforeground=FG,
                  command=lambda: self._queue.put("emg:REST"),
                  **btn_cfg).pack(side=tk.LEFT, padx=1)

        self._sequence_label = tk.Label(main, text="(empty)", font=("", 8), fg=FG_DIM, bg=BG, anchor=tk.W)
        self._sequence_label.pack(fill=tk.X, pady=(0, 2))

        # ── Content area (word confirmation) ──────────────────
        self._list_frame = tk.Frame(main, bg=BG)
        self._list_frame.pack(fill=tk.X, pady=(0, 4))

        # ── Prompt display + Go ────────────────────────────────
        prompt_row = tk.Frame(main, bg=BG)
        prompt_row.pack(fill=tk.X, pady=(0, 4))

        self._prompt_label = tk.Label(prompt_row, text="", font=("", 8), fg=FG_DIM, bg=BG,
                                      anchor=tk.W, wraplength=220)
        self._prompt_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._goal_btn = tk.Button(
            prompt_row, text="Go", font=("", 9, "bold"),
            bg=ACCENT, fg="#ffffff", activebackground="#0277bd", activeforeground="#ffffff",
            bd=0, padx=10, pady=2, cursor="hand2",
            command=lambda: self._queue.put("goal:run"),
        )
        self._goal_btn.pack(side=tk.LEFT)

        # ── Stop agent button (hidden by default) ────────────
        self._stop_btn = tk.Button(
            main, text="STOP AGENT", font=("SF Pro Display", 12, "bold"),
            bg="#1a1a1a", fg="#ff6666", activebackground="#333333", activeforeground="#ff6666",
            bd=0, pady=8, cursor="hand2",
            command=lambda: self._queue.put("stop_agent"),
        )

        root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # ── Internal callbacks ────────────────────────────────────
    def _on_quit(self):
        self._queue.put("quit")
        self._root.quit()

    # ── Public API (unchanged signatures) ─────────────────────
    def set_status(self, ready: bool):
        """Set the status indicator: green (ready) or red (processing). Thread-safe."""
        self._root.after(0, self._update_status, ready)

    def _update_status(self, ready: bool):
        self._is_ready = ready
        if self._status_dot:
            self._status_dot.config(fg=GREEN if ready else RED)
        if self._status_label:
            text = "Ready" if ready else "Processing…"
            self._status_label.config(text=text)

    def update_sequence(self, sequence: list[str]):
        """Update the displayed EMG sequence (thread-safe)."""
        def _do():
            if self._sequence_label:
                s = " → ".join(sequence) if sequence else "(empty)"
                self._sequence_label.config(text=s)
        self._root.after(0, _do)

    def update_prompt_display(self, words: list[str]):
        """Update the displayed prompt (thread-safe)."""
        def _do():
            if self._prompt_label:
                self._prompt_label.config(text=" ".join(words) if words else "")
        self._root.after(0, _do)

    def show_word_confirmation(self, suggested_word: str, has_alternatives: bool):
        """Show 'Did you mean X?' with Yes / No / Retry. Thread-safe."""
        self._root.after(0, self._show_word_confirmation_ui, suggested_word, has_alternatives)

    def _show_word_confirmation_ui(self, suggested_word: str, has_alternatives: bool):
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()

        no_matches = suggested_word == "(no matches)"

        card = tk.Frame(frame, bg=BG_CARD, padx=8, pady=6)
        card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(card, text="No matches" if no_matches else "Did you mean?",
                 font=("", 8, "bold"), fg=ACCENT, bg=BG_CARD, anchor=tk.W).pack(anchor=tk.W, pady=(0, 2))
        msg = ("\"" + suggested_word + "\"?" if not no_matches
               else "No words match. Click No to retry.")
        tk.Label(card, text=msg, font=("", 9, "bold"), fg=FG, bg=BG_CARD,
                 anchor=tk.W, wraplength=220).pack(anchor=tk.W)

        btn_row = tk.Frame(frame, bg=BG)
        btn_row.pack(fill=tk.X, pady=(0, 4))

        btn_cfg = dict(font=("", 8, "bold"), bd=0, padx=12, pady=4, cursor="hand2")
        if not no_matches:
            tk.Button(btn_row, text="Yes", bg="#1a1a1a", fg="#44ee88",
                      activebackground="#333333", activeforeground="#44ee88",
                      command=lambda: self._queue.put("word_yes"),
                      **btn_cfg).pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(btn_row, text="No", bg="#1a1a1a", fg="#ff6666",
                  activebackground="#333333", activeforeground="#ff6666",
                  command=lambda: self._queue.put("word_no"),
                  **btn_cfg).pack(side=tk.LEFT, padx=(0, 4))

        if has_alternatives and not no_matches:
            tk.Button(btn_row, text="Retry", bg="#1a1a1a", fg="#88aaff",
                      activebackground="#333333", activeforeground="#88aaff",
                      command=lambda: self._queue.put("word_retry"),
                      **btn_cfg).pack(side=tk.LEFT)

    def clear_content_area(self):
        """Clear the content area (list_frame) and return to idle (thread-safe)."""
        self._root.after(0, self._clear_content_ui)

    def _clear_content_ui(self):
        frame = self._list_frame
        if frame:
            for w in frame.winfo_children():
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
                self._status_label.config(text=t[:35] + "…" if len(t) > 35 else t)
        self._root.after(0, _do)

    def update_cursor_info(self, x, y, action):
        """Update the cursor position display (thread-safe). No-op when minimal UI."""
        pass

    def show_stop_button(self):
        """Show the stop agent button (thread-safe)."""
        self._root.after(0, lambda: self._stop_btn.pack(fill=tk.X, pady=(6, 0)))

    def hide_stop_button(self):
        """Hide the stop agent button (thread-safe)."""
        self._root.after(0, self._stop_btn.pack_forget)

    def clear(self):
        """Clear the list (thread-safe)."""
        self._root.after(0, self._clear_ui)

    def _clear_ui(self):
        self.clear_content_area()
