"""EMG thought-class to word lookup. Maps letter classes (OPEN, CLOSE, TIP, BACK, LIPS) to dictionary words."""

from pathlib import Path

# Letter → thought class mapping (from EMG-UKA / SilentSpeech corpus)
# OPEN: A, O, U, H
# CLOSE: E, I, Y
# TIP: T, D, N, L, S, Z
# BACK: K, G, J, C, R
# LIPS: B, M, F, P, W
LETTER_TO_CLASS = {
    "a": "OPEN", "o": "OPEN", "u": "OPEN", "h": "OPEN",
    "e": "CLOSE", "i": "CLOSE", "y": "CLOSE",
    "t": "TIP", "d": "TIP", "n": "TIP", "l": "TIP", "s": "TIP", "z": "TIP",
    "k": "BACK", "g": "BACK", "j": "BACK", "c": "BACK", "r": "BACK",
    "b": "LIPS", "m": "LIPS", "f": "LIPS", "p": "LIPS", "w": "LIPS",
}

EMG_CLASSES = ("OPEN", "CLOSE", "TIP", "BACK", "LIPS")

_WORD_LIST: list[str] | None = None


def _load_words() -> list[str]:
    global _WORD_LIST
    if _WORD_LIST is not None:
        return _WORD_LIST
    # Try bundled list first
    bundled = Path(__file__).parent / "words_common.txt"
    if bundled.exists():
        words = [w.strip().lower() for w in bundled.read_text().splitlines() if w.strip()]
        # Filter: only words where every letter maps to a class (no q, x, v)
        valid = []
        for w in words:
            if all(c in LETTER_TO_CLASS for c in w):
                valid.append(w)
        _WORD_LIST = valid
        return _WORD_LIST
    # Fallback: /usr/share/dict/words (Unix/macOS)
    for p in ("/usr/share/dict/words", "/usr/share/dict/web2"):
        if Path(p).exists():
            words = [w.strip().lower() for w in Path(p).read_text().splitlines() if w.strip()]
            valid = [w for w in words if 2 <= len(w) <= 20 and all(c in LETTER_TO_CLASS for c in w)]
            _WORD_LIST = valid[:15000]
            return _WORD_LIST
    _WORD_LIST = []
    return _WORD_LIST


def word_to_classes(word: str) -> list[str] | None:
    """Convert word to class sequence. Returns None if any letter is unmappable."""
    out = []
    for c in word.lower():
        if c not in LETTER_TO_CLASS:
            return None
        out.append(LETTER_TO_CLASS[c])
    return out


def find_words_for_sequence(class_sequence: list[str]) -> list[str]:
    """Given a sequence of thought classes (OPEN, CLOSE, TIP, BACK, LIPS), return matching words."""
    if not class_sequence:
        return []
    words = _load_words()
    matches = []
    seq = [c.upper() for c in class_sequence]
    for w in words:
        if len(w) != len(seq):
            continue
        cs = word_to_classes(w)
        if cs and cs == seq:
            matches.append(w)
    return matches
