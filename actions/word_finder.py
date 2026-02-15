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

# Pre-computed: word list and class-sequence index for fast lookup
_WORD_LIST: list[str] | None = None
_WORD_CLASSES: dict[str, list[str]] | None = None


def _load_words() -> list[str]:
    global _WORD_LIST, _WORD_CLASSES
    if _WORD_LIST is not None:
        return _WORD_LIST
    # Try bundled list first
    bundled = Path(__file__).parent / "words_common.txt"
    if bundled.exists():
        words = [w.strip().lower() for w in bundled.read_text().splitlines() if w.strip()]
    else:
        # Fallback: /usr/share/dict/words (Unix/macOS)
        words = []
        for p in ("/usr/share/dict/words", "/usr/share/dict/web2"):
            if Path(p).exists():
                words = [w.strip().lower() for w in Path(p).read_text().splitlines() if w.strip()]
                words = [w for w in words if 2 <= len(w) <= 20][:15000]
                break

    # Filter and pre-compute class sequences (preserves frequency order from file)
    valid = []
    classes = {}
    for w in words:
        cs = word_to_classes(w)
        if cs is not None:
            valid.append(w)
            classes[w] = cs
    _WORD_LIST = valid
    _WORD_CLASSES = classes
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
    """Given a sequence of thought classes, return matching words sorted by frequency.

    The word list is already ordered by frequency (most common first),
    so results preserve that ordering — candidates[0] is the most common match.
    """
    if not class_sequence:
        return []
    words = _load_words()
    seq = [c.upper() for c in class_sequence]
    seq_len = len(seq)
    matches = []
    for w in words:
        if len(w) != seq_len:
            continue
        if _WORD_CLASSES[w] == seq:
            matches.append(w)
    return matches
