#!/usr/bin/env python3
"""
SilentPilot - EMG Calibration & Inference GUI
==============================================
Single-process web app: FastAPI + embedded HTML/JS frontend.

Usage:
    python scripts/emg_app.py --port /dev/cu.usbmodemXXXX --user aarush

Opens browser at http://localhost:8080
"""

import argparse
import asyncio
import collections
import json
import os
from dotenv import load_dotenv
load_dotenv()
import sys
import threading
import time
import webbrowser

import numpy as np
import serial
import serial.tools.list_ports

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from emg_core.dsp.features import extract_features

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 250
NUM_CHANNELS = 2
RECORD_SEC = 1.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * RECORD_SEC)
CROP_MARGIN_SEC = 0.04
ONSET_RATIO = 0.25
MIN_GESTURE_SEC = 0.12
PLOT_HISTORY = 5  # seconds shown in live plot
BAUD = 115200
WS_FPS = 25  # WebSocket stream rate

LETTERS_BY_GROUP = {
    "OPEN":  ["A", "O", "U", "H"],
    "CLOSE": ["E", "I", "Y"],
    "TIP":   ["T", "D", "N", "L", "S", "Z"],
    "BACK":  ["K", "G", "J", "C", "R"],
    "LIPS":  ["B", "M", "F", "P", "W"],
}

GROUP_TO_LETTERS = {
    "OPEN":  set("aouh"),
    "CLOSE": set("eiy"),
    "TIP":   set("tdnlsz"),
    "BACK":  set("kgjcqxr"),
    "LIPS":  set("pbmfvw"),
}

LETTER_TO_GROUP = {}
for _g, _letters in GROUP_TO_LETTERS.items():
    for _l in _letters:
        LETTER_TO_GROUP[_l] = _g

GROUP_COLORS = {
    "REST":  "#6b7280",
    "OPEN":  "#ef4444",
    "CLOSE": "#22c55e",
    "TIP":   "#06b6d4",
    "BACK":  "#a855f7",
    "LIPS":  "#eab308",
}

ALL_GROUPS = ["REST", "OPEN", "CLOSE", "TIP", "BACK", "LIPS"]
ACTIVE_GROUPS = ["OPEN", "CLOSE", "TIP", "BACK", "LIPS"]

COMMON_WORDS = [
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","her","she","or",
    "an","will","my","one","all","would","there","their","what","so","up","out","if","about",
    "who","get","which","go","me","when","make","can","like","time","no","just","him","know",
    "take","people","into","year","your","good","some","could","them","see","other","than",
    "then","now","look","only","come","its","over","think","also","back","after","use","two",
    "how","our","work","first","well","way","even","new","want","because","any","these","give",
    "day","most","us","great","help","need","find","here","thing","many","still","long","send",
    "open","read","search","create","shop","buy","door","nice","john","meeting","email",
    "calendar","inbox","coffee","pizza","google","amazon","headphones","laptop","phone",
    "review","weather","please","sorry","hello","thanks","world","every","money","love","name",
    "food","play","home","right","hand","house","school","place","point","life","water","room",
    "book","word","music","stop","start","keep","move","next","last","best","sure","much",
    "own","old","big","high","small","man","never","each","same","both","few","turn","end",
    "real","kind","off","head","tell","call","before","why","while","show","side","might",
    "part","too","close","eye","ask","late","run","hard","try","left","lot","begin",
    "line","since","sit","stand","lose","pay","hear","let","meet","put","set","order","early",
    "walk","white","today","lead","live","hold","free","study","power","learn","company",
    "city","team","face","game","group","done","along","during","carry","state","car","night",
    "young","idea","under","body","table","change","watch","plan","story","girl","boy","case",
    "bit","class","bring","clear","system","full","hot","feel","offer","fact","street","already",
    "cut","pass","market","north","south","east","west","fire","air","human","morning","serve",
    "message","compare","cheapest","reply","draft","flight","cart","price","address",
]
FREQ_RANK = {}
for _i, _w in enumerate(COMMON_WORDS):
    if _w not in FREQ_RANK:
        FREQ_RANK[_w] = _i

API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4.1-nano"


# ═══════════════════════════════════════════════════════════════════════════════
#  EMGStream — background serial reader
# ═══════════════════════════════════════════════════════════════════════════════

class EMGStream:
    """Reads ASCII tab-separated EMG data in a background thread."""

    def __init__(self, port, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(2)
        self.ser.reset_input_buffer()

        self._buf = collections.deque(maxlen=SAMPLE_RATE * PLOT_HISTORY)
        self._lock = threading.Lock()
        self._recording = False
        self._recorded = []
        self._alive = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._alive:
            try:
                line = self.ser.readline()
            except serial.SerialException:
                continue
            if not line:
                continue
            try:
                parts = line.decode("ascii", errors="ignore").strip().split("\t")
                if len(parts) >= NUM_CHANNELS:
                    sample = [int(parts[i]) for i in range(NUM_CHANNELS)]
                    with self._lock:
                        self._buf.append(sample)
                        if self._recording:
                            self._recorded.append(sample)
            except (ValueError, IndexError):
                continue

    def snapshot(self, last_n=None):
        """Return recent history as (N, 2) numpy array."""
        with self._lock:
            if self._buf:
                data = list(self._buf)
                if last_n and len(data) > last_n:
                    data = data[-last_n:]
                return np.array(data)
            return np.zeros((1, NUM_CHANNELS))

    def start_rec(self):
        with self._lock:
            self._recorded = []
            self._recording = True

    def stop_rec(self):
        with self._lock:
            self._recording = False
            out = np.array(self._recorded) if self._recorded else np.zeros((1, NUM_CHANNELS))
            self._recorded = []
            return out

    @property
    def is_recording(self):
        return self._recording

    def close(self):
        self._alive = False
        self._thread.join(timeout=2)
        try:
            self.ser.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Smart Crop — energy-based onset/offset detection
# ═══════════════════════════════════════════════════════════════════════════════

def smart_crop(segment):
    """Trim buffer regions, keeping only the active gesture + small margin."""
    if len(segment) < 20:
        return segment

    energy = np.sqrt(np.mean(segment.astype(float) ** 2, axis=1))

    win = max(3, SAMPLE_RATE // 25)
    kernel = np.ones(win) / win
    smooth = np.convolve(energy, kernel, mode="same")

    n = len(smooth)
    edge = max(1, n // 7)
    baseline = np.mean(np.concatenate([smooth[:edge], smooth[-edge:]]))
    peak = np.max(smooth)

    if peak < baseline * 1.5:
        return segment

    thresh = baseline + ONSET_RATIO * (peak - baseline)
    above = smooth > thresh
    if not np.any(above):
        return segment

    onset = int(np.argmax(above))
    offset = int(n - 1 - np.argmax(above[::-1]))

    margin = int(CROP_MARGIN_SEC * SAMPLE_RATE)
    onset = max(0, onset - margin)
    offset = min(n - 1, offset + margin)

    if offset - onset + 1 < int(MIN_GESTURE_SEC * SAMPLE_RATE):
        return segment

    return segment[onset : offset + 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  DataStore — in-memory + .npz persistence
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    """Manages captured segments in memory and on disk."""

    def __init__(self, user):
        self.user = user
        self.data_dir = os.path.join(PROJECT_ROOT, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.path = os.path.join(self.data_dir, f"{user}_calib.npz")
        self.segments = []  # list of np arrays
        self.labels = []    # list of strings
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            data = np.load(self.path, allow_pickle=True)
            self.segments = list(data["segments"])
            self.labels = list(data["labels"])

    def add(self, segment, label):
        self.segments.append(segment)
        self.labels.append(label)
        self._save()
        return len(self.segments) - 1

    def delete_sample(self, idx):
        if 0 <= idx < len(self.segments):
            self.segments.pop(idx)
            self.labels.pop(idx)
            self._save()
            return True
        return False

    def delete_label(self, label):
        indices = [i for i, l in enumerate(self.labels) if l == label]
        for i in sorted(indices, reverse=True):
            self.segments.pop(i)
            self.labels.pop(i)
        self._save()
        return len(indices)

    def delete_all(self):
        self.segments = []
        self.labels = []
        self._save()

    def summary(self):
        counts = {}
        energies = {}
        for seg, lab in zip(self.segments, self.labels):
            counts[lab] = counts.get(lab, 0) + 1
            e = float(np.sqrt(np.mean(seg.astype(float) ** 2)))
            energies.setdefault(lab, []).append(e)

        result = {}
        for lab in ALL_GROUPS:
            c = counts.get(lab, 0)
            es = energies.get(lab, [])
            result[lab] = {
                "count": c,
                "mean_energy": float(np.mean(es)) if es else 0,
                "min_energy": float(np.min(es)) if es else 0,
                "max_energy": float(np.max(es)) if es else 0,
            }
        return result

    def get_sample(self, idx):
        if 0 <= idx < len(self.segments):
            seg = self.segments[idx]
            return {
                "label": self.labels[idx],
                "ch0": seg[:, 0].tolist(),
                "ch1": seg[:, 1].tolist(),
                "length": len(seg),
                "energy_ch0": float(np.sqrt(np.mean(seg[:, 0].astype(float) ** 2))),
                "energy_ch1": float(np.sqrt(np.mean(seg[:, 1].astype(float) ** 2))),
            }
        return None

    def get_samples_for_label(self, label):
        results = []
        for i, (seg, lab) in enumerate(zip(self.segments, self.labels)):
            if lab == label:
                results.append({
                    "idx": i,
                    "length": len(seg),
                    "energy_ch0": float(np.sqrt(np.mean(seg[:, 0].astype(float) ** 2))),
                    "energy_ch1": float(np.sqrt(np.mean(seg[:, 1].astype(float) ** 2))),
                })
        return results

    def compare_sample(self, segment, label):
        """Compare a new sample's features against existing samples of the same class.
        Returns stats for the new sample plus aggregate stats of existing samples,
        and flags potential outliers."""
        seg = segment.astype(float)
        new_e0 = float(np.sqrt(np.mean(seg[:, 0] ** 2)))
        new_e1 = float(np.sqrt(np.mean(seg[:, 1] ** 2)))
        new_wl0 = float(np.sum(np.abs(np.diff(seg[:, 0]))))
        new_wl1 = float(np.sum(np.abs(np.diff(seg[:, 1]))))
        new_peak0 = float(np.max(seg[:, 0]))
        new_peak1 = float(np.max(seg[:, 1]))
        new_len = len(seg)

        # Gather stats from existing samples of same label
        existing_e0, existing_e1 = [], []
        existing_wl0, existing_wl1 = [], []
        existing_peak0, existing_peak1 = [], []
        existing_lens = []

        for s, l in zip(self.segments, self.labels):
            if l == label:
                sf = s.astype(float)
                existing_e0.append(float(np.sqrt(np.mean(sf[:, 0] ** 2))))
                existing_e1.append(float(np.sqrt(np.mean(sf[:, 1] ** 2))))
                existing_wl0.append(float(np.sum(np.abs(np.diff(sf[:, 0])))))
                existing_wl1.append(float(np.sum(np.abs(np.diff(sf[:, 1])))))
                existing_peak0.append(float(np.max(sf[:, 0])))
                existing_peak1.append(float(np.max(sf[:, 1])))
                existing_lens.append(len(sf))

        n_existing = len(existing_e0)

        # Build comparison result
        result = {
            "new": {
                "energy_ch0": new_e0, "energy_ch1": new_e1,
                "waveform_length_ch0": new_wl0, "waveform_length_ch1": new_wl1,
                "peak_ch0": new_peak0, "peak_ch1": new_peak1,
                "length": new_len,
            },
            "n_existing": n_existing,
            "warnings": [],
        }

        if n_existing >= 2:
            def _stats(vals):
                return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}

            result["existing"] = {
                "energy_ch0": _stats(existing_e0), "energy_ch1": _stats(existing_e1),
                "waveform_length_ch0": _stats(existing_wl0), "waveform_length_ch1": _stats(existing_wl1),
                "peak_ch0": _stats(existing_peak0), "peak_ch1": _stats(existing_peak1),
                "length": _stats(existing_lens),
            }

            # Flag outliers (>2 std from mean)
            def _check(name, new_val, vals):
                mu, sigma = np.mean(vals), np.std(vals)
                if sigma > 0 and abs(new_val - mu) > 2 * sigma:
                    direction = "high" if new_val > mu else "low"
                    return f"{name} is {direction} ({new_val:.0f} vs avg {mu:.0f} +/- {sigma:.0f})"
                return None

            for w in [
                _check("CH0 energy", new_e0, existing_e0),
                _check("CH1 energy", new_e1, existing_e1),
                _check("CH0 peak", new_peak0, existing_peak0),
                _check("CH1 peak", new_peak1, existing_peak1),
            ]:
                if w:
                    result["warnings"].append(w)

            # Check if signal looks flat (very low energy)
            if new_e0 < 3 and new_e1 < 3 and label != "REST":
                result["warnings"].append("Both channels near zero - possible electrode issue")

        # Comparison with other groups (to check distinctiveness)
        other_groups_e0 = {}
        other_groups_e1 = {}
        for s, l in zip(self.segments, self.labels):
            if l != label:
                sf = s.astype(float)
                other_groups_e0.setdefault(l, []).append(float(np.sqrt(np.mean(sf[:, 0] ** 2))))
                other_groups_e1.setdefault(l, []).append(float(np.sqrt(np.mean(sf[:, 1] ** 2))))

        result["other_groups"] = {}
        for g in ALL_GROUPS:
            if g in other_groups_e0 and len(other_groups_e0[g]) >= 2:
                result["other_groups"][g] = {
                    "mean_e0": float(np.mean(other_groups_e0[g])),
                    "mean_e1": float(np.mean(other_groups_e1[g])),
                }

        return result

    def _save(self):
        if self.segments:
            np.savez(self.path,
                     segments=np.array(self.segments, dtype=object),
                     labels=np.array(self.labels))
        elif os.path.exists(self.path):
            os.remove(self.path)


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelManager — train / load / predict
# ═══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Handles training and prediction."""

    def __init__(self, user):
        self.user = user
        self.models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, f"{user}_model.joblib")
        self.clf = None
        self.labels = []
        self.accuracy = 0
        self.n_samples = 0
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            import joblib
            data = joblib.load(self.model_path)
            self.clf = data["model"]
            self.labels = data["labels"]
            self.accuracy = data.get("accuracy", 0)
            self.n_samples = data.get("n_samples", 0)

    @property
    def is_loaded(self):
        return self.clf is not None

    def train(self, datastore: DataStore):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix
        import joblib

        segments = datastore.segments
        labels = datastore.labels

        if len(segments) < 10:
            return {"error": "Need at least 10 samples to train"}

        # Extract features
        X = np.array([extract_features(seg, sample_rate=SAMPLE_RATE) for seg in segments])
        y = np.array(labels)

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        label_set = sorted(set(y))
        counts = {l: int(np.sum(y == l)) for l in label_set}

        # Check minimum samples per class
        min_count = min(counts.values())
        n_splits = min(5, min_count)
        if n_splits < 2:
            return {"error": f"Need at least 2 samples per class. Min class has {min_count}."}

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        # Last fold for detailed report
        train_idx, test_idx = None, None
        for train_idx, test_idx in cv.split(X, y):
            pass

        clf_eval = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        clf_eval.fit(X[train_idx], y[train_idx])
        y_pred = clf_eval.predict(X[test_idx])

        report = classification_report(y[test_idx], y_pred, zero_division=0, output_dict=True)
        cm = confusion_matrix(y[test_idx], y_pred, labels=label_set)

        # Train final on all data
        clf.fit(X, y)

        # Feature importance
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        top_features = [{"index": int(i), "importance": float(importances[i])} for i in top_idx]

        # Save
        joblib.dump({
            "model": clf,
            "labels": label_set,
            "mode": "all",
            "n_samples": len(X),
            "accuracy": float(scores.mean()),
        }, self.model_path)

        self.clf = clf
        self.labels = label_set
        self.accuracy = float(scores.mean())
        self.n_samples = len(X)

        return {
            "accuracy": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "per_fold": [float(s) for s in scores],
            "labels": label_set,
            "counts": counts,
            "confusion_matrix": cm.tolist(),
            "report": {k: v for k, v in report.items() if isinstance(v, dict)},
            "top_features": top_features,
            "n_samples": len(X),
        }

    def predict(self, segment):
        if not self.is_loaded:
            return {"error": "No model loaded. Train first."}

        features = extract_features(segment, sample_rate=SAMPLE_RATE)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

        prediction = self.clf.predict(features)[0]
        probas = self.clf.predict_proba(features)[0]
        confidence = float(max(probas))

        top3_idx = np.argsort(probas)[-3:][::-1]
        top3 = [{"label": self.labels[i], "prob": float(probas[i])} for i in top3_idx]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top3": top3,
        }

    def info(self):
        return {
            "loaded": self.is_loaded,
            "accuracy": self.accuracy,
            "labels": self.labels,
            "n_samples": self.n_samples,
            "path": self.model_path,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SpellEngine — Trie + LLM
# ═══════════════════════════════════════════════════════════════════════════════

def word_to_groups(word):
    return [LETTER_TO_GROUP[c] for c in word.lower() if c in LETTER_TO_GROUP]


def word_freq_score(w):
    return FREQ_RANK.get(w, 10000 + len(w))


class Trie:
    def __init__(self, dictionary):
        self.root = {}
        self.word_count = 0
        for word in dictionary:
            groups = word_to_groups(word)
            if groups:
                self._insert(groups, word)
                self.word_count += 1

    def _insert(self, groups, word):
        node = self.root
        for g in groups:
            if g not in node:
                node[g] = {"_words": []}
            node = node[g]
        node["_words"].append(word)

    def exact_lookup(self, groups):
        node = self.root
        for g in groups:
            if g not in node:
                return []
            node = node[g]
        return sorted(node.get("_words", []), key=word_freq_score)

    def prefix_count(self, groups):
        node = self.root
        for g in groups:
            if g not in node:
                return 0
            node = node[g]
        return self._count_words(node)

    def _count_words(self, node):
        count = len(node.get("_words", []))
        for key, child in node.items():
            if key != "_words" and isinstance(child, dict):
                count += self._count_words(child)
        return count

    def fuzzy_lookup(self, groups, max_errors=1):
        results = []
        for word_groups, words in self._all_words_with_groups():
            if len(word_groups) != len(groups):
                continue
            errors = sum(1 for a, b in zip(groups, word_groups) if a != b)
            if errors <= max_errors:
                for w in words:
                    results.append((w, errors))
        results.sort(key=lambda x: (x[1], word_freq_score(x[0])))
        return results

    def _all_words_with_groups(self):
        def _walk(node, path):
            if node.get("_words"):
                yield (list(path), node["_words"])
            for key, child in node.items():
                if key != "_words" and isinstance(child, dict):
                    path.append(key)
                    yield from _walk(child, path)
                    path.pop()
        yield from _walk(self.root, [])


class LLMPicker:
    def __init__(self):
        self.client = None
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=API_KEY)
        except Exception:
            pass

    def pick_word(self, context, candidates, max_candidates=10):
        if not self.client or not candidates:
            return candidates[0] if candidates else None
        cands = candidates[:max_candidates]
        system = ("You complete sentences by picking the single best word from a list. "
                  "Output ONLY the word, nothing else.")
        user_msg = (f'Sentence so far: "{context} ___"\n'
                    f'Candidate words: {", ".join(cands)}\nBest word:')
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_msg}],
                max_tokens=15, temperature=0,
            )
            answer = resp.choices[0].message.content.strip().lower()
            if answer in cands:
                return answer
            for c in cands:
                if c.startswith(answer) or answer.startswith(c):
                    return c
            return cands[0]
        except Exception:
            return cands[0]

    def expand_command(self, keywords):
        if not self.client:
            return " ".join(keywords)
        system = ("You expand keyword sequences into natural English sentences. "
                  "Output ONLY the sentence, nothing else.")
        user_msg = f'Keywords: {" ".join(keywords)}\nSentence:'
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_msg}],
                max_tokens=50, temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return " ".join(keywords)


class SpellEngine:
    """Trie + LLM integration for word prediction."""

    def __init__(self):
        self.dictionary = self._load_dictionary()
        self.trie = Trie(self.dictionary)
        self.llm = LLMPicker()

    def _load_dictionary(self):
        words = set()
        valid_chars = set(LETTER_TO_GROUP.keys())
        # Use COMMON_WORDS for fast startup; skip large system dictionaries
        for w in COMMON_WORDS:
            if set(w) <= valid_chars:
                words.add(w)
        custom_path = os.path.join(PROJECT_ROOT, "data", "custom_dictionary.txt")
        if os.path.exists(custom_path):
            with open(custom_path) as f:
                for line in f:
                    w = line.strip().lower()
                    if w and w.isalpha() and set(w) <= valid_chars:
                        words.add(w)
        return words

    def lookup(self, groups, context=""):
        exact = self.trie.exact_lookup(groups)
        fuzzy = self.trie.fuzzy_lookup(groups, max_errors=1)
        fuzzy_words = [w for w, _ in fuzzy]
        all_candidates = list(dict.fromkeys(exact + fuzzy_words))

        prefix_count = self.trie.prefix_count(groups) if groups else self.trie.word_count

        result = {
            "exact": exact[:20],
            "fuzzy": fuzzy_words[:20],
            "all_candidates": all_candidates[:20],
            "prefix_count": prefix_count,
            "total_candidates": len(all_candidates),
        }

        if all_candidates:
            if len(all_candidates) == 1:
                result["best"] = all_candidates[0]
            else:
                result["best"] = self.llm.pick_word(context, all_candidates[:10])

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="SilentPilot EMG App")

# Global state (set in main)
stream: EMGStream = None
datastore: DataStore = None
model_mgr: ModelManager = None
spell_engine: SpellEngine = None


# ── Pydantic models ──

class CaptureAutoRequest(BaseModel):
    label: str
    letter: str = ""
    countdown: float = 3.0

class DeleteRequest(BaseModel):
    idx: int = -1
    label: str = ""
    all: bool = False

class SpellLookupRequest(BaseModel):
    groups: list[str]
    context: str = ""


# ── REST Endpoints ──

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/api/status")
async def get_status():
    summary = datastore.summary() if datastore else {}
    return {
        "connected": stream is not None,
        "recording": stream.is_recording if stream else False,
        "total_samples": len(datastore.segments) if datastore else 0,
        "model_loaded": model_mgr.is_loaded if model_mgr else False,
        "model_accuracy": model_mgr.accuracy if model_mgr else 0,
        "summary": summary,
    }


@app.post("/api/capture/auto")
async def capture_auto(req: CaptureAutoRequest):
    if not stream:
        return JSONResponse({"error": "Not connected"}, 400)

    # Wait for countdown (client handles visual countdown, we just wait)
    await asyncio.sleep(req.countdown)

    # Record
    stream.start_rec()
    await asyncio.sleep(RECORD_SEC)
    raw = stream.stop_rec()

    # Smart crop
    cropped = smart_crop(raw)
    raw_n = len(raw)
    crop_n = len(cropped)

    # Energy
    e0 = float(np.sqrt(np.mean(cropped[:, 0].astype(float) ** 2))) if crop_n > 0 else 0
    e1 = float(np.sqrt(np.mean(cropped[:, 1].astype(float) ** 2))) if crop_n > 0 else 0

    # Downsample waveform for display (send max 500 points)
    step = max(1, raw_n // 500)
    raw_ch0 = raw[::step, 0].tolist()
    raw_ch1 = raw[::step, 1].tolist()

    step_c = max(1, crop_n // 500)
    crop_ch0 = cropped[::step_c, 0].tolist()
    crop_ch1 = cropped[::step_c, 1].tolist()

    # Comparison with existing samples of same class
    comparison = datastore.compare_sample(cropped, req.label)

    return {
        "raw_length": raw_n,
        "crop_length": crop_n,
        "crop_pct": crop_n / raw_n * 100 if raw_n > 0 else 100,
        "energy_ch0": e0,
        "energy_ch1": e1,
        "raw_ch0": raw_ch0,
        "raw_ch1": raw_ch1,
        "crop_ch0": crop_ch0,
        "crop_ch1": crop_ch1,
        "label": req.label,
        "letter": req.letter,
        "segment": cropped.tolist(),  # full segment for storing
        "comparison": comparison,
    }


@app.post("/api/capture/accept")
async def capture_accept(data: dict):
    seg = np.array(data["segment"], dtype=np.float64)
    label = data["label"]
    idx = datastore.add(seg, label)
    return {"idx": idx, "total": len(datastore.segments), "summary": datastore.summary()}


@app.get("/api/data/summary")
async def data_summary():
    return datastore.summary()


@app.post("/api/data/delete")
async def data_delete(req: DeleteRequest):
    if req.all:
        datastore.delete_all()
    elif req.label:
        datastore.delete_label(req.label)
    elif req.idx >= 0:
        datastore.delete_sample(req.idx)
    return {"total": len(datastore.segments), "summary": datastore.summary()}


@app.get("/api/data/sample/{idx}")
async def data_sample(idx: int):
    s = datastore.get_sample(idx)
    if s is None:
        return JSONResponse({"error": "Sample not found"}, 404)
    return s


@app.get("/api/data/samples/{label}")
async def data_samples_for_label(label: str):
    return datastore.get_samples_for_label(label)


@app.post("/api/train")
async def train_model():
    result = model_mgr.train(datastore)
    return result


@app.get("/api/model/info")
async def model_info():
    return model_mgr.info()


@app.get("/api/model/stats")
async def model_stats():
    """Return full model evaluation stats from existing data + model."""
    if not model_mgr.is_loaded or not datastore or len(datastore.segments) == 0:
        return {"error": "No model or data available"}

    from sklearn.metrics import classification_report, confusion_matrix

    segments = datastore.segments
    labels = datastore.labels

    X = np.array([extract_features(seg, sample_rate=SAMPLE_RATE) for seg in segments])
    y = np.array(labels)

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    label_set = sorted(set(y))
    counts = {l: int(np.sum(y == l)) for l in label_set}

    y_pred = model_mgr.clf.predict(X)
    report = classification_report(y, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=label_set)

    importances = model_mgr.clf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    top_features = [{"index": int(i), "importance": float(importances[i])} for i in top_idx]

    return {
        "accuracy": model_mgr.accuracy,
        "labels": label_set,
        "counts": counts,
        "confusion_matrix": cm.tolist(),
        "report": {k: v for k, v in report.items() if isinstance(v, dict)},
        "top_features": top_features,
        "n_samples": len(X),
    }


@app.post("/api/predict")
async def predict_gesture():
    if not stream:
        return JSONResponse({"error": "Not connected"}, 400)
    if not model_mgr.is_loaded:
        return JSONResponse({"error": "No model loaded"}, 400)

    stream.start_rec()
    await asyncio.sleep(RECORD_SEC)
    raw = stream.stop_rec()
    cropped = smart_crop(raw)

    result = model_mgr.predict(cropped)
    result["raw_length"] = len(raw)
    result["crop_length"] = len(cropped)
    result["energy_ch0"] = float(np.sqrt(np.mean(cropped[:, 0].astype(float) ** 2)))
    result["energy_ch1"] = float(np.sqrt(np.mean(cropped[:, 1].astype(float) ** 2)))

    # Downsample for display
    step = max(1, len(cropped) // 300)
    result["ch0"] = cropped[::step, 0].tolist()
    result["ch1"] = cropped[::step, 1].tolist()

    return result


@app.post("/api/spell/lookup")
async def spell_lookup(req: SpellLookupRequest):
    return spell_engine.lookup(req.groups, req.context)


# ── WebSocket for live streaming ──

@app.websocket("/ws/emg")
async def ws_emg(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if stream:
                data = stream.snapshot(last_n=SAMPLE_RATE * PLOT_HISTORY)
                n = len(data)
                # Downsample for transmission (max ~300 points)
                step = max(1, n // 300)
                ch0 = data[::step, 0].tolist()
                ch1 = data[::step, 1].tolist()

                # Sensor health
                recent = data[-min(50, n):]
                mu0 = float(np.mean(recent[:, 0]))
                mu1 = float(np.mean(recent[:, 1]))

                health0 = "ok" if 3 < mu0 < 500 else ("high" if mu0 >= 500 else "none")
                health1 = "ok" if 3 < mu1 < 500 else ("high" if mu1 >= 500 else "none")

                await websocket.send_json({
                    "ch0": ch0, "ch1": ch1,
                    "n_raw": n,
                    "mu0": round(mu0, 1), "mu1": round(mu1, 1),
                    "health0": health0, "health1": health1,
                    "recording": stream.is_recording,
                })
            await asyncio.sleep(1.0 / WS_FPS)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML Frontend (embedded)
# ═══════════════════════════════════════════════════════════════════════════════

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SilentPilot - EMG Calibration</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0f1117;
  --bg2: #161b22;
  --bg3: #1c2128;
  --border: #30363d;
  --text: #e6edf3;
  --text2: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --purple: #a855f7;
  --cyan: #06b6d4;
  --open: #ef4444;
  --close: #22c55e;
  --tip: #06b6d4;
  --back: #a855f7;
  --lips: #eab308;
  --rest: #6b7280;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
  background: var(--bg); color: var(--text);
  min-height: 100vh;
}
.header {
  background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center; gap: 20px;
}
.header h1 { font-size: 18px; font-weight: 600; }
.header .status { display: flex; gap: 16px; margin-left: auto; font-size: 12px; }
.header .status .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }
.header .status .dot.green { background: var(--green); }
.header .status .dot.red { background: var(--red); }
.header .status .dot.yellow { background: var(--yellow); }

.tabs {
  display: flex; background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 0 16px;
}
.tab {
  padding: 10px 20px; cursor: pointer; font-size: 13px; color: var(--text2);
  border-bottom: 2px solid transparent; transition: all 0.15s;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

.tab-content { display: none; padding: 20px; height: calc(100vh - 100px); overflow-y: auto; }
.tab-content.active { display: block; }

/* Cards */
.card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-bottom: 16px;
}
.card h3 { font-size: 14px; color: var(--text2); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }

/* Buttons */
.btn {
  padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg3); color: var(--text); cursor: pointer; font-size: 13px;
  font-family: inherit; transition: all 0.15s;
}
.btn:hover { border-color: var(--accent); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn.primary { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }
.btn.primary:hover { opacity: 0.9; }
.btn.danger { background: #da3633; color: #fff; border-color: #da3633; }
.btn.danger:hover { opacity: 0.9; }
.btn.success { background: var(--green); color: #000; border-color: var(--green); }
.btn.success:hover { opacity: 0.9; }

/* Grid layouts */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.grid-auto { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 16px; }

/* Chart container */
.chart-container { position: relative; width: 100%; height: 200px; }
.chart-container canvas { width: 100% !important; height: 100% !important; }

/* Letter cards */
.letter-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.letter-card {
  width: 52px; height: 52px; border-radius: 8px; display: flex;
  flex-direction: column; align-items: center; justify-content: center;
  cursor: pointer; border: 2px solid transparent; transition: all 0.15s;
  font-weight: 600; font-size: 18px;
}
.letter-card:hover { transform: scale(1.1); border-color: #fff3; }
.letter-card .count { font-size: 9px; font-weight: 400; opacity: 0.7; }
.letter-card.OPEN { background: #ef444430; color: var(--open); }
.letter-card.CLOSE { background: #22c55e30; color: var(--close); }
.letter-card.TIP { background: #06b6d430; color: var(--tip); }
.letter-card.BACK { background: #a855f730; color: var(--back); }
.letter-card.LIPS { background: #eab30830; color: var(--lips); }
.letter-card.REST { background: #6b728030; color: var(--rest); }

/* Progress bars */
.progress-bar {
  height: 6px; border-radius: 3px; background: var(--bg);
  overflow: hidden; margin-top: 4px;
}
.progress-bar .fill { height: 100%; border-radius: 3px; transition: width 0.3s; }

/* Sensor health */
.sensor-indicator {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 4px; font-size: 12px;
}
.sensor-indicator.ok { background: #3fb95020; color: var(--green); }
.sensor-indicator.high { background: #d2992220; color: var(--yellow); }
.sensor-indicator.none { background: #f8514920; color: var(--red); }

/* Countdown overlay */
.countdown-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.85); display: flex; flex-direction: column;
  align-items: center; justify-content: center; z-index: 1000;
}
.countdown-number { font-size: 120px; font-weight: 700; color: var(--accent); }
.countdown-label { font-size: 24px; color: var(--text2); margin-top: 16px; }
.countdown-go { color: var(--green); }
.countdown-recording { color: var(--red); animation: pulse 0.5s infinite; }
@keyframes pulse { 50% { opacity: 0.5; } }

/* Confusion matrix */
.cm-grid { display: grid; gap: 2px; font-size: 11px; text-align: center; }
.cm-cell {
  padding: 6px; border-radius: 2px; min-width: 40px;
}
.cm-header { font-weight: 600; color: var(--text2); }

/* Confidence bar */
.conf-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.conf-bar .bar-bg { flex: 1; height: 20px; background: var(--bg); border-radius: 4px; overflow: hidden; }
.conf-bar .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.conf-bar .label { width: 60px; font-size: 12px; }
.conf-bar .value { width: 40px; font-size: 12px; text-align: right; }

/* Spell groups display */
.group-sequence { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.group-tag {
  padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: 600;
}
.group-tag.OPEN { background: #ef444440; color: var(--open); }
.group-tag.CLOSE { background: #22c55e40; color: var(--close); }
.group-tag.TIP { background: #06b6d440; color: var(--tip); }
.group-tag.BACK { background: #a855f740; color: var(--back); }
.group-tag.LIPS { background: #eab30840; color: var(--lips); }

/* Modal */
.modal-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.7); display: flex; align-items: center;
  justify-content: center; z-index: 999;
}
.modal { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; }
.modal h2 { margin-bottom: 16px; font-size: 18px; }

/* Big number */
.big-number { font-size: 64px; font-weight: 700; text-align: center; }
.big-label { font-size: 14px; color: var(--text2); text-align: center; }

/* Toast */
.toast {
  position: fixed; bottom: 24px; right: 24px; padding: 12px 20px;
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  font-size: 13px; z-index: 2000; transition: opacity 0.3s;
}
.toast.success { border-color: var(--green); }
.toast.error { border-color: var(--red); }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>SilentPilot</h1>
  <span style="font-size:12px;color:var(--text2)">EMG Calibration & Inference</span>
  <div class="status">
    <span><span class="dot green" id="dot-serial"></span>Serial</span>
    <span><span class="dot" id="dot-model"></span>Model: <span id="model-acc-header">--</span></span>
    <span id="sample-count-header">0 samples</span>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('monitor')">Live Monitor</div>
  <div class="tab" onclick="switchTab('capture')">Data Capture</div>
  <div class="tab" onclick="switchTab('train')">Training</div>
  <div class="tab" onclick="switchTab('infer')">Inference</div>
</div>

<!-- ═══════════════ TAB 1: LIVE MONITOR ═══════════════ -->
<div id="tab-monitor" class="tab-content active">
  <div class="card">
    <h3>Live EMG Waveform</h3>
    <div style="display:flex;gap:16px;margin-bottom:12px;">
      <span class="sensor-indicator" id="health-ch0">CH0 (Chin): --</span>
      <span class="sensor-indicator" id="health-ch1">CH1 (Cheek): --</span>
    </div>
    <div class="chart-container" style="height:300px">
      <canvas id="chart-monitor"></canvas>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <h3>CH0 - Chin / Submental</h3>
      <div class="big-number" id="baseline-ch0" style="color:var(--cyan)">--</div>
      <div class="big-label">Baseline (mean last 0.2s)</div>
    </div>
    <div class="card">
      <h3>CH1 - Cheek / Perioral</h3>
      <div class="big-number" id="baseline-ch1" style="color:#ff6b6b">--</div>
      <div class="big-label">Baseline (mean last 0.2s)</div>
    </div>
  </div>
  <div class="card" style="background:#1a1500;border-color:#44380a">
    <h3 style="color:var(--yellow)">Setup Tips</h3>
    <ul style="font-size:13px;color:var(--text2);list-style:disc;padding-left:20px;line-height:1.8">
      <li>Baseline should be 20-60 for chin, 15-50 for cheek when relaxed</li>
      <li>If a channel reads 0 or &gt;500 at rest, check electrode contact</li>
      <li>Try clenching jaw &mdash; both channels should spike clearly</li>
      <li>If signal is clipping near 1000, reduce GAIN on MyoWare</li>
    </ul>
  </div>
</div>

<!-- ═══════════════ TAB 2: DATA CAPTURE ═══════════════ -->
<div id="tab-capture" class="tab-content">
  <div class="grid-2">
    <!-- Left: Letter grid + controls -->
    <div>
      <div class="card">
        <h3>Select Letter to Capture</h3>
        <div style="margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap;">
          <button class="btn" onclick="captureRest()">Capture REST</button>
          <button class="btn primary" id="btn-batch" onclick="startBatchCapture()">Collect All (Batch)</button>
          <button class="btn danger" onclick="confirmDeleteAll()">Delete All Data</button>
        </div>
        <div id="letter-grid-container"></div>
      </div>
      <!-- Progress Summary -->
      <div class="card">
        <h3>Progress</h3>
        <div id="progress-bars"></div>
      </div>
    </div>
    <!-- Right: Last captured sample + comparison -->
    <div>
      <div class="card" id="capture-review-card">
        <h3>Sample Review</h3>
        <div id="capture-result" style="color:var(--text2);font-size:13px;text-align:center;padding:40px 0;">
          Click a letter to begin capturing
        </div>
      </div>
      <div class="card" id="capture-accept-card" style="display:none">
        <div style="display:flex;gap:8px;margin-bottom:12px;">
          <button class="btn success" onclick="acceptCapture()" id="btn-accept" style="flex:1;padding:10px;font-size:14px;">Accept</button>
          <button class="btn danger" onclick="discardCapture()" style="flex:1;padding:10px;font-size:14px;">Discard</button>
          <button class="btn" onclick="retryCapture()" id="btn-retry" style="flex:1;padding:10px;font-size:14px;">Retry</button>
        </div>
        <!-- Warnings -->
        <div id="capture-warnings" style="margin-bottom:8px;"></div>
        <!-- Feature comparison -->
        <div id="capture-comparison"></div>
        <!-- Raw stats -->
        <div id="capture-stats" style="margin-top:8px;font-size:11px;color:var(--text2);border-top:1px solid var(--border);padding-top:8px;"></div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════ TAB 3: TRAINING ═══════════════ -->
<div id="tab-train" class="tab-content">
  <div class="card" style="text-align:center;">
    <button class="btn primary" onclick="trainModel()" id="btn-train" style="font-size:16px;padding:12px 32px;">
      Train Model
    </button>
    <div id="train-spinner" style="display:none;margin-top:16px;color:var(--text2);">Training...</div>
  </div>

  <div id="train-results" style="display:none">
    <div class="grid-3">
      <div class="card">
        <div class="big-number" id="train-accuracy" style="color:var(--green)">--</div>
        <div class="big-label">CV Accuracy</div>
      </div>
      <div class="card">
        <div class="big-number" id="train-samples" style="color:var(--accent)">--</div>
        <div class="big-label">Samples</div>
      </div>
      <div class="card">
        <div class="big-number" id="train-classes" style="color:var(--purple)">--</div>
        <div class="big-label">Classes</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>Confusion Matrix</h3>
        <div id="confusion-matrix"></div>
      </div>
      <div class="card">
        <h3>Per-Class Metrics</h3>
        <div id="class-metrics"></div>
      </div>
    </div>

    <div class="card">
      <h3>Top Features</h3>
      <div id="feature-importance"></div>
    </div>
  </div>

  <!-- Sample Browser -->
  <div class="card">
    <h3>Sample Browser</h3>
    <div style="display:flex;gap:8px;margin-bottom:12px;">
      <select id="browse-class" class="btn" onchange="loadClassSamples()">
        <option value="">Select class...</option>
      </select>
    </div>
    <div id="sample-browser" style="font-size:13px;color:var(--text2)">Select a class to browse samples</div>
  </div>
</div>

<!-- ═══════════════ TAB 4: INFERENCE ═══════════════ -->
<div id="tab-infer" class="tab-content">
  <div class="grid-2">
    <!-- Single gesture test -->
    <div class="card">
      <h3>Single Gesture Test</h3>
      <button class="btn primary" onclick="predictSingle()" id="btn-predict" style="width:100%;padding:12px;font-size:15px;">
        Record & Classify
      </button>
      <div id="predict-result" style="margin-top:16px;"></div>
    </div>

    <!-- Prediction detail -->
    <div class="card">
      <h3>Prediction Detail</h3>
      <div id="predict-detail" style="color:var(--text2);font-size:13px;">
        Click "Record & Classify" to test
      </div>
    </div>
  </div>

  <!-- Word Spelling -->
  <div class="card">
    <h3>Word Spelling Mode</h3>
    <div style="background:var(--bg);padding:12px;border-radius:6px;margin-bottom:12px;">
      <div style="font-size:12px;color:var(--text2);margin-bottom:4px;">Sentence:</div>
      <div id="spell-sentence" style="font-size:18px;min-height:24px;">--</div>
    </div>
    <div style="margin-bottom:12px;">
      <div style="font-size:12px;color:var(--text2);margin-bottom:4px;">Current groups:</div>
      <div class="group-sequence" id="spell-groups">
        <span style="color:var(--text2);font-size:13px;">No groups yet</span>
      </div>
    </div>
    <div style="font-size:12px;color:var(--text2);margin-bottom:8px;">
      Candidates: <span id="spell-candidates">--</span>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button class="btn primary" onclick="spellRecord()">Record Gesture</button>
      <button class="btn success" onclick="spellDone()">Done (Predict Word)</button>
      <button class="btn" onclick="spellUndo()">Undo</button>
      <button class="btn" onclick="spellClear()">Clear Word</button>
      <button class="btn" onclick="spellNewSentence()">New Sentence</button>
    </div>
    <div id="spell-result" style="margin-top:12px;"></div>
  </div>

  <!-- Quick test (simulate) -->
  <div class="card">
    <h3>Quick Group Test (No EMG)</h3>
    <div style="font-size:12px;color:var(--text2);margin-bottom:8px;">Click groups to simulate input without recording:</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button class="btn" style="color:var(--open)" onclick="spellAddGroup('OPEN')">OPEN</button>
      <button class="btn" style="color:var(--close)" onclick="spellAddGroup('CLOSE')">CLOSE</button>
      <button class="btn" style="color:var(--tip)" onclick="spellAddGroup('TIP')">TIP</button>
      <button class="btn" style="color:var(--back)" onclick="spellAddGroup('BACK')">BACK</button>
      <button class="btn" style="color:var(--lips)" onclick="spellAddGroup('LIPS')">LIPS</button>
    </div>
  </div>
</div>

<!-- Countdown overlay (hidden by default) -->
<div class="countdown-overlay" id="countdown-overlay" style="display:none">
  <div class="countdown-number" id="countdown-number">3</div>
  <div class="countdown-label" id="countdown-label">Get ready...</div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════

const GROUPS = ['REST','OPEN','CLOSE','TIP','BACK','LIPS'];
const ACTIVE_GROUPS = ['OPEN','CLOSE','TIP','BACK','LIPS'];
const LETTERS_BY_GROUP = {
  OPEN:['A','O','U','H'], CLOSE:['E','I','Y'],
  TIP:['T','D','N','L','S','Z'], BACK:['K','G','J','C','R'],
  LIPS:['B','M','F','P','W']
};
const GROUP_COLORS = {
  REST:'#6b7280', OPEN:'#ef4444', CLOSE:'#22c55e',
  TIP:'#06b6d4', BACK:'#a855f7', LIPS:'#eab308'
};

let pendingCapture = null;
let spellGroups = [];
let spellSentence = [];
let sampleCounts = {};
let batchRunning = false;

// ═══════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ═══════════════════════════════════════════════════════════════

function switchTab(id) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['monitor','capture','train','infer'][i] === id);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
}

// ═══════════════════════════════════════════════════════════════
//  WEBSOCKET + LIVE CHART
// ═══════════════════════════════════════════════════════════════

let ws = null;
let monitorChart = null;

function initWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws/emg');
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    updateMonitor(d);
  };
  ws.onclose = () => {
    setTimeout(initWebSocket, 2000);
  };
}

function initMonitorChart() {
  const ctx = document.getElementById('chart-monitor').getContext('2d');
  monitorChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label:'CH0 (Chin)', data:[], borderColor:'#06b6d4', borderWidth:1.2, pointRadius:0, tension:0.1 },
        { label:'CH1 (Cheek)', data:[], borderColor:'#ff6b6b', borderWidth:1.2, pointRadius:0, tension:0.1 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { display:false },
        y: { min:0, max:200, grid:{ color:'#30363d' }, ticks:{ color:'#666', font:{size:10} } }
      },
      plugins: {
        legend: { labels:{ color:'#8b949e', font:{size:11} } }
      }
    }
  });
}

function updateMonitor(d) {
  if (!monitorChart) return;
  const labels = d.ch0.map((_,i) => i);
  monitorChart.data.labels = labels;
  monitorChart.data.datasets[0].data = d.ch0;
  monitorChart.data.datasets[1].data = d.ch1;

  // Auto-scale Y
  const maxVal = Math.max(...d.ch0, ...d.ch1, 100);
  monitorChart.options.scales.y.max = Math.min(1024, maxVal * 1.3);
  monitorChart.update('none');

  // Update health indicators
  updateHealth('health-ch0', 'CH0 (Chin)', d.health0, d.mu0);
  updateHealth('health-ch1', 'CH1 (Cheek)', d.health1, d.mu1);

  // Update baselines
  document.getElementById('baseline-ch0').textContent = d.mu0.toFixed(0);
  document.getElementById('baseline-ch1').textContent = d.mu1.toFixed(0);

  // Header
  document.getElementById('dot-serial').className = 'dot green';
}

function updateHealth(id, name, status, value) {
  const el = document.getElementById(id);
  el.className = 'sensor-indicator ' + status;
  const statusText = status==='ok' ? 'OK' : status==='high' ? 'HIGH' : 'NO SIGNAL';
  el.textContent = name + ': ' + value.toFixed(0) + ' (' + statusText + ')';
}

// ═══════════════════════════════════════════════════════════════
//  LETTER GRID
// ═══════════════════════════════════════════════════════════════

function buildLetterGrid() {
  const container = document.getElementById('letter-grid-container');
  let html = '';
  for (const grp of ACTIVE_GROUPS) {
    html += '<div style="margin-bottom:12px;"><div style="font-size:11px;color:' + GROUP_COLORS[grp] +
            ';font-weight:600;margin-bottom:6px;">' + grp + '</div><div class="letter-grid">';
    for (const letter of LETTERS_BY_GROUP[grp]) {
      html += '<div class="letter-card ' + grp + '" onclick="captureLetter(\'' + letter + '\',\'' + grp +
              '\')" id="lcard-' + letter + '">' + letter +
              '<div class="count" id="lcount-' + letter + '">0</div></div>';
    }
    html += '</div></div>';
  }
  container.innerHTML = html;
}

function updateLetterCounts() {
  for (const grp of ACTIVE_GROUPS) {
    for (const letter of LETTERS_BY_GROUP[grp]) {
      const el = document.getElementById('lcount-' + letter);
      // We track counts per group, not per letter
    }
  }
}

// ═══════════════════════════════════════════════════════════════
//  DATA CAPTURE
// ═══════════════════════════════════════════════════════════════

async function captureLetter(letter, group) {
  showCountdown(letter, group);
  try {
    const resp = await fetch('/api/capture/auto', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ label: group, letter: letter, countdown: 3.0 })
    });
    const data = await resp.json();
    hideCountdown();
    showCaptureResult(data);
  } catch(err) {
    hideCountdown();
    showToast('Capture failed: ' + err.message, 'error');
  }
}

async function captureRest() {
  showCountdown('REST', 'REST');
  try {
    const resp = await fetch('/api/capture/auto', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ label: 'REST', letter: '', countdown: 3.0 })
    });
    const data = await resp.json();
    hideCountdown();
    showCaptureResult(data);
  } catch(err) {
    hideCountdown();
    showToast('Capture failed: ' + err.message, 'error');
  }
}

function showCountdown(letter, group) {
  const overlay = document.getElementById('countdown-overlay');
  const numEl = document.getElementById('countdown-number');
  const labelEl = document.getElementById('countdown-label');
  overlay.style.display = 'flex';

  let count = 3;
  labelEl.textContent = 'Prepare: ' + (letter || group);
  labelEl.className = 'countdown-label';
  numEl.textContent = count;
  numEl.style.color = 'var(--accent)';

  const timer = setInterval(() => {
    count--;
    if (count > 0) {
      numEl.textContent = count;
    } else if (count === 0) {
      numEl.textContent = 'GO';
      numEl.style.color = 'var(--green)';
      labelEl.textContent = 'Recording...';
      labelEl.className = 'countdown-label countdown-recording';
      // Play beep
      try {
        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        osc.frequency.value = 800;
        osc.connect(ctx.destination);
        osc.start();
        setTimeout(() => osc.stop(), 150);
      } catch(e) {}
    } else {
      clearInterval(timer);
    }
  }, 1000);
}

function hideCountdown() {
  document.getElementById('countdown-overlay').style.display = 'none';
}

function showCaptureResult(data) {
  pendingCapture = data;
  const resultDiv = document.getElementById('capture-result');
  const acceptCard = document.getElementById('capture-accept-card');
  const comp = data.comparison || {};
  const newF = comp.new || {};
  const nExist = comp.n_existing || 0;

  // ── Waveform chart ──
  let html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">' +
    '<span style="font-size:16px;font-weight:600;color:' + (GROUP_COLORS[data.label]||'#fff') + '">' +
    data.label + '</span>' +
    (data.letter ? '<span style="font-size:14px;color:var(--text2);">(' + data.letter + ')</span>' : '') +
    '<span style="margin-left:auto;font-size:11px;color:var(--text2);">' +
    data.crop_length + ' pts (' + data.crop_pct.toFixed(0) + '% kept)</span></div>' +
    '<div class="chart-container" style="height:150px"><canvas id="chart-capture"></canvas></div>';

  resultDiv.innerHTML = html;

  const ctx = document.getElementById('chart-capture').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.crop_ch0.map((_,i)=>i),
      datasets: [
        { label:'CH0', data:data.crop_ch0, borderColor:'#06b6d4', borderWidth:1.5, pointRadius:0 },
        { label:'CH1', data:data.crop_ch1, borderColor:'#ff6b6b', borderWidth:1.5, pointRadius:0 },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:false,
      scales: { x:{display:false}, y:{grid:{color:'#30363d'},ticks:{color:'#666'}} },
      plugins: { legend:{labels:{color:'#8b949e',font:{size:10}}} }
    }
  });

  // ── Warnings ──
  const warningsDiv = document.getElementById('capture-warnings');
  const warnings = comp.warnings || [];
  if (warnings.length > 0) {
    warningsDiv.innerHTML = warnings.map(w =>
      '<div style="background:#f8514920;border:1px solid #f8514940;border-radius:4px;padding:6px 10px;margin-bottom:4px;font-size:12px;color:var(--red);">&#9888; ' + w + '</div>'
    ).join('');
  } else if (nExist >= 2) {
    warningsDiv.innerHTML = '<div style="background:#3fb95015;border:1px solid #3fb95030;border-radius:4px;padding:6px 10px;font-size:12px;color:var(--green);">&#10003; Sample looks consistent with existing data</div>';
  } else {
    warningsDiv.innerHTML = '';
  }

  // ── Feature comparison bars ──
  let compHtml = '';
  if (nExist >= 2 && comp.existing) {
    compHtml += '<div style="font-size:11px;color:var(--text2);margin-bottom:6px;font-weight:600;">FEATURES vs ' + nExist + ' EXISTING SAMPLES</div>';
    const features = [
      { name: 'CH0 Energy', newVal: newF.energy_ch0, ex: comp.existing.energy_ch0, color: '#06b6d4' },
      { name: 'CH1 Energy', newVal: newF.energy_ch1, ex: comp.existing.energy_ch1, color: '#ff6b6b' },
      { name: 'CH0 Peak',   newVal: newF.peak_ch0,   ex: comp.existing.peak_ch0,   color: '#06b6d4' },
      { name: 'CH1 Peak',   newVal: newF.peak_ch1,   ex: comp.existing.peak_ch1,   color: '#ff6b6b' },
      { name: 'CH0 WaveLen', newVal: newF.waveform_length_ch0, ex: comp.existing.waveform_length_ch0, color: '#06b6d4' },
      { name: 'CH1 WaveLen', newVal: newF.waveform_length_ch1, ex: comp.existing.waveform_length_ch1, color: '#ff6b6b' },
    ];

    for (const f of features) {
      const maxRange = Math.max(f.ex.max * 1.3, f.newVal * 1.1, 1);
      const newPct = Math.min(100, f.newVal / maxRange * 100);
      const meanPct = Math.min(100, f.ex.mean / maxRange * 100);
      const minPct = Math.min(100, f.ex.min / maxRange * 100);
      const maxPct = Math.min(100, f.ex.max / maxRange * 100);

      // Check if outlier
      const isOutlier = f.ex.std > 0 && Math.abs(f.newVal - f.ex.mean) > 2 * f.ex.std;
      const borderStyle = isOutlier ? 'border-left:3px solid var(--red);' : '';

      compHtml += '<div style="margin-bottom:6px;padding:2px 0 2px 6px;' + borderStyle + '">' +
        '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text2);margin-bottom:2px;">' +
        '<span>' + f.name + '</span>' +
        '<span>new: ' + f.newVal.toFixed(0) + ' | avg: ' + f.ex.mean.toFixed(0) + ' &plusmn; ' + f.ex.std.toFixed(0) + '</span></div>' +
        '<div style="position:relative;height:14px;background:var(--bg);border-radius:3px;overflow:visible;">' +
        // Range band (min to max of existing)
        '<div style="position:absolute;left:' + minPct + '%;width:' + Math.max(1, maxPct - minPct) + '%;height:100%;background:' + f.color + '20;border-radius:2px;"></div>' +
        // Mean marker
        '<div style="position:absolute;left:' + meanPct + '%;width:2px;height:100%;background:' + f.color + '60;"></div>' +
        // New value marker
        '<div style="position:absolute;left:calc(' + newPct + '% - 4px);top:-1px;width:8px;height:16px;background:' + f.color + ';border-radius:2px;border:1px solid #fff3;"></div>' +
        '</div></div>';
    }

    // ── Cross-group comparison ──
    if (comp.other_groups && Object.keys(comp.other_groups).length > 0) {
      compHtml += '<div style="font-size:11px;color:var(--text2);margin:10px 0 4px;font-weight:600;">ENERGY vs OTHER GROUPS</div>';
      compHtml += '<div style="display:grid;grid-template-columns:60px 1fr 1fr;gap:2px;font-size:10px;">';
      compHtml += '<div style="color:var(--text2);">Group</div><div style="color:#06b6d4;">CH0</div><div style="color:#ff6b6b;">CH1</div>';

      // Current group first
      compHtml += '<div style="color:' + (GROUP_COLORS[data.label]||'#fff') + ';font-weight:600;">' + data.label + ' *</div>';
      compHtml += '<div style="font-weight:600;">' + newF.energy_ch0.toFixed(0) + (nExist >= 2 ? ' (avg ' + comp.existing.energy_ch0.mean.toFixed(0) + ')' : '') + '</div>';
      compHtml += '<div style="font-weight:600;">' + newF.energy_ch1.toFixed(0) + (nExist >= 2 ? ' (avg ' + comp.existing.energy_ch1.mean.toFixed(0) + ')' : '') + '</div>';

      for (const [g, stats] of Object.entries(comp.other_groups)) {
        compHtml += '<div style="color:' + (GROUP_COLORS[g]||'#fff') + ';">' + g + '</div>';
        compHtml += '<div>' + stats.mean_e0.toFixed(0) + '</div>';
        compHtml += '<div>' + stats.mean_e1.toFixed(0) + '</div>';
      }
      compHtml += '</div>';
    }

  } else if (nExist < 2) {
    compHtml = '<div style="font-size:11px;color:var(--text2);padding:4px 0;">Need 2+ existing samples for comparison stats</div>';
  }

  document.getElementById('capture-comparison').innerHTML = compHtml;

  // ── Raw stats ──
  document.getElementById('capture-stats').innerHTML =
    'Raw: ' + data.raw_length + ' samples &rarr; Cropped: ' + data.crop_length +
    ' (' + data.crop_pct.toFixed(0) + '% kept) | ' +
    'Energy CH0: ' + data.energy_ch0.toFixed(1) + ' | CH1: ' + data.energy_ch1.toFixed(1) +
    ' | Length: ' + newF.length + ' pts';

  acceptCard.style.display = 'block';

  // Resolve batch promise if waiting
  if (_batchResolve) _batchResolve('shown');
}

// ── Batch capture uses promises to wait for user decision ──
let _batchResolve = null;
let _batchDecision = null;

function _resolveBatch(decision) {
  _batchDecision = decision;
  if (_batchResolve) { _batchResolve(decision); _batchResolve = null; }
}

async function acceptCapture() {
  if (!pendingCapture) return;
  try {
    const resp = await fetch('/api/capture/accept', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ segment: pendingCapture.segment, label: pendingCapture.label })
    });
    const data = await resp.json();
    document.getElementById('capture-accept-card').style.display = 'none';
    showToast('Sample saved (' + data.total + ' total)', 'success');
    pendingCapture = null;
    refreshProgress(data.summary);
    updateHeaderCounts(data.total);
    _resolveBatch('accepted');
  } catch(err) {
    showToast('Save failed: ' + err.message, 'error');
  }
}

function discardCapture() {
  pendingCapture = null;
  document.getElementById('capture-accept-card').style.display = 'none';
  document.getElementById('capture-result').innerHTML =
    '<div style="color:var(--text2);text-align:center;padding:40px 0;">Sample discarded</div>';
  document.getElementById('capture-warnings').innerHTML = '';
  document.getElementById('capture-comparison').innerHTML = '';
  _resolveBatch('discarded');
}

function retryCapture() {
  if (!pendingCapture) return;
  const letter = pendingCapture.letter;
  const group = pendingCapture.label;
  pendingCapture = null;
  document.getElementById('capture-accept-card').style.display = 'none';
  _resolveBatch('retry');
  // Re-trigger capture for same letter
  if (letter && letter !== 'REST') {
    captureLetter(letter, group);
  } else {
    captureRest();
  }
}

async function confirmDeleteAll() {
  if (!confirm('Delete ALL captured data? This cannot be undone.')) return;
  const resp = await fetch('/api/data/delete', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({all:true})
  });
  const data = await resp.json();
  refreshProgress(data.summary);
  updateHeaderCounts(data.total);
  showToast('All data deleted', 'success');
}

// ═══════════════════════════════════════════════════════════════
//  BATCH CAPTURE — pauses for review after each sample
// ═══════════════════════════════════════════════════════════════

function waitForUserDecision() {
  return new Promise(resolve => { _batchResolve = resolve; });
}

async function batchCaptureOne(letter, group) {
  // Capture
  showCountdown(letter || 'REST', group);
  try {
    const resp = await fetch('/api/capture/auto', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ label: group, letter: letter || '', countdown: 3.0 })
    });
    const data = await resp.json();
    hideCountdown();
    showCaptureResult(data);
  } catch(err) {
    hideCountdown();
    showToast('Capture failed: ' + err.message, 'error');
    return 'error';
  }

  // Wait for user to click Accept, Discard, or Retry
  const decision = await waitForUserDecision();
  return decision;
}

async function startBatchCapture() {
  if (batchRunning) { batchRunning = false; return; }
  const samplesPerLetter = parseInt(prompt('Samples per letter?', '5'));
  if (!samplesPerLetter || samplesPerLetter < 1) return;

  batchRunning = true;
  document.getElementById('btn-batch').textContent = 'Stop Batch';

  // Build queue: [ {letter, group}, ... ]
  const queue = [];
  for (let s = 0; s < samplesPerLetter; s++) queue.push({letter:'', group:'REST'});
  for (const grp of ACTIVE_GROUPS) {
    for (const letter of LETTERS_BY_GROUP[grp]) {
      for (let s = 0; s < samplesPerLetter; s++) {
        queue.push({letter, group: grp});
      }
    }
  }

  let i = 0;
  while (i < queue.length && batchRunning) {
    const item = queue[i];
    const progress = '(' + (i+1) + '/' + queue.length + ')';
    document.getElementById('btn-batch').textContent = 'Stop Batch ' + progress;

    const decision = await batchCaptureOne(item.letter, item.group);

    if (decision === 'retry') {
      // Don't advance — retry same item
      continue;
    } else if (decision === 'accepted' || decision === 'discarded') {
      i++;
      await sleep(300);
    } else {
      // Error or unknown
      i++;
    }
  }

  batchRunning = false;
  document.getElementById('btn-batch').textContent = 'Collect All (Batch)';
  if (i >= queue.length) showToast('Batch capture complete!', 'success');
  else showToast('Batch capture stopped', 'error');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════════
//  PROGRESS BARS
// ═══════════════════════════════════════════════════════════════

function refreshProgress(summary) {
  sampleCounts = summary;
  const container = document.getElementById('progress-bars');
  let html = '';
  for (const grp of GROUPS) {
    const info = summary[grp] || {count:0, mean_energy:0};
    const pct = Math.min(100, info.count / 20 * 100);
    html += '<div style="margin-bottom:10px;display:flex;align-items:center;gap:10px;">' +
      '<span style="width:50px;font-size:12px;color:' + GROUP_COLORS[grp] + ';font-weight:600;">' + grp + '</span>' +
      '<div style="flex:1;"><div class="progress-bar"><div class="fill" style="width:' + pct +
      '%;background:' + GROUP_COLORS[grp] + '"></div></div></div>' +
      '<span style="width:40px;font-size:12px;text-align:right;">' + info.count + '</span>' +
      '<span style="width:60px;font-size:10px;color:var(--text2);">E:' + info.mean_energy.toFixed(0) + '</span>' +
      (info.count > 0 ? '<button class="btn danger" style="font-size:9px;padding:2px 6px;" onclick="deleteCategory(\'' + grp + '\')">&times;</button>' : '<span style="width:28px;"></span>') +
      '</div>';
  }
  container.innerHTML = html;
}

async function deleteCategory(label) {
  const info = sampleCounts[label] || {count:0};
  if (!confirm('Delete all ' + info.count + ' samples for ' + label + '?')) return;
  const resp = await fetch('/api/data/delete', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({label: label})
  });
  const data = await resp.json();
  refreshProgress(data.summary);
  updateHeaderCounts(data.total);
  showToast('Deleted ' + label + ' data', 'success');
}

async function loadProgress() {
  try {
    const resp = await fetch('/api/data/summary');
    const data = await resp.json();
    refreshProgress(data);
    let total = 0;
    for (const g of GROUPS) total += (data[g]||{count:0}).count;
    updateHeaderCounts(total);
  } catch(e) {}
}

function updateHeaderCounts(total) {
  document.getElementById('sample-count-header').textContent = total + ' samples';
}

// ═══════════════════════════════════════════════════════════════
//  TRAINING
// ═══════════════════════════════════════════════════════════════

async function trainModel() {
  document.getElementById('btn-train').disabled = true;
  document.getElementById('train-spinner').style.display = 'block';
  document.getElementById('train-results').style.display = 'none';

  try {
    const resp = await fetch('/api/train', {method:'POST'});
    const data = await resp.json();

    if (data.error) {
      showToast(data.error, 'error');
      return;
    }

    document.getElementById('train-results').style.display = 'block';
    document.getElementById('train-accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
    document.getElementById('train-samples').textContent = data.n_samples;
    document.getElementById('train-classes').textContent = data.labels.length;

    // Color accuracy
    const accEl = document.getElementById('train-accuracy');
    accEl.style.color = data.accuracy > 0.8 ? 'var(--green)' : data.accuracy > 0.6 ? 'var(--yellow)' : 'var(--red)';

    // Confusion matrix
    buildConfusionMatrix(data.confusion_matrix, data.labels);

    // Class metrics
    buildClassMetrics(data.report, data.labels);

    // Feature importance
    buildFeatureImportance(data.top_features);

    // Update header
    document.getElementById('dot-model').className = 'dot green';
    document.getElementById('model-acc-header').textContent = (data.accuracy * 100).toFixed(1) + '%';

    // Populate sample browser dropdown
    const sel = document.getElementById('browse-class');
    sel.innerHTML = '<option value="">Select class...</option>';
    for (const l of data.labels) {
      sel.innerHTML += '<option value="' + l + '">' + l + ' (' + (data.counts[l]||0) + ')</option>';
    }

    showToast('Model trained: ' + (data.accuracy*100).toFixed(1) + '% accuracy', 'success');
  } catch(err) {
    showToast('Training failed: ' + err.message, 'error');
  } finally {
    document.getElementById('btn-train').disabled = false;
    document.getElementById('train-spinner').style.display = 'none';
  }
}

function buildConfusionMatrix(cm, labels) {
  const n = labels.length;
  const container = document.getElementById('confusion-matrix');
  const maxVal = Math.max(...cm.flat(), 1);

  let html = '<div class="cm-grid" style="grid-template-columns:80px repeat(' + n + ', 1fr);">';
  // Header row
  html += '<div class="cm-cell cm-header"></div>';
  for (const l of labels) html += '<div class="cm-cell cm-header" style="color:' + (GROUP_COLORS[l]||'#fff') + '">' + l.slice(0,4) + '</div>';

  for (let i = 0; i < n; i++) {
    html += '<div class="cm-cell cm-header" style="color:' + (GROUP_COLORS[labels[i]]||'#fff') + '">' + labels[i].slice(0,4) + '</div>';
    for (let j = 0; j < n; j++) {
      const v = cm[i][j];
      const intensity = v / maxVal;
      const bg = i === j ?
        'rgba(59,185,80,' + (0.1 + intensity*0.6) + ')' :
        'rgba(248,81,73,' + (intensity*0.5) + ')';
      html += '<div class="cm-cell" style="background:' + bg + '">' + v + '</div>';
    }
  }
  html += '</div>';
  container.innerHTML = html;
}

function buildClassMetrics(report, labels) {
  const container = document.getElementById('class-metrics');
  let html = '<table style="width:100%;font-size:12px;border-collapse:collapse;">';
  html += '<tr style="color:var(--text2);"><th style="text-align:left;padding:4px;">Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>';
  for (const l of labels) {
    const r = report[l];
    if (!r) continue;
    html += '<tr><td style="padding:4px;color:' + (GROUP_COLORS[l]||'#fff') + ';font-weight:600;">' + l + '</td>';
    html += '<td style="text-align:center;">' + (r.precision*100).toFixed(0) + '%</td>';
    html += '<td style="text-align:center;">' + (r.recall*100).toFixed(0) + '%</td>';
    html += '<td style="text-align:center;">' + (r['f1-score']*100).toFixed(0) + '%</td></tr>';
  }
  html += '</table>';
  container.innerHTML = html;
}

function buildFeatureImportance(features) {
  const container = document.getElementById('feature-importance');
  const maxImp = features.length > 0 ? features[0].importance : 1;
  let html = '';
  for (const f of features) {
    const pct = (f.importance / maxImp * 100).toFixed(0);
    html += '<div class="conf-bar"><span class="label">F' + f.index + '</span>' +
      '<div class="bar-bg"><div class="bar-fill" style="width:' + pct + '%;background:var(--accent)"></div></div>' +
      '<span class="value">' + (f.importance*100).toFixed(1) + '%</span></div>';
  }
  container.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════
//  SAMPLE BROWSER
// ═══════════════════════════════════════════════════════════════

async function loadClassSamples() {
  const label = document.getElementById('browse-class').value;
  if (!label) { document.getElementById('sample-browser').textContent = 'Select a class'; return; }

  const resp = await fetch('/api/data/samples/' + label);
  const samples = await resp.json();
  const container = document.getElementById('sample-browser');

  if (samples.length === 0) {
    container.textContent = 'No samples for ' + label;
    return;
  }

  let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;">';
  for (const s of samples) {
    html += '<div style="background:var(--bg);padding:8px;border-radius:6px;font-size:11px;">' +
      '<div style="display:flex;justify-content:space-between;align-items:center;">' +
      '<span>#' + s.idx + ' (' + s.length + ' pts)</span>' +
      '<button class="btn danger" style="font-size:10px;padding:2px 6px;" onclick="deleteSample(' + s.idx + ')">Del</button></div>' +
      '<div style="margin-top:4px;">CH0: ' + s.energy_ch0.toFixed(1) + ' | CH1: ' + s.energy_ch1.toFixed(1) + '</div></div>';
  }
  html += '</div>';
  container.innerHTML = html;
}

async function deleteSample(idx) {
  if (!confirm('Delete sample #' + idx + '?')) return;
  await fetch('/api/data/delete', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({idx: idx})
  });
  loadClassSamples();
  loadProgress();
  showToast('Sample deleted', 'success');
}

// ═══════════════════════════════════════════════════════════════
//  INFERENCE
// ═══════════════════════════════════════════════════════════════

async function predictSingle() {
  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.textContent = 'Recording...';
  document.getElementById('predict-result').innerHTML = '<div style="color:var(--text2)">Recording 1s...</div>';

  try {
    const resp = await fetch('/api/predict', {method:'POST'});
    const data = await resp.json();

    if (data.error) {
      document.getElementById('predict-result').innerHTML = '<div style="color:var(--red)">' + data.error + '</div>';
      return;
    }

    const color = GROUP_COLORS[data.prediction] || '#fff';
    document.getElementById('predict-result').innerHTML =
      '<div style="font-size:36px;font-weight:700;color:' + color + ';text-align:center;">' +
      data.prediction + '</div>' +
      '<div style="text-align:center;color:var(--text2);font-size:14px;">' +
      (data.confidence*100).toFixed(0) + '% confidence</div>';

    // Detail
    let detailHtml = '<div style="margin-bottom:12px;">';
    for (const t of data.top3) {
      const pct = (t.prob*100).toFixed(0);
      const c = GROUP_COLORS[t.label] || '#666';
      detailHtml += '<div class="conf-bar"><span class="label" style="color:' + c + '">' + t.label + '</span>' +
        '<div class="bar-bg"><div class="bar-fill" style="width:' + pct + '%;background:' + c + '"></div></div>' +
        '<span class="value">' + pct + '%</span></div>';
    }
    detailHtml += '</div>';
    detailHtml += '<div style="font-size:11px;color:var(--text2);">' +
      'Crop: ' + data.raw_length + ' -> ' + data.crop_length + ' | ' +
      'Energy CH0: ' + data.energy_ch0.toFixed(0) + ' CH1: ' + data.energy_ch1.toFixed(0) + '</div>';

    // Mini waveform
    if (data.ch0 && data.ch0.length > 0) {
      detailHtml += '<div class="chart-container" style="height:120px;margin-top:8px;"><canvas id="chart-predict"></canvas></div>';
    }

    document.getElementById('predict-detail').innerHTML = detailHtml;

    if (data.ch0 && data.ch0.length > 0) {
      const ctx = document.getElementById('chart-predict').getContext('2d');
      new Chart(ctx, {
        type:'line',
        data: {
          labels: data.ch0.map((_,i)=>i),
          datasets: [
            { label:'CH0', data:data.ch0, borderColor:'#06b6d4', borderWidth:1, pointRadius:0 },
            { label:'CH1', data:data.ch1, borderColor:'#ff6b6b', borderWidth:1, pointRadius:0 },
          ]
        },
        options: {
          responsive:true, maintainAspectRatio:false, animation:false,
          scales:{ x:{display:false}, y:{grid:{color:'#30363d'},ticks:{color:'#666',font:{size:9}}} },
          plugins:{legend:{display:false}}
        }
      });
    }
  } catch(err) {
    document.getElementById('predict-result').innerHTML = '<div style="color:var(--red)">Error: ' + err.message + '</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Record & Classify';
  }
}

// ═══════════════════════════════════════════════════════════════
//  SPELL ENGINE
// ═══════════════════════════════════════════════════════════════

function updateSpellUI() {
  const groupsEl = document.getElementById('spell-groups');
  if (spellGroups.length === 0) {
    groupsEl.innerHTML = '<span style="color:var(--text2);font-size:13px;">No groups yet</span>';
  } else {
    groupsEl.innerHTML = spellGroups.map(g =>
      '<span class="group-tag ' + g + '">' + g + '</span>'
    ).join('<span style="color:var(--text2);">&rarr;</span>');
  }
  document.getElementById('spell-sentence').textContent =
    spellSentence.length > 0 ? spellSentence.join(' ') : '--';

  // Lookup candidates
  if (spellGroups.length > 0) {
    fetch('/api/spell/lookup', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({groups: spellGroups, context: spellSentence.join(' ')})
    }).then(r=>r.json()).then(data => {
      document.getElementById('spell-candidates').textContent =
        data.prefix_count + ' possible (' + data.total_candidates + ' at this length)';
    });
  } else {
    document.getElementById('spell-candidates').textContent = '--';
  }
}

async function spellRecord() {
  const btn = document.querySelector('[onclick="spellRecord()"]');
  btn.disabled = true;
  btn.textContent = 'Recording...';

  try {
    const resp = await fetch('/api/predict', {method:'POST'});
    const data = await resp.json();
    if (data.error) { showToast(data.error, 'error'); return; }

    if (data.prediction === 'REST') {
      showToast('REST detected - no gesture', 'error');
      return;
    }

    spellGroups.push(data.prediction);
    updateSpellUI();
    showToast(data.prediction + ' (' + (data.confidence*100).toFixed(0) + '%)', 'success');
  } catch(err) {
    showToast('Error: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Record Gesture';
  }
}

function spellAddGroup(group) {
  spellGroups.push(group);
  updateSpellUI();
}

async function spellDone() {
  if (spellGroups.length === 0) { showToast('No groups to predict', 'error'); return; }

  document.getElementById('spell-result').innerHTML = '<div style="color:var(--text2)">Looking up...</div>';

  const resp = await fetch('/api/spell/lookup', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({groups: spellGroups, context: spellSentence.join(' ')})
  });
  const data = await resp.json();

  let html = '';
  if (data.best) {
    html += '<div style="font-size:24px;font-weight:700;color:var(--green);margin-bottom:8px;">Predicted: "' + data.best + '"</div>';
    html += '<div style="display:flex;gap:8px;margin-bottom:8px;">' +
      '<button class="btn success" onclick="spellAcceptWord(\'' + data.best + '\')">Accept</button>' +
      '<button class="btn" onclick="spellClear()">Retry</button></div>';
  }
  if (data.exact.length > 0) {
    html += '<div style="font-size:12px;color:var(--text2);margin-bottom:4px;">Exact matches: ' + data.exact.slice(0,15).join(', ') + '</div>';
  }
  if (data.all_candidates.length > data.exact.length) {
    html += '<div style="font-size:12px;color:var(--text2);">Fuzzy matches: ' + data.all_candidates.slice(0,15).join(', ') + '</div>';
  }

  document.getElementById('spell-result').innerHTML = html;
}

function spellAcceptWord(word) {
  spellSentence.push(word);
  spellGroups = [];
  document.getElementById('spell-result').innerHTML = '';
  updateSpellUI();
  showToast('Added: "' + word + '"', 'success');
}

function spellUndo() {
  if (spellGroups.length > 0) {
    spellGroups.pop();
    updateSpellUI();
  }
}

function spellClear() {
  spellGroups = [];
  document.getElementById('spell-result').innerHTML = '';
  updateSpellUI();
}

function spellNewSentence() {
  spellSentence = [];
  spellGroups = [];
  document.getElementById('spell-result').innerHTML = '';
  updateSpellUI();
}

// ═══════════════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════════════

function showToast(msg, type) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast ' + (type||'');
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
}

// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════

async function init() {
  initMonitorChart();
  initWebSocket();
  buildLetterGrid();
  await loadProgress();

  // Check model status and load stats
  try {
    const resp = await fetch('/api/model/info');
    const info = await resp.json();
    if (info.loaded) {
      document.getElementById('dot-model').className = 'dot green';
      document.getElementById('model-acc-header').textContent = (info.accuracy*100).toFixed(1) + '%';

      // Auto-load full training stats
      const statsResp = await fetch('/api/model/stats');
      const data = await statsResp.json();
      if (!data.error) {
        document.getElementById('train-results').style.display = 'block';
        document.getElementById('train-accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
        document.getElementById('train-samples').textContent = data.n_samples;
        document.getElementById('train-classes').textContent = data.labels.length;

        const accEl = document.getElementById('train-accuracy');
        accEl.style.color = data.accuracy > 0.8 ? 'var(--green)' : data.accuracy > 0.6 ? 'var(--yellow)' : 'var(--red)';

        buildConfusionMatrix(data.confusion_matrix, data.labels);
        buildClassMetrics(data.report, data.labels);
        buildFeatureImportance(data.top_features);

        // Populate sample browser dropdown
        const sel = document.getElementById('browse-class');
        sel.innerHTML = '<option value="">Select class...</option>';
        for (const l of data.labels) {
          sel.innerHTML += '<option value="' + l + '">' + l + ' (' + (data.counts[l]||0) + ')</option>';
        }
      }
    } else {
      document.getElementById('dot-model').className = 'dot yellow';
    }
  } catch(e) {}
}

window.addEventListener('load', init);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def detect_port():
    for p in serial.tools.list_ports.comports():
        if any(tag in p.device for tag in ("usbmodem", "usbserial", "ACM")):
            return p.device
    ports = list(serial.tools.list_ports.comports())
    return ports[0].device if ports else None


def main():
    parser = argparse.ArgumentParser(description="SilentPilot EMG Calibration App")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--user", default="aarush", help="User ID")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--no-serial", action="store_true", help="Run without serial/Arduino connection")
    args = parser.parse_args()

    global stream, datastore, model_mgr, spell_engine

    print(f"\n  SilentPilot EMG Calibration App")
    print(f"  ================================")
    print(f"  User: {args.user}")

    # Serial connection (optional)
    if args.no_serial:
        print("  Serial: SKIPPED (data-only mode)")
        stream = None
    else:
        port = args.port or detect_port()
        if port:
            print(f"  Serial port: {port}")
            try:
                print(f"  Connecting to Arduino...", end=" ", flush=True)
                stream = EMGStream(port)
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")
                stream = None
        else:
            print("  No Arduino found -- running in data-only mode")
            stream = None

    print(f"  Loading data store...", end=" ", flush=True)
    datastore = DataStore(args.user)
    print(f"OK ({len(datastore.segments)} samples)")

    print(f"  Loading model manager...", end=" ", flush=True)
    model_mgr = ModelManager(args.user)
    if model_mgr.is_loaded:
        print(f"OK ({model_mgr.accuracy:.1%} accuracy)")
    else:
        print("No model yet")

    print(f"  Loading spell engine...", end=" ", flush=True)
    spell_engine = SpellEngine()
    print(f"OK ({spell_engine.trie.word_count:,} words)")

    url = f"http://localhost:{args.web_port}"
    print(f"\n  Server starting at {url}")
    print(f"  Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.web_port, log_level="warning")


if __name__ == "__main__":
    main()
