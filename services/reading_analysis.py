"""
Reading programme — read-aloud analysis service.

Every score is computed here in Python from Whisper word timestamps +
Parselmouth F0 — the LLM only writes the coach prose (and may reword the
flow chip). Pipeline per the spec:

  normalise + Whisper (controller, via services/audio.py) → tokenise/align
  (difflib) → hesitance → pace → flow/contour → overall → Claude feedback
  with a deterministic templated fallback (the endpoint never fails on the LLM).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import parselmouth

from config import CLAUDE_MODEL, anthropic_client
from models.reading import HesitationEvent, WordFeedbackEntry
from prompts.reading_feedback import READING_FEEDBACK_PROMPT
from services.listening import _SG_PARTICLE_RE
from utils.text import clamp, strip_fences

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

# Guide-pace defaults when koras-api sends none: comfortable oral-reading
# rates by level — ~90 wpm for grades 4–6, ~110 for grades 7–9, ~130 for
# grade 10+ and adult/professional readers.
_GUIDE_WPM_LOWER = 90
_GUIDE_WPM_MID = 110
_GUIDE_WPM_UPPER = 130

_GRADE_RE = re.compile(r"grade[\s_-]?(\d{1,2})")

# Punctuation that marks a phrase boundary; a pause there is phrasing, not
# hesitation. Terminal marks additionally earn the phrasing bonus.
_PUNCT_CHARS = ",;:.!?—"
_TERMINAL_PUNCT = ".!?"

# Hesitation-vowel fillers. "ah" is included deliberately: the listening
# service treats it as a Singapore English discourse particle, but in a
# read-aloud it is almost always a hesitation vowel — which is exactly why
# it is stripped from the particle pattern below.
_FILLER_TOKENS = {"um", "uh", "er", "hmm", "ah"}

# Reading-specific copy of the listening service's SG-particle pattern with
# the standalone "ah" alternative removed (the "ah" inside "haiyah" is
# unaffected). Built from the pattern string — the original is not mutated.
_READING_PARTICLE_RE = re.compile(
    _SG_PARTICLE_RE.pattern.replace("|ah|", "|"), re.IGNORECASE
)

# CV-of-F0 → flow score: monotone below 0.08, peak plateau across 0.15–0.30,
# penalised again at the erratic extreme.
_FLOW_CV_X = [0.0, 0.08, 0.15, 0.30, 0.45, 0.60]
_FLOW_CV_Y = [35.0, 70.0, 92.0, 92.0, 62.0, 40.0]

_CONTOUR_POINTS = 40

_CONTRACTIONS: dict[str, tuple[str, ...]] = {
    "i'm": ("i", "am"), "you're": ("you", "are"), "we're": ("we", "are"),
    "they're": ("they", "are"), "it's": ("it", "is"), "he's": ("he", "is"),
    "she's": ("she", "is"), "that's": ("that", "is"), "there's": ("there", "is"),
    "here's": ("here", "is"), "what's": ("what", "is"), "who's": ("who", "is"),
    "where's": ("where", "is"), "let's": ("let", "us"),
    "can't": ("cannot",), "won't": ("will", "not"), "shan't": ("shall", "not"),
    "ain't": ("is", "not"),
}

_EDGE_PUNCT_RE = re.compile(r"^[^\w']+|[^\w']+$")


# ─────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────

def _norm_token(raw: str) -> str:
    token = raw.lower().replace("’", "'")
    token = _EDGE_PUNCT_RE.sub("", token)
    return token.strip("'")


def _expand(token: str) -> tuple[str, ...]:
    """Expand common contractions — applied identically to both sides."""
    if token in _CONTRACTIONS:
        return _CONTRACTIONS[token]
    for suffix, replacement in (
        ("n't", "not"), ("'ll", "will"), ("'re", "are"),
        ("'ve", "have"), ("'d", "would"),
    ):
        if token.endswith(suffix) and len(token) > len(suffix):
            return (token[: -len(suffix)], replacement)
    return (token,)


def _tokenize_passage(text: str) -> list[tuple[str, str]]:
    """[(token, punct_after)] — punct_after is the punctuation character
    immediately following the token ('' when none). Drives the
    phrasing-vs-hesitation classification."""
    entries: list[tuple[str, str]] = []
    for raw in text.replace("—", " — ").split():
        tail = raw.rstrip("\"'”’)]")
        punct = tail[-1] if tail and tail[-1] in _PUNCT_CHARS else ""
        norm = _norm_token(raw)
        if not norm:
            # standalone punctuation (e.g. a spaced em-dash) attaches to the
            # previous word
            if punct and entries:
                word, prev = entries[-1]
                entries[-1] = (word, prev or punct)
            continue
        parts = _expand(norm)
        for i, part in enumerate(parts):
            entries.append((part, punct if i == len(parts) - 1 else ""))
    return entries


def _tokenize_transcript(words: list[dict]) -> tuple[list[str], list[int]]:
    """Returns (tokens, whisper word index per token). Expanded contraction
    tokens share their source word's index (and therefore its timestamps)."""
    tokens: list[str] = []
    widx: list[int] = []
    for j, w in enumerate(words):
        norm = _norm_token(str(w.get("word") or ""))
        if not norm:
            continue
        for part in _expand(norm):
            tokens.append(part)
            widx.append(j)
    return tokens, widx


def default_guide_wpm(level_hint: str | None) -> int:
    match = _GRADE_RE.search((level_hint or "").lower())
    if match:
        grade = int(match.group(1))
        if grade <= 6:
            return _GUIDE_WPM_LOWER
        if grade <= 9:
            return _GUIDE_WPM_MID
    return _GUIDE_WPM_UPPER


# ─────────────────────────────────────────────────────────────────────────
# F0 / contour helpers
# ─────────────────────────────────────────────────────────────────────────

def _voiced_f0(wav_bytes: bytes) -> np.ndarray:
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name
        snd = parselmouth.Sound(wav_path)
        values = snd.to_pitch().selected_array["frequency"]
        return values[values > 0]
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def _flow_from_cv(cv: float) -> int:
    return clamp(round(float(np.interp(cv, _FLOW_CV_X, _FLOW_CV_Y))))


def _resample_z(arr: np.ndarray) -> Optional[np.ndarray]:
    resampled = np.interp(
        np.linspace(0.0, 1.0, num=_CONTOUR_POINTS),
        np.linspace(0.0, 1.0, num=len(arr)),
        arr,
    )
    std = float(resampled.std())
    if std <= 1e-9:
        return None
    return (resampled - resampled.mean()) / std


def _contour_similarity(user_f0: np.ndarray, target: list[float]) -> Optional[int]:
    tgt = np.asarray([float(v) for v in target], dtype=float)
    if len(user_f0) < 4 or len(tgt) < 4:
        return None
    user_z = _resample_z(user_f0.astype(float))
    tgt_z = _resample_z(tgt)
    if user_z is None or tgt_z is None:
        return None
    cos = float(np.dot(user_z, tgt_z) / (np.linalg.norm(user_z) * np.linalg.norm(tgt_z)))
    return clamp(round(100 * max(0.0, cos)))


# ─────────────────────────────────────────────────────────────────────────
# Core metric computation (CPU-bound; run via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────

def compute_reading_metrics(
    passage_text: str,
    transcript_data: dict,
    wav_bytes: bytes,
    attempt_type: str,
    guide_wpm: int | None,
    level_hint: str | None,
    target_contour: list[float] | None,
) -> dict:
    whisper_words = transcript_data.get("words") or []
    p_entries = _tokenize_passage(passage_text)
    p_tokens = [token for token, _ in p_entries]
    t_tokens, t_widx = _tokenize_transcript(whisper_words)

    # ── 4. Align (difflib.SequenceMatcher over normalised tokens) ────────
    matcher = difflib.SequenceMatcher(None, p_tokens, t_tokens, autojunk=False)
    status: list[str] = ["skipped"] * len(p_tokens)
    heard: list[Optional[str]] = [None] * len(p_tokens)
    p2t: list[Optional[int]] = [None] * len(p_tokens)
    inserted_before: dict[int, list[int]] = defaultdict(list)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                status[i1 + k] = "correct"
                p2t[i1 + k] = j1 + k
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for k in range(paired):
                status[i1 + k] = "substituted"
                heard[i1 + k] = t_tokens[j1 + k]
                p2t[i1 + k] = j1 + k
            for j in range(j1 + paired, j2):
                inserted_before[i2].append(j)
        elif tag == "insert":
            for j in range(j1, j2):
                inserted_before[i1].append(j)

    correct_count = sum(1 for s in status if s == "correct")
    match_pct = clamp(round(100 * correct_count / len(p_tokens))) if p_tokens else 0

    aligned_pairs = sorted(
        (p2t[i], i) for i in range(len(p_tokens)) if p2t[i] is not None
    )
    aligned_t = [tj for tj, _ in aligned_pairs]
    aligned_t_set = set(aligned_t)

    def _nearest_aligned_before(j: int) -> int:
        k = bisect_right(aligned_t, j) - 1
        return aligned_pairs[k][1] if k >= 0 else 0

    def _gap_before_ms(tj: int) -> Optional[int]:
        wj = t_widx[tj]
        if tj > 0 and t_widx[tj - 1] == wj:
            return 0  # later token of a contraction expansion
        if wj <= 0:
            return None
        prev_w, cur_w = whisper_words[wj - 1], whisper_words[wj]
        return max(0, int(round((float(cur_w["start"]) - float(prev_w["end"])) * 1000)))

    word_feedback = [
        WordFeedbackEntry(word=p_tokens[i], index=i, status=status[i], heard=heard[i])
        for i in range(len(p_tokens))
    ]

    # ── 5. Hesitance — start at 100, penalties/bonuses per event ─────────
    penalty = 0.0
    bonus = 0.0
    events: list[HesitationEvent] = []

    prev_pi: Optional[int] = None
    for tj, pi in aligned_pairs:
        gap_ms = _gap_before_ms(tj)
        if gap_ms is not None:
            word_feedback[pi].pause_before_ms = gap_ms
        if prev_pi is not None and gap_ms is not None:
            punct = p_entries[prev_pi][1]
            if gap_ms > 350 and not punct:
                penalty += min(12.0, gap_ms / 150.0)
                events.append(HesitationEvent(
                    after_word_index=prev_pi, gap_ms=gap_ms, kind="mid_clause_pause",
                ))
                word_feedback[pi].hesitation = True
            elif punct in _TERMINAL_PUNCT and 250 <= gap_ms <= 900:
                bonus = min(6.0, bonus + 1.0)  # phrasing bonus, capped at +6
        prev_pi = pi

    # restarts: insert-run immediately followed by a match of the same word
    restart_js: set[int] = set()
    for i, js in sorted(inserted_before.items()):
        if not js:
            continue
        last = js[-1]
        matched_tj = p2t[i] if i < len(p_tokens) else None
        if matched_tj is not None and t_tokens[last] == t_tokens[matched_tj]:
            penalty += 6.0
            events.append(HesitationEvent(
                after_word_index=max(0, i - 1),
                gap_ms=_gap_before_ms(last) or 0,
                kind="restart",
            ))
            word_feedback[i].hesitation = True
            restart_js.add(last)

    # repetitions: identical consecutive transcript tokens (skip restart pairs
    # and words the passage itself repeats)
    for j in range(1, len(t_tokens)):
        if t_tokens[j] != t_tokens[j - 1] or t_widx[j] == t_widx[j - 1]:
            continue
        if j in restart_js or (j - 1) in restart_js:
            continue
        if j in aligned_t_set and (j - 1) in aligned_t_set:
            continue
        penalty += 6.0
        events.append(HesitationEvent(
            after_word_index=_nearest_aligned_before(j),
            gap_ms=_gap_before_ms(j) or 0,
            kind="repetition",
        ))

    # fillers among inserted tokens — SG discourse particles are exempt
    for _, js in sorted(inserted_before.items()):
        for j in js:
            if j in restart_js:
                continue
            token = t_tokens[j]
            if token in _FILLER_TOKENS and not _READING_PARTICLE_RE.fullmatch(token):
                penalty += 4.0
                events.append(HesitationEvent(
                    after_word_index=_nearest_aligned_before(j),
                    gap_ms=_gap_before_ms(j) or 0,
                    kind="filler",
                ))

    hesitance_score = clamp(round(100.0 - penalty + bonus))
    events.sort(key=lambda e: e.after_word_index)

    # non-restart insertions attach to the adjacent passage token for display;
    # match_pct above stays computed from the raw alignment
    for i, js in inserted_before.items():
        if not any(j not in restart_js for j in js):
            continue
        anchor = i if i < len(p_tokens) else len(p_tokens) - 1
        if anchor >= 0 and word_feedback[anchor].status == "correct":
            word_feedback[anchor].status = "inserted-neighbour"

    # ── 6. Pace — WPM over speech trimmed to first/last word timestamps ──
    speaking_s = 0.0
    if whisper_words:
        speaking_s = max(
            0.0, float(whisper_words[-1]["end"]) - float(whisper_words[0]["start"])
        )
    duration_s = float(transcript_data.get("duration") or 0.0) or speaking_s

    aligned_count = len(aligned_pairs)
    wpm = round(aligned_count * 60.0 / speaking_s) if speaking_s > 0 else 0
    guide = int(guide_wpm) if guide_wpm else default_guide_wpm(level_hint)
    pace_delta_pct = round(100.0 * (wpm - guide) / guide) if guide > 0 else 0
    abs_delta = abs(pace_delta_pct)
    # 100 within ±5%, linear falloff to 40 at ±40%, same slope beyond, clamped
    pace_score = 100 if abs_delta <= 5 else clamp(round(100.0 - (abs_delta - 5) * (60.0 / 35.0)))

    # ── 7. Flow / prosody — Parselmouth F0 CV, plus echo contour blend ────
    voiced = _voiced_f0(wav_bytes)
    mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    cv = float(np.std(voiced) / mean_f0) if mean_f0 > 0 else 0.0
    flow_score = _flow_from_cv(cv)

    contour_similarity: Optional[int] = None
    if attempt_type == "echo" and target_contour:
        contour_similarity = _contour_similarity(voiced, target_contour)
        if contour_similarity is not None:
            flow_score = clamp(round(0.6 * flow_score + 0.4 * contour_similarity))

    # ── 8. Overall — fixed weights, clamped ──────────────────────────────
    overall_score = clamp(round(
        0.35 * match_pct + 0.30 * hesitance_score
        + 0.15 * pace_score + 0.20 * flow_score
    ))

    return {
        "transcript": (transcript_data.get("text") or "").strip(),
        "duration_seconds": round(duration_s, 1),
        "words_per_minute": int(wpm),
        "match_pct": match_pct,
        "hesitance_score": hesitance_score,
        "pace_score": pace_score,
        "pace_delta_pct": int(pace_delta_pct),
        "flow_score": flow_score,
        "contour_similarity": contour_similarity,
        "overall_score": overall_score,
        "word_feedback": word_feedback,
        "hesitation_events": events,
    }


# ─────────────────────────────────────────────────────────────────────────
# Coach feedback (Claude, optional path — deterministic fallback)
# ─────────────────────────────────────────────────────────────────────────

def _flow_band(flow_score: int) -> str:
    if flow_score >= 75:
        return "Smooth FLOW"
    if flow_score >= 55:
        return "Steady FLOW"
    if flow_score >= 35:
        return "Uneven FLOW"
    return "Choppy FLOW"


def default_chips(metrics: dict) -> list[str]:
    return [
        f"{metrics['match_pct']}% MATCH",
        f"{metrics['pace_delta_pct']:+d}% PACE",
        _flow_band(metrics["flow_score"]),
    ]


def reading_feedback_fallback(metrics: dict) -> str:
    """Deterministic templated feedback — the endpoint must never fail
    because of the LLM."""
    match = metrics["match_pct"]
    hesitance = metrics["hesitance_score"]
    delta = metrics["pace_delta_pct"]

    sentences: list[str] = []
    if match >= 90:
        sentences.append(f"Lovely reading — {match}% of the passage came through word for word.")
    elif match >= 70:
        sentences.append(f"Good effort — you matched {match}% of the passage, and the trickier words will settle with practice.")
    else:
        sentences.append(f"You matched {match}% of the passage this time — take it slowly and give it another go.")

    if hesitance >= 80:
        sentences.append("Your reading flowed smoothly, with barely a stumble.")
    elif hesitance >= 55:
        sentences.append("A few pauses crept in mid-sentence; let the punctuation do the breathing for you.")
    else:
        sentences.append("There were quite a few hesitations — try reading the passage silently once before you record.")

    if abs(delta) <= 5:
        sentences.append("Your pace was spot on the guide.")
    elif delta > 0:
        sentences.append(f"You ran about {abs(delta)}% quicker than the guide — ease off a touch next time.")
    else:
        sentences.append(f"You ran about {abs(delta)}% slower than the guide — trust yourself and press on a little.")

    return " ".join(sentences)


async def generate_reading_feedback(
    metrics: dict,
    attempt_type: str,
    level_hint: str | None,
) -> tuple[str, list[str]]:
    """Returns (coach_feedback, chips). Falls back to the deterministic
    template + chips on any Claude failure."""
    chips = default_chips(metrics)
    problem_words = [
        wf.word for wf in metrics["word_feedback"]
        if wf.status in ("substituted", "skipped")
    ][:6]
    summary = {
        "match_pct": metrics["match_pct"],
        "hesitance_score": metrics["hesitance_score"],
        "pace_score": metrics["pace_score"],
        "pace_delta_pct": metrics["pace_delta_pct"],
        "flow_score": metrics["flow_score"],
        "contour_similarity": metrics["contour_similarity"],
        "overall_score": metrics["overall_score"],
        "words_per_minute": metrics["words_per_minute"],
        "problem_words": problem_words,
        "hesitation_kinds": dict(Counter(e.kind for e in metrics["hesitation_events"])),
    }
    prompt = READING_FEEDBACK_PROMPT.format(
        attempt_type=attempt_type,
        level_hint=level_hint or "general adult",
        metrics_json=json.dumps(summary),
        chips_json=json.dumps(chips),
    )

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))
        feedback = str(data.get("coach_feedback") or "").strip()
        if not feedback:
            return reading_feedback_fallback(metrics), chips
        raw_chips = data.get("chips")
        out_chips = (
            [str(c).strip() for c in raw_chips if isinstance(c, (str, int, float)) and str(c).strip()][:3]
            if isinstance(raw_chips, list) else []
        )
        return feedback, (out_chips if len(out_chips) == 3 else chips)
    except Exception:
        return reading_feedback_fallback(metrics), chips
