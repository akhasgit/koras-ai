"""
Voice Refinement — analysis + plan generation service.

Non-goals here:
  - do NOT re-implement /analyze. Reuse `extract_features`, `transcribe`,
    `analyze_transcript`, `generate_coach_feedback` from the shared helpers.

Goals:
  - Compute a natural-pitch band from a wav (IQR of voiced frames, clamped
    to [75, 500] Hz — parselmouth's default speaking-range bounds).
  - Compute three band-energy ratios (low/mid/high) from librosa STFT.
  - Classify the slider intent against the band using the honesty-contract rule
    (see `classify_target_intent`).
  - Generate the 14-day plan via Claude, validate the JSON strictly, and fall
    back to a deterministic minimal plan on parse failure.
"""
from __future__ import annotations

import io
import json
import math
import os
import tempfile
from typing import Any, Iterable, List, Optional

import librosa
import numpy as np
import parselmouth
from pydantic import ValidationError

from config import CLAUDE_MODEL, anthropic_client
from models.voice_refinement import (
    BandFeatures,
    NaturalRangeHz,
    TargetClassification,
    TargetIntent,
    VfCatalogItem,
    VoiceRefinementPlan,
    VoiceRefinementPlanActivity,
    VoiceRefinementPlanDay,
)
from prompts.voice_refinement import VOICE_REFINEMENT_PLAN_PROMPT
from utils.text import strip_fences

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

# parselmouth defaults for adult speaking voice; anything outside is
# implausible and almost certainly a tracker error.
ABS_FLOOR_HZ = 75.0
ABS_CEILING_HZ = 500.0

# Band edges for the low/mid/high energy ratio.
BAND_LOW_HZ = (80.0, 500.0)
BAND_MID_HZ = (500.0, 2500.0)
BAND_HIGH_HZ = (2500.0, 8000.0)

# Gender-affirming / out-of-scope threshold on the slider.
OUT_OF_SCOPE_SEMITONES = 5.0


# ─────────────────────────────────────────────────────────────────────────
# Natural range + band energies
# ─────────────────────────────────────────────────────────────────────────

def compute_natural_range(wav_bytes: bytes) -> NaturalRangeHz:
    """Estimate the user's habitual pitch band using the IQR of voiced frames.

    Clamps to [ABS_FLOOR_HZ, ABS_CEILING_HZ] to protect against tracker errors.
    Returns a `NaturalRangeHz` with mean, Q1, Q3, and clamped low/high edges.
    """
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch()
        values = pitch.selected_array["frequency"]
        voiced = values[values > 0]

        if len(voiced) == 0:
            return NaturalRangeHz(
                mean=0.0, q1=0.0, q3=0.0,
                low=ABS_FLOOR_HZ, high=ABS_CEILING_HZ,
            )

        mean = float(np.mean(voiced))
        q1 = float(np.percentile(voiced, 25))
        q3 = float(np.percentile(voiced, 75))

        low = max(ABS_FLOOR_HZ, q1)
        high = min(ABS_CEILING_HZ, q3)
        # If IQR collapses to a single value (very monotone rep), open a
        # small band around mean so classification still works.
        if high <= low:
            low = max(ABS_FLOOR_HZ, mean - 10.0)
            high = min(ABS_CEILING_HZ, mean + 10.0)

        return NaturalRangeHz(mean=mean, q1=q1, q3=q3, low=low, high=high)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def compute_band_energies(wav_bytes: bytes) -> BandFeatures:
    """Three-band energy ratios via librosa STFT.

    Normalised so `low + mid + high == 1.0`. Useful for grounded copy in the
    plan prompt (e.g. "voice already balanced toward warmth").
    """
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if len(y) == 0:
            return BandFeatures(low_band_energy=0.0, mid_band_energy=0.0, high_band_energy=0.0)

        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        def _band_sum(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(stft[mask]))

        low = _band_sum(*BAND_LOW_HZ)
        mid = _band_sum(*BAND_MID_HZ)
        high = _band_sum(*BAND_HIGH_HZ)
        total = low + mid + high

        if total <= 0:
            return BandFeatures(low_band_energy=0.0, mid_band_energy=0.0, high_band_energy=0.0)

        return BandFeatures(
            low_band_energy=round(low / total, 4),
            mid_band_energy=round(mid / total, 4),
            high_band_energy=round(high / total, 4),
        )
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────

def _hz_from_semitones(base_hz: float, semitones: float) -> float:
    if base_hz <= 0:
        return 0.0
    return base_hz * (2.0 ** (semitones / 12.0))


def classify_target_intent(
    intent: TargetIntent,
    natural: NaturalRangeHz,
) -> tuple[TargetClassification, List[str], Optional[str]]:
    """Classify slider intent → (classification, recommended_focus, clinical_note).

    Rules (from the spec):
      - Target Hz inside [low, high] → "trainable"
      - Outside band AND |semitones| < 5 (moderate) → "perceptual_proxy"
      - |semitones| >= 5 AND outside band AND consistent direction → "out_of_scope"
        + clinical_note pointing to a speech-language pathologist
    """
    semis = float(intent.pitch_semitones or 0.0)
    mean = natural.mean or 0.0
    target_hz = _hz_from_semitones(mean, semis)

    notes: List[str] = []
    focus: List[str] = []
    clinical: Optional[str] = None

    inside_band = (natural.low <= target_hz <= natural.high) if mean > 0 else True
    magnitude = abs(semis)
    consistent_direction = magnitude > 0  # a single-slider value is inherently one-directional

    if magnitude < 0.5:
        pitch_class = "trainable"
        notes.append("Pitch target is close to your natural range — small habitual shifts are trainable.")
    elif inside_band:
        pitch_class = "trainable"
        notes.append(
            f"Pitch target ({round(target_hz)}Hz) lands inside your natural range "
            f"({round(natural.low)}–{round(natural.high)}Hz)."
        )
    elif magnitude >= OUT_OF_SCOPE_SEMITONES and not inside_band and consistent_direction:
        pitch_class = "out_of_scope"
        direction = "lower" if semis < 0 else "higher"
        notes.append(
            f"A {round(magnitude, 1)}-semitone {direction} target sits well outside your "
            f"natural range ({round(natural.low)}–{round(natural.high)}Hz). Fundamental pitch "
            "beyond that band is anatomical, not trainable, so the plan focuses on the "
            "perceptual proxies that make a voice sound deeper/brighter (resonance, pace, prosody)."
        )
        clinical = (
            "If you're aiming for a substantially different-sounding voice — for example, "
            "as part of gender-affirming voice work — a qualified speech-language pathologist "
            "is the right person to guide that. Koras Voice Refinement can complement clinical "
            "training but is not a substitute for it."
        )
    else:
        pitch_class = "perceptual_proxy"
        direction = "lower / warmer" if semis < 0 else "higher / brighter"
        notes.append(
            f"Pitch target is outside your natural range in the {direction} direction, "
            "but the shift is moderate. The plan will target the perceptual proxies "
            "(resonance, pace, prosody) rather than promising a fundamental-pitch change."
        )

    # Focus recommendations
    if intent.speed_ratio < 0.95:
        focus.append("slower pace + intentional pauses")
    elif intent.speed_ratio > 1.05:
        focus.append("energy + articulation for a livelier pace")
    if intent.resonance < -0.2:
        focus.append("chest resonance / warmth")
    elif intent.resonance > 0.2:
        focus.append("forward resonance / brightness")
    if intent.brightness < -0.2:
        focus.append("soften brittle high frequencies")
    elif intent.brightness > 0.2:
        focus.append("presence and clarity in the top end")
    if pitch_class == "trainable" and abs(semis) >= 0.5:
        delta_hz = int(round(target_hz - mean))
        direction = "lower" if delta_hz < 0 else "raise"
        focus.append(f"{direction} habitual pitch by ~{abs(delta_hz)}Hz within natural range")

    if not focus:
        focus.append("prosody + pause control")

    return TargetClassification(pitch=pitch_class, notes=notes), focus, clinical


# ─────────────────────────────────────────────────────────────────────────
# Plan generation
# ─────────────────────────────────────────────────────────────────────────

def _minimal_fallback_plan(
    classification: TargetClassification,
    focus: List[str],
    clinical_note: Optional[str],
    baseline_prompt_hint: str,
) -> VoiceRefinementPlan:
    """Deterministic minimal 14-day plan used when the LLM output can't be parsed twice.

    Not marketing-quality copy — the point is to never 500 and to keep the
    honesty contract intact even in degraded mode.
    """
    def _day(day_num: int, title: str, extras: Optional[List[VoiceRefinementPlanActivity]] = None) -> VoiceRefinementPlanDay:
        base = [
            VoiceRefinementPlanActivity(
                id=f"vr-d{day_num}-breath-reset",
                day=day_num, type="drill",
                title="Breath Reset",
                description="Ground your breath so the rest of the day's practice has support.",
                durationMinutes="3–5 min",
                purpose="Steady diaphragmatic breathing is the foundation of every trainable dimension.",
                instructions=[],
                referencedActivityId="vf-d1-breath-reset",
            ),
            VoiceRefinementPlanActivity(
                id=f"vr-d{day_num}-humming",
                day=day_num, type="drill",
                title="Resonance Hum",
                description="Wake up the resonators before speaking.",
                durationMinutes="3–5 min",
                purpose="Humming activates the chest and face resonance the plan builds on.",
                instructions=[],
                referencedActivityId="vf-d2-humming-warmup",
            ),
            VoiceRefinementPlanActivity(
                id=f"vr-d{day_num}-focus",
                day=day_num, type="training",
                title="Focus of the day",
                description=", ".join(focus[:2]) or "Prosody + pause control",
                durationMinutes="4–6 min",
                purpose="Direct practice on your top personalised focus for today.",
                instructions=[
                    "Read a short paragraph aloud twice.",
                    "The first read: at your usual pace.",
                    "The second read: apply today's focus deliberately.",
                    "Notice which felt more natural.",
                ],
            ),
        ]
        base.extend(extras or [])
        return VoiceRefinementPlanDay(day=day_num, title=title, activities=base)

    checkpoint_d7 = VoiceRefinementPlanActivity(
        id="vr-d7-checkpoint", day=7, type="recording",
        title="Day 7 checkpoint",
        description="Rerecord the baseline prompt so we can hear the shift.",
        durationMinutes="1–2 min",
        purpose="A comparable rep that surfaces the trainable-metric delta since Day 1.",
        instructions=[
            "Read the same prompt as your baseline: " + baseline_prompt_hint,
            "Use what you've practised — breath, pace, pauses.",
            "Stop when you're done.",
        ],
        isCheckpoint=True, targetSeconds=45,
    )
    checkpoint_d14 = VoiceRefinementPlanActivity(
        id="vr-d14-checkpoint", day=14, type="recording",
        title="Day 14 checkpoint",
        description="Final comparable rep to close the cycle.",
        durationMinutes="1–2 min",
        purpose="Full-cycle comparison against baseline + Day 7.",
        instructions=[
            "Read the same prompt as your baseline: " + baseline_prompt_hint,
            "Apply everything you've built over two weeks.",
            "Stop when you're done.",
        ],
        isCheckpoint=True, targetSeconds=45,
    )

    day_titles = [
        "Groundwork", "Warmup Habits", "Articulation", "Pace + Pauses",
        "Filler Control", "Confident Endings", "Day 7 Checkpoint",
        "Prosody", "Resonance", "Projection",
        "Sustained Practice", "Real-World Reps", "Consolidation", "Day 14 Checkpoint",
    ]
    days: List[VoiceRefinementPlanDay] = []
    for i, title in enumerate(day_titles, start=1):
        extras: List[VoiceRefinementPlanActivity] = []
        if i == 7:
            extras.append(checkpoint_d7)
        if i == 14:
            extras.append(checkpoint_d14)
        days.append(_day(i, title, extras))

    return VoiceRefinementPlan(
        totalDays=14,
        days=days,
        clinicalNote=clinical_note,
    )


def _catalog_to_json(catalog: Iterable[VfCatalogItem]) -> str:
    return json.dumps([
        {
            "id": c.id,
            "day": c.day,
            "title": c.title,
            "type": c.type,
            "durationMinutes": c.duration_minutes,
        }
        for c in catalog
    ], indent=2)


async def _call_claude_for_plan(**prompt_vars: Any) -> Optional[dict]:
    prompt = VOICE_REFINEMENT_PLAN_PROMPT.format(**prompt_vars)
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        cleaned = strip_fences("\n".join(text_blocks).strip())
        return json.loads(cleaned)
    except Exception:
        return None


def _validate_plan_dict(raw: dict, expect_out_of_scope: bool, clinical_note: Optional[str]) -> Optional[VoiceRefinementPlan]:
    """Validate the parsed plan against pydantic and enforce the hard rules.

    Returns None if any hard rule is violated — the caller can decide to retry.
    """
    try:
        plan = VoiceRefinementPlan.model_validate(raw)
    except ValidationError:
        return None

    if plan.total_days != 14 or len(plan.days) != 14:
        return None

    seen_days = {d.day for d in plan.days}
    if seen_days != set(range(1, 15)):
        return None

    ids_seen: set[str] = set()
    checkpoints = {"d7": False, "d14": False}
    for day in plan.days:
        if not (3 <= len(day.activities) <= 5):
            return None
        for activity in day.activities:
            if activity.day != day.day:
                return None
            if not activity.id.startswith(f"vr-d{day.day}-"):
                return None
            if activity.id in ids_seen:
                return None
            ids_seen.add(activity.id)
            if activity.type == "recording" and activity.is_checkpoint:
                if day.day == 7:
                    checkpoints["d7"] = True
                elif day.day == 14:
                    checkpoints["d14"] = True

    if not (checkpoints["d7"] and checkpoints["d14"]):
        return None

    if expect_out_of_scope:
        # We required the plan to reflect the clinical note when pitch is out-of-scope.
        if clinical_note and (plan.clinical_note or "").strip() == "":
            plan.clinical_note = clinical_note

    return plan


async def generate_plan(
    *,
    baseline_features: dict,
    natural: NaturalRangeHz,
    band: BandFeatures,
    transcript: str,
    intent: TargetIntent,
    classification: TargetClassification,
    recommended_focus: List[str],
    clinical_note: Optional[str],
    vf_catalog: List[VfCatalogItem],
    baseline_prompt: str,
) -> VoiceRefinementPlan:
    """Two-shot LLM call with strict validation + deterministic fallback."""
    metrics = baseline_features.get("metrics", {}) or {}
    prompt_vars = {
        "mean_f0": round(natural.mean, 1),
        "mean_f0_minus_10": max(ABS_FLOOR_HZ, round(natural.mean - 10.0, 1)) if natural.mean else 0,
        "q1": round(natural.q1, 1),
        "q3": round(natural.q3, 1),
        "low": round(natural.low, 1),
        "high": round(natural.high, 1),
        "std_f0": round(metrics.get("pitch_std_hz") or 0.0, 1),
        "wpm": round(metrics.get("words_per_minute") or 0.0, 1),
        "hnr": round(metrics.get("hnr_db") or 0.0, 1),
        "filler_count": (
            (baseline_features.get("transcript_analysis") or {})
            .get("filler_analysis", {})
            .get("count", 0)
        ),
        "low_energy": band.low_band_energy,
        "mid_energy": band.mid_band_energy,
        "high_energy": band.high_band_energy,
        "transcript": (transcript or "")[:600],
        "pitch_semitones": round(intent.pitch_semitones or 0.0, 2),
        "speed_ratio": round(intent.speed_ratio or 1.0, 3),
        "resonance": round(intent.resonance or 0.0, 2),
        "brightness": round(intent.brightness or 0.0, 2),
        "pitch_classification": classification.pitch,
        "classification_notes": " | ".join(classification.notes) or "n/a",
        "recommended_focus": " | ".join(recommended_focus) or "n/a",
        "clinical_note": clinical_note or "",
        "vf_catalog_json": _catalog_to_json(vf_catalog),
        "baseline_prompt": (baseline_prompt or "")[:800],
    }

    expect_out_of_scope = classification.pitch == "out_of_scope"

    for _ in range(2):
        raw = await _call_claude_for_plan(**prompt_vars)
        if not raw:
            continue
        plan = _validate_plan_dict(raw, expect_out_of_scope, clinical_note)
        if plan is not None:
            return plan

    # Fall back to a deterministic minimal plan — never 500.
    return _minimal_fallback_plan(
        classification=classification,
        focus=recommended_focus,
        clinical_note=clinical_note,
        baseline_prompt_hint=(baseline_prompt or "your baseline prompt")[:200],
    )
