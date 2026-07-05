"""
Voice Refinement — controller.

Orchestrates one call to `/analyze-voice-refinement`:
  1. Parse the multipart body (audio + target_intent + prompt_kind + vf_catalog).
  2. Normalise the audio through the shared ffmpeg helper.
  3. Extract acoustic features + natural range + band energies + Whisper transcript.
  4. Build a baseline AnalyzeResponse (reusing the same LLM helpers as /analyze).
  5. Classify the slider intent.
  6. Optionally generate the 14-day plan (only for prompt_kind = "baseline").
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import HTTPException, Request

from config import MAX_UPLOAD_BYTES, MIN_TRANSCRIPT_WORDS, extract_sem
from models.analyze import AcousticMetrics, AnalyzeResponse
from models.voice_refinement import (
    TargetIntent,
    VfCatalogItem,
    VoiceRefinementReport,
)
from services.audio import extract_features, normalize_audio, transcribe
from services.llm import analyze_transcript, generate_coach_feedback
from services.scoring import compute_voice_scores, pick_archetype
from services.voice_refinement import (
    classify_target_intent,
    compute_band_energies,
    compute_natural_range,
    generate_plan,
)
from utils.logging import log


def _parse_json_field(form: Any, field: str, default: Any) -> Any:
    raw = form.get(field)
    if raw is None:
        return default
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"'{field}' must be valid JSON: {e}") from e


async def handle_analyze_voice_refinement(request: Request) -> VoiceRefinementReport:
    req_id = uuid.uuid4().hex[:8]

    form = await request.form()

    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    prompt_kind = str(form.get("prompt_kind") or "baseline").strip()
    if prompt_kind not in ("baseline", "checkpoint_d7", "checkpoint_d14"):
        raise HTTPException(400, f"Unsupported prompt_kind: {prompt_kind}")

    intent_raw = _parse_json_field(form, "target_intent", {})
    if not isinstance(intent_raw, dict):
        raise HTTPException(400, "'target_intent' must be a JSON object.")
    try:
        intent = TargetIntent(**intent_raw)
    except Exception as e:
        raise HTTPException(400, f"Invalid target_intent: {e}") from e

    catalog_raw = _parse_json_field(form, "vf_activity_catalog", [])
    if not isinstance(catalog_raw, list):
        catalog_raw = []
    vf_catalog: list[VfCatalogItem] = []
    for item in catalog_raw:
        if not isinstance(item, dict):
            continue
        try:
            vf_catalog.append(VfCatalogItem(**item))
        except Exception:
            continue

    baseline_prompt = str(form.get("baseline_prompt") or "").strip()

    log("/analyze-voice-refinement", req_id, None, prompt_kind=prompt_kind,
        catalog_size=len(vf_catalog))

    # ── Audio normalise + feature extraction (shared with /analyze) ──────
    async with extract_sem:
        try:
            filename = getattr(audio_field, "filename", "audio") or "audio"
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, filename)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        try:
            features, transcript_data, natural, band = await asyncio.gather(
                asyncio.to_thread(extract_features, wav_bytes),
                transcribe(wav_bytes),
                asyncio.to_thread(compute_natural_range, wav_bytes),
                asyncio.to_thread(compute_band_energies, wav_bytes),
            )
        except Exception as e:
            raise HTTPException(500, f"Audio analysis failed: {e}") from e

    transcript_text = (transcript_data.get("text") or "").strip()
    if len(transcript_text.split()) < MIN_TRANSCRIPT_WORDS:
        raise HTTPException(400, "Audio too short or unclear. Please record at least 10 words.")

    # ── Build the baseline AnalyzeResponse (mirrors /analyze) ────────────
    try:
        transcript_analysis = await analyze_transcript(transcript_text)
    except Exception as e:
        raise HTTPException(500, f"Transcript analysis failed: {e}") from e

    scores, metrics = compute_voice_scores(features, transcript_data, transcript_analysis)

    if metrics.duration_seconds > 0:
        transcript_analysis.filler_analysis.rate_per_minute = round(
            transcript_analysis.filler_analysis.count / (metrics.duration_seconds / 60), 2
        )

    try:
        coach_feedback = await generate_coach_feedback(scores, transcript_analysis)
    except Exception as e:
        log("/analyze-voice-refinement", req_id, None,
            event="coach_feedback_fallback", error=str(e))
        coach_feedback = "Nice work on this rep. Try one thing on the next take: slow down slightly and let your pauses breathe."

    archetype = pick_archetype(scores)

    baseline_response = AnalyzeResponse(
        scores=scores,
        metrics=metrics,
        transcript=transcript_text,
        transcript_analysis=transcript_analysis,
        coach_feedback=coach_feedback,
        archetype=archetype,
    )

    # ── Intent classification (uses slider values + natural band) ────────
    classification, recommended_focus, clinical_note = classify_target_intent(intent, natural)

    log("/analyze-voice-refinement", req_id, None,
        event="classified",
        pitch_class=classification.pitch,
        semis=intent.pitch_semitones,
        band=f"{round(natural.low)}..{round(natural.high)}Hz")

    # ── Plan generation (only for prompt_kind = baseline) ────────────────
    plan = None
    if prompt_kind == "baseline":
        plan = await generate_plan(
            baseline_features=baseline_response.model_dump(),
            natural=natural,
            band=band,
            transcript=transcript_text,
            intent=intent,
            classification=classification,
            recommended_focus=recommended_focus,
            clinical_note=clinical_note,
            vf_catalog=vf_catalog,
            baseline_prompt=baseline_prompt,
        )

    log("/analyze-voice-refinement", req_id, None,
        event="done",
        overall=scores.overall,
        plan_generated=plan is not None)

    return VoiceRefinementReport(
        baseline=baseline_response,
        band_features=band,
        natural_range_hz=natural,
        target_intent=intent,
        target_classification=classification,
        recommended_focus=recommended_focus,
        clinical_note=clinical_note,
        plan=plan,
        prompt_kind=prompt_kind,
    )
