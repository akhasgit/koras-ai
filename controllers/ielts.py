import asyncio
import uuid

from fastapi import HTTPException, Request

from config import MAX_UPLOAD_BYTES, extract_sem
from models.ielts import IELTSAnalyzeRequest, IELTSReportModel
from services.audio import extract_features, normalize_audio, transcribe
from services.ielts import analyze_ielts_core
from utils.logging import log


async def handle_analyze_ielts(request: Request) -> IELTSReportModel:
    req_id = uuid.uuid4().hex[:8]
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = await request.json()
        req = IELTSAnalyzeRequest(**body)
        log("/analyze-ielts-speaking", req_id, req.user_id, mode="json", part=req.part)
        if not req.transcript or len(req.transcript.split()) < 5:
            raise HTTPException(400, "Transcript too short for analysis (minimum 5 words).")
        report = await analyze_ielts_core(
            part=req.part, prompt_text=req.prompt, transcript=req.transcript,
            duration=req.duration_seconds, acoustic_metrics=req.acoustic_metrics,
        )
        return IELTSReportModel(**report)

    form = await request.form()
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    part = str(form.get("part") or "part_1")
    prompt_text = str(form.get("prompt") or "")
    lesson_id = str(form.get("lesson_id") or "ielts")
    user_id = str(form.get("user_id") or "") or None
    log("/analyze-ielts-speaking", req_id, user_id, mode="audio", part=part, lesson_id=lesson_id)

    async with extract_sem:
        try:
            filename = getattr(audio_field, "filename", "audio") or "audio"
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, filename)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        try:
            features, transcript_data = await asyncio.gather(
                asyncio.to_thread(extract_features, wav_bytes),
                transcribe(wav_bytes),
            )
        except Exception as e:
            raise HTTPException(500, f"Audio analysis failed: {e}") from e

    transcript_text = (transcript_data.get("text") or "").strip()
    if len(transcript_text.split()) < 5:
        raise HTTPException(400, "Audio too short or unclear. Please speak for longer (at least 5 words).")

    duration = int(features["duration"])
    acoustic_metrics = {
        "duration": features["duration"], "mean_f0": features["mean_f0"],
        "std_f0": features["std_f0"], "hnr": features["hnr"],
        "pause_count": features["pause_count"], "long_pause_count": features["long_pause_count"],
        "spectral_centroid": features["spectral_centroid"],
    }

    report = await analyze_ielts_core(
        part=part, prompt_text=prompt_text, transcript=transcript_text,
        duration=duration, acoustic_metrics=acoustic_metrics,
    )

    if part == "part_2" and duration < 60:
        advice = list(report["feedback"].get("ieltsAdvice") or [])
        advice.append(f"Part 2 long turns should run close to 2 minutes — you stopped at {duration}s.")
        report["feedback"]["ieltsAdvice"] = advice

    if report["korasMetrics"].get("longPauseCount") is None:
        report["korasMetrics"]["longPauseCount"] = int(features["long_pause_count"])

    log("/analyze-ielts-speaking", req_id, user_id, event="done",
        lesson_id=lesson_id, part=part, duration=duration, band=report["practiceBandEstimate"])

    return IELTSReportModel(**report)
