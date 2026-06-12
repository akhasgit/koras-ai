import asyncio
import uuid

from fastapi import HTTPException, Request

from config import MAX_UPLOAD_BYTES, extract_sem
from models.interview import (
    GenerateInterviewQuestionsRequest,
    GenerateInterviewQuestionsResponse,
    InterviewAnalyzeJSONRequest,
    InterviewAnswerReport,
)
from services.audio import extract_features, normalize_audio, transcribe
from services.interview import (
    generate_interview_questions,
    grade_interview_answer,
    interview_fallback_question_list,
)
from utils.logging import log


async def handle_generate_questions(req: GenerateInterviewQuestionsRequest) -> GenerateInterviewQuestionsResponse:
    req_id = uuid.uuid4().hex[:8]
    log("/generate-interview-questions", req_id, None, role=req.jobRole, type=req.interviewType)

    if not any([(req.description or "").strip(), (req.jobRole or "").strip(),
                (req.notes or "").strip(), (req.title or "").strip()]):
        return GenerateInterviewQuestionsResponse(
            questions=interview_fallback_question_list(), extractedContext={},
            warning="No scenario context provided — returning default questions.",
        )

    result = await generate_interview_questions(
        title=req.title, job_role=req.jobRole, company=req.company,
        interview_type=req.interviewType, experience_level=req.experienceLevel,
        description=req.description, notes=req.notes,
    )
    log("/generate-interview-questions", req_id, None, event="done", questions=len(result.questions))
    return result


async def handle_analyze_interview_answer(request: Request) -> InterviewAnswerReport:
    req_id = uuid.uuid4().hex[:8]
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = await request.json()
        req = InterviewAnalyzeJSONRequest(**body)
        log("/analyze-interview-answer", req_id, None, mode="json", question_type=req.questionType)
        transcript = (req.transcript or "").strip()
        if len(transcript.split()) < 5:
            raise HTTPException(400, "Transcript too short for analysis (minimum 5 words).")
        report = await grade_interview_answer(
            question=req.question, question_type=req.questionType,
            scenario_title=req.scenarioTitle, job_role=req.jobRole,
            company=req.company, description=req.description,
            transcript=transcript, duration=req.durationSeconds, acoustic_metrics=None,
        )
        return InterviewAnswerReport(**report)

    form = await request.form()
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    question = str(form.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "Missing 'question' field.")
    question_type = str(form.get("questionType") or "").strip() or None
    scenario_title = str(form.get("scenarioTitle") or "") or None
    job_role = str(form.get("jobRole") or "") or None
    company = str(form.get("company") or "") or None
    description = str(form.get("description") or "") or None
    user_id = str(form.get("user_id") or "") or None
    duration_field = form.get("durationSeconds")
    duration: int | None = None
    if duration_field is not None:
        try:
            duration = int(str(duration_field))
        except (TypeError, ValueError):
            pass

    log("/analyze-interview-answer", req_id, user_id, mode="audio", question_type=question_type)

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
        raise HTTPException(400, "We couldn't detect a meaningful answer. Please try again and speak for at least a few sentences.")

    measured_duration = duration if duration and duration > 0 else int(features["duration"])
    acoustic_metrics = {
        "duration": features["duration"], "mean_f0": features["mean_f0"],
        "std_f0": features["std_f0"], "hnr": features["hnr"],
        "pause_count": features["pause_count"], "long_pause_count": features["long_pause_count"],
        "spectral_centroid": features["spectral_centroid"],
    }

    report = await grade_interview_answer(
        question=question, question_type=question_type, scenario_title=scenario_title,
        job_role=job_role, company=company, description=description,
        transcript=transcript_text, duration=measured_duration, acoustic_metrics=acoustic_metrics,
    )

    if report["metrics"].get("longPauseCount") is None:
        report["metrics"]["longPauseCount"] = int(features["long_pause_count"])

    log("/analyze-interview-answer", req_id, user_id, event="done",
        question_type=question_type, duration=measured_duration,
        words=len(transcript_text.split()), overall=report["scores"]["overall"])

    return InterviewAnswerReport(**report)
