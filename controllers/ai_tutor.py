import uuid

from fastapi import HTTPException

from models.ai_tutor import AITutorAnalyzeRequest, AITutorReportResponse
from services.ai_tutor import grade_ai_tutor_session
from utils.logging import log


async def handle_analyze_ai_tutor(req: AITutorAnalyzeRequest) -> AITutorReportResponse:
    req_id = uuid.uuid4().hex[:8]
    log("/analyze-ai-tutor", req_id, req.user_id, session_id=req.session_id, turns=len(req.turns))

    if not req.turns:
        raise HTTPException(400, "No conversation turns provided.")
    if not any(t.role == "user" for t in req.turns):
        raise HTTPException(400, "No user turns found in conversation.")
    user_text = " ".join(t.transcript for t in req.turns if t.role == "user")
    if len(user_text.strip()) < 20:
        raise HTTPException(400, "User responses too short for meaningful analysis.")

    report = await grade_ai_tutor_session(req)
    log("/analyze-ai-tutor", req_id, req.user_id, event="done",
        overall=report.overall, user_turns=sum(1 for t in req.turns if t.role == "user"))
    return report
