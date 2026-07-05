from fastapi import APIRouter, Depends, Request

from controllers.voice_foundations import handle_analyze_voice_foundations
from models.analyze import AnalyzeResponse
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/analyze-voice-foundations",
    response_model=AnalyzeResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_voice_foundations(request: Request) -> AnalyzeResponse:
    """Authenticated Voice Foundations analysis. Semantically identical to
    the public `POST /analyze` endpoint but gated by `x-koras-secret` and
    accepts an optional `activity_id` / `day` for logging."""
    return await handle_analyze_voice_foundations(request)
