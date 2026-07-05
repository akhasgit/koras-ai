from fastapi import APIRouter, Depends, Request

from controllers.voice_refinement import handle_analyze_voice_refinement
from models.voice_refinement import VoiceRefinementReport
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/analyze-voice-refinement",
    response_model=VoiceRefinementReport,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_voice_refinement(request: Request) -> VoiceRefinementReport:
    return await handle_analyze_voice_refinement(request)
