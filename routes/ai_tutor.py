from fastapi import APIRouter, Depends

from controllers.ai_tutor import handle_analyze_ai_tutor
from models.ai_tutor import AITutorAnalyzeRequest, AITutorReportResponse
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post("/analyze-ai-tutor", response_model=AITutorReportResponse, dependencies=[Depends(verify_internal_secret)])
async def analyze_ai_tutor(req: AITutorAnalyzeRequest) -> AITutorReportResponse:
    return await handle_analyze_ai_tutor(req)
