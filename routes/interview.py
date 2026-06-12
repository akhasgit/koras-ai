from fastapi import APIRouter, Depends, Request

from controllers.interview import handle_analyze_interview_answer, handle_generate_questions
from models.interview import (
    GenerateInterviewQuestionsRequest,
    GenerateInterviewQuestionsResponse,
    InterviewAnswerReport,
)
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post("/generate-interview-questions", response_model=GenerateInterviewQuestionsResponse, dependencies=[Depends(verify_internal_secret)])
async def generate_interview_questions(req: GenerateInterviewQuestionsRequest) -> GenerateInterviewQuestionsResponse:
    return await handle_generate_questions(req)


@router.post("/analyze-interview-answer", response_model=InterviewAnswerReport, dependencies=[Depends(verify_internal_secret)])
async def analyze_interview_answer(request: Request) -> InterviewAnswerReport:
    return await handle_analyze_interview_answer(request)
