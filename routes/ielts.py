from fastapi import APIRouter, Depends, Request

from controllers.ielts import handle_analyze_ielts
from models.ielts import IELTSReportModel
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post("/analyze-ielts-speaking", response_model=IELTSReportModel, dependencies=[Depends(verify_internal_secret)])
async def analyze_ielts_speaking(request: Request) -> IELTSReportModel:
    return await handle_analyze_ielts(request)
