from fastapi import APIRouter, Depends

from controllers.daily_plan import handle_generate_daily_plan
from models.daily_plan import GenerateDailyPlanRequest, GenerateDailyPlanResponse
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post("/generate-daily-plan", response_model=GenerateDailyPlanResponse, dependencies=[Depends(verify_internal_secret)])
async def generate_daily_plan(req: GenerateDailyPlanRequest) -> GenerateDailyPlanResponse:
    return await handle_generate_daily_plan(req)
