import uuid

from models.daily_plan import GenerateDailyPlanRequest, GenerateDailyPlanResponse
from services.daily_plan import generate_daily_plan
from utils.logging import log


async def handle_generate_daily_plan(req: GenerateDailyPlanRequest) -> GenerateDailyPlanResponse:
    req_id = uuid.uuid4().hex[:8]
    log("/generate-daily-plan", req_id, req.user_id, event="start")
    result = await generate_daily_plan(req)
    log("/generate-daily-plan", req_id, req.user_id, event="done", items=len(result.items))
    return result
