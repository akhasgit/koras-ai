from typing import List

from pydantic import BaseModel


class DailyPlanItemModel(BaseModel):
    item_id: str | None = None
    type: str
    program_id: str
    route: str
    title: str
    reason: str
    estimated_minutes: int
    priority: int
    status: str = "pending"
    completed_at: str | None = None


class GenerateDailyPlanRequest(BaseModel):
    user_id: str
    signals: dict
    rules_plan: dict


class GenerateDailyPlanResponse(BaseModel):
    summary: str
    focus_area: str
    advice: str
    items: List[DailyPlanItemModel]
