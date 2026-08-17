import json

from config import anthropic_client, CLAUDE_MODEL
from models.daily_plan import DailyPlanItemModel, GenerateDailyPlanRequest, GenerateDailyPlanResponse
from prompts.daily_plan import DAILY_PLAN_PROMPT
from utils.text import strip_fences

DAILY_PLAN_ALLOWED_PROGRAMS = {
    "ai-tutor",
    "ielts-speaking",
    "interview-prep",
    "daily-vocabulary",
    "speech-clarity",
}
DAILY_PLAN_ROUTE_BY_PROGRAM = {
    "ai-tutor": "/ai-tutor",
    "ielts-speaking": "/ielts",
    "interview-prep": "/interview-prep",
    "daily-vocabulary": "/vocabulary",
    "speech-clarity": "/programs/speech-clarity",
}
DAILY_PLAN_ALLOWED_TYPES = {"program_session", "review", "reflection", "streak_save"}


def _coerce_int(raw, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _build_items_from_raw(raw_items: list, rules_items_by_program: dict | None = None) -> list[DailyPlanItemModel]:
    out: list[DailyPlanItemModel] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        program_id = raw.get("program_id")
        if program_id not in DAILY_PLAN_ALLOWED_PROGRAMS:
            continue
        route = raw.get("route") or DAILY_PLAN_ROUTE_BY_PROGRAM.get(program_id, "")
        if not route:
            continue
        type_ = raw.get("type", "program_session")
        if type_ not in DAILY_PLAN_ALLOWED_TYPES:
            type_ = "program_session"
        rules_item = (rules_items_by_program or {}).get(program_id, {})
        priority = max(1, min(3, _coerce_int(raw.get("priority"), _coerce_int(rules_item.get("priority"), 2))))
        minutes = max(2, min(45, _coerce_int(raw.get("estimated_minutes"), _coerce_int(rules_item.get("estimated_minutes"), 10))))
        title = (str(raw.get("title") or rules_item.get("title") or "")).strip()[:120]
        reason = (str(raw.get("reason") or rules_item.get("reason") or "")).strip()[:280]
        if not title or not reason:
            continue
        item_id = (
            str(raw["item_id"]) if isinstance(raw.get("item_id"), str) and raw.get("item_id")
            else rules_item.get("item_id")
        )
        out.append(DailyPlanItemModel(
            item_id=item_id, type=type_, program_id=str(program_id), route=str(route),
            title=title, reason=reason, estimated_minutes=minutes, priority=priority,
            status="pending", completed_at=None,
        ))
        if len(out) >= 3:
            break
    return out


def build_fallback(rules_plan: dict) -> GenerateDailyPlanResponse:
    items = _build_items_from_raw(rules_plan.get("items") or [])
    return GenerateDailyPlanResponse(
        summary=str(rules_plan.get("summary") or "")[:500],
        focus_area=str(rules_plan.get("focus_area") or "")[:80],
        advice=str(rules_plan.get("advice") or "")[:800],
        items=items,
    )


async def generate_daily_plan(req: GenerateDailyPlanRequest) -> GenerateDailyPlanResponse:
    rules_plan = req.rules_plan or {}
    fallback = build_fallback(rules_plan)
    if not fallback.items:
        return fallback

    try:
        signals_json = json.dumps(req.signals, ensure_ascii=False)[:6000]
        rules_plan_json = json.dumps(rules_plan, ensure_ascii=False)[:3000]
    except (TypeError, ValueError):
        return fallback

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=900,
            messages=[{"role": "user", "content": DAILY_PLAN_PROMPT.format(
                signals_json=signals_json, rules_plan_json=rules_plan_json,
            )}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))
    except Exception:
        return fallback

    if not isinstance(data, dict):
        return fallback

    rules_items_by_program = {
        str(r["program_id"]): r
        for r in (rules_plan.get("items") or [])
        if isinstance(r, dict) and r.get("program_id") in DAILY_PLAN_ALLOWED_PROGRAMS
    }
    out_items = _build_items_from_raw(data.get("items") or [], rules_items_by_program)
    if not out_items:
        return fallback

    return GenerateDailyPlanResponse(
        summary=(str(data.get("summary") or fallback.summary)).strip()[:500],
        focus_area=(str(data.get("focus_area") or fallback.focus_area)).strip()[:80],
        advice=(str(data.get("advice") or fallback.advice)).strip()[:800],
        items=out_items,
    )
