"""
Reading programme — Claude generation service (skeleton + stage content).

Mirrors the validate/retry structure of `generate_plan` in
services/voice_refinement.py, except failures RAISE after one retry —
koras-api owns the template fallback, this service never invents content.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from config import CLAUDE_MODEL, anthropic_client
from models.reading import (
    GenerateReadingProgramRequest,
    GenerateReadingProgramResponse,
    GenerateReadingStageRequest,
    GenerateReadingStageResponse,
    StageContent,
)
from prompts.reading_program import READING_PROGRAM_PROMPT
from prompts.reading_stage import READING_STAGE_PROMPT
from services.reading_analysis import default_guide_wpm
from utils.text import strip_fences

MIN_STAGES, MAX_STAGES = 5, 7
MIN_LESSONS, MAX_LESSONS = 3, 5
MIN_STEPS, MAX_STEPS = 1, 6
# The prompt asks for 60–120-word passages; validate with slack so a
# 58-word passage doesn't burn the single retry, but still reject
# degenerate output.
MIN_PASSAGE_WORDS, MAX_PASSAGE_WORDS = 40, 160

EXCLUDE_WORDS_CAP = 60


async def _call_claude_json(prompt: str, max_tokens: int = 8000) -> Optional[dict]:
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return json.loads(strip_fences("\n".join(text_blocks).strip()))
    except Exception:
        return None


def _level_hint(req: GenerateReadingProgramRequest) -> str:
    if req.profile.persona == "student" and req.profile.grade_level:
        return f"grade_{req.profile.grade_level}"
    return req.profile.persona or "professional"


def _validate_stage_content(content: StageContent, exclude_words: list[str]) -> bool:
    """Hard structural rules; returns False so the caller can retry.

    Repairs in place where safe: lesson.xp is coerced to the step sum by the
    model validator, and target_vocab entries that violate the exclude list
    (or point at an unknown lesson) are dropped.
    """
    if not (MIN_LESSONS <= len(content.lessons) <= MAX_LESSONS):
        return False

    lesson_ids: set[str] = set()
    step_ids: set[str] = set()
    for lesson in content.lessons:
        lid = lesson.lesson_id.strip()
        if not lid or lid in lesson_ids:
            return False
        lesson_ids.add(lid)
        if not (MIN_STEPS <= len(lesson.steps) <= MAX_STEPS):
            return False
        for step in lesson.steps:
            sid = step.step_id.strip()
            if not sid or sid in step_ids:
                return False
            step_ids.add(sid)
            if step.xp < 1:
                return False
            if step.type == "echo":
                if not (step.sentence or "").strip():
                    return False
            else:
                words = len((step.passage or "").split())
                if not (MIN_PASSAGE_WORDS <= words <= MAX_PASSAGE_WORDS):
                    return False

    excluded = {w.strip().lower() for w in exclude_words}
    content.target_vocab = [
        v for v in content.target_vocab
        if v.word.strip()
        and v.word.strip().lower() not in excluded
        and (v.in_lesson is None or v.in_lesson in lesson_ids)
    ]
    return True


def _validate_program_dict(
    raw: dict, exclude_words: list[str]
) -> Optional[GenerateReadingProgramResponse]:
    try:
        resp = GenerateReadingProgramResponse.model_validate(raw)
    except ValidationError:
        return None
    if not (MIN_STAGES <= len(resp.skeleton) <= MAX_STAGES):
        return None
    if [s.position for s in resp.skeleton] != list(range(1, len(resp.skeleton) + 1)):
        return None
    if any(not s.title.strip() for s in resp.skeleton):
        return None
    if not _validate_stage_content(resp.stage_1_content, exclude_words):
        return None
    return resp


def _prompt_vars(req: GenerateReadingProgramRequest) -> dict:
    profile = req.profile
    return {
        "persona": profile.persona,
        "grade_level": profile.grade_level if profile.grade_level is not None else "null",
        "intent": (profile.intent or "").strip() or "improve reading fluency",
        "goals_json": json.dumps(profile.goals),
        "onboarding_json": json.dumps(req.onboarding.model_dump() if req.onboarding else None),
        "baseline_json": json.dumps(req.baseline.model_dump() if req.baseline else None),
        "calibration_json": json.dumps(req.calibration.model_dump()),
        "exclude_words_json": json.dumps(req.exclude_words[:EXCLUDE_WORDS_CAP]),
        "default_guide_wpm": default_guide_wpm(_level_hint(req)),
    }


async def generate_reading_program(
    req: GenerateReadingProgramRequest,
) -> GenerateReadingProgramResponse:
    """One-shot programme generation: skeleton + Stage 1 content.

    Validate, retry once on invalid output, then raise — koras-api applies
    the template fallback (§3.3).
    """
    prompt = READING_PROGRAM_PROMPT.format(**_prompt_vars(req))
    for _ in range(2):
        raw = await _call_claude_json(prompt)
        if not raw:
            continue
        resp = _validate_program_dict(raw, req.exclude_words)
        if resp is not None:
            return resp
    raise RuntimeError("reading programme generation failed after retry")


async def generate_reading_stage(
    req: GenerateReadingStageRequest,
) -> GenerateReadingStageResponse:
    prompt_vars = _prompt_vars(req)
    prompt_vars["skeleton_entry_json"] = json.dumps(req.skeleton_entry.model_dump())
    prompt_vars["performance_json"] = json.dumps(
        req.completed_stage_performance.model_dump()
        if req.completed_stage_performance else None
    )
    prompt_vars["stage_position"] = req.skeleton_entry.position
    prompt = READING_STAGE_PROMPT.format(**prompt_vars)

    for _ in range(2):
        raw = await _call_claude_json(prompt)
        if not raw:
            continue
        try:
            resp = GenerateReadingStageResponse.model_validate(raw)
        except ValidationError:
            continue
        if _validate_stage_content(resp.stage_content, req.exclude_words):
            return resp
    raise RuntimeError("reading stage generation failed after retry")
