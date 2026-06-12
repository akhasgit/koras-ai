import json

from config import anthropic_client, CLAUDE_MODEL
from models.ai_tutor import (
    AITutorAnalyzeRequest,
    AITutorFeedback,
    AITutorReportResponse,
    AITutorScores,
)
from prompts.ai_tutor import AI_TUTOR_GRADING_PROMPT
from services.scoring import weighted_overall
from utils.text import clamp, count_fillers, strip_fences


def build_fallback_report(
    req: AITutorAnalyzeRequest, user_text: str, filler_count: int, wpm: float | None
) -> AITutorReportResponse:
    word_count = len(user_text.split())
    user_turn_count = sum(1 for t in req.turns if t.role == "user")
    raw = 50 + min(15, word_count // 10) + min(10, user_turn_count * 2) - min(20, filler_count * 3)
    filler_score = clamp(100 - filler_count * 8)
    scores_dict = {
        "relevance": clamp(raw + 5), "eloquence": clamp(raw - 5), "fluency": clamp(raw),
        "grammar": clamp(raw), "tense": clamp(raw), "fillerControl": filler_score,
        "clarity": clamp(raw), "confidence": clamp(raw - 3), "vocabulary": clamp(raw - 2),
        "listening": clamp(raw + 3),
    }
    return AITutorReportResponse(
        overall=weighted_overall(scores_dict),
        scores=AITutorScores(**scores_dict),
        metrics={
            "durationSeconds": req.duration_seconds,
            "wordsPerMinute": round(wpm, 1) if wpm else None,
            "fillerCount": filler_count,
            "fillerRatePerMinute": (
                round(filler_count / (req.duration_seconds / 60), 1)
                if req.duration_seconds and req.duration_seconds > 0 else None
            ),
            "longPauseCount": None, "grammarIssueCount": None, "tenseIssueCount": None,
        },
        feedback=AITutorFeedback(
            summary="Analysis completed with limited data. Practice more for a detailed breakdown.",
            strengths=["Completed a conversation session"],
            improvements=["Try giving longer, more detailed answers"],
            repeatedMistakes=[],
        ),
        turnFeedback=[],
    )


async def grade_ai_tutor_session(req: AITutorAnalyzeRequest) -> AITutorReportResponse:
    user_turns = [t for t in req.turns if t.role == "user"]
    user_text = " ".join(t.transcript for t in user_turns)
    filler_count = count_fillers(user_text)
    word_count = len(user_text.split())
    wpm: float | None = (word_count / req.duration_seconds * 60) if req.duration_seconds and req.duration_seconds > 0 else None

    lines = [
        f"[{'Tutor' if t.role == 'assistant' else 'Student'}]: {t.transcript}"
        for t in sorted(req.turns, key=lambda x: x.turn_index or 0)
    ]
    metadata_parts = []
    if req.duration_seconds:
        metadata_parts.append(f"Duration: {req.duration_seconds}s")
    if wpm:
        metadata_parts.append(f"Approximate WPM: {round(wpm, 1)}")
    metadata_parts += [f"Filler words detected: {filler_count}", f"Total user words: {word_count}"]

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": AI_TUTOR_GRADING_PROMPT.format(
                transcript="\n".join(lines),
                metadata_section="Metadata:\n" + "\n".join(metadata_parts),
            )}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))

        scores_raw = data.get("scores", {})
        for k in scores_raw:
            scores_raw[k] = clamp(scores_raw[k])
        data["overall"] = weighted_overall(scores_raw)

        llm_metrics = data.get("metrics", {})
        llm_metrics["durationSeconds"] = req.duration_seconds
        llm_metrics["wordsPerMinute"] = round(wpm, 1) if wpm else None
        llm_metrics["fillerCount"] = max(filler_count, llm_metrics.get("fillerCount", 0))
        if req.duration_seconds and req.duration_seconds > 0:
            llm_metrics["fillerRatePerMinute"] = round(
                llm_metrics["fillerCount"] / (req.duration_seconds / 60), 1
            )
        data["metrics"] = llm_metrics
        return AITutorReportResponse(**data)
    except Exception as e:
        return build_fallback_report(req, user_text, filler_count, wpm)
