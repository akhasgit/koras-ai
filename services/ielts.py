import json

from config import anthropic_client, CLAUDE_MODEL
from prompts.ielts import IELTS_GRADING_PROMPT, IELTS_NORMALIZE_PROMPT
from services.scoring import compute_practice_band
from utils.text import clamp, count_fillers, strip_fences


async def normalize_ielts_transcript(raw_transcript: str) -> dict:
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": IELTS_NORMALIZE_PROMPT.format(transcript=raw_transcript)}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))
        return {
            "clean_transcript": str(data.get("clean_transcript", raw_transcript)),
            "detected_languages": data.get("detected_languages", ["English"]),
            "code_switching_detected": bool(data.get("code_switching_detected", False)),
            "non_english_fragments": data.get("non_english_fragments", []),
        }
    except Exception:
        return {
            "clean_transcript": raw_transcript,
            "detected_languages": ["English"],
            "code_switching_detected": False,
            "non_english_fragments": [],
        }


def ielts_fallback_report(
    req_part: str, transcript: str, duration: int | None, filler_count: int, normalization: dict
) -> dict:
    word_count = len(transcript.split())
    wpm = (word_count / duration) * 60 if duration and duration > 0 else None
    raw_score = max(40, min(80, 55 + min(15, word_count // 12) - min(20, filler_count * 3)))
    criteria = {
        "fluencyCoherence": raw_score, "lexicalResource": max(40, raw_score - 3),
        "grammarRangeAccuracy": max(40, raw_score - 2), "pronunciation": max(45, raw_score),
    }
    overall_band, per_band = compute_practice_band(criteria)
    return {
        "criteriaScores": criteria, "criteriaBand": per_band,
        "practiceBandEstimate": overall_band, "overallScore": int(sum(criteria.values()) / 4),
        "korasMetrics": {
            "wordsPerMinute": round(wpm, 1) if wpm else None,
            "fillerCount": filler_count,
            "fillerRatePerMinute": (round(filler_count / (duration / 60), 1) if duration and duration > 0 else None),
            "longPauseCount": None, "answerRelevance": 60, "structureScore": 55,
            "specificExampleScore": 50, "vocabularyRangeScore": 55, "grammarIssueCount": 0,
            "tenseIssueCount": 0, "clarityScore": 60, "pronunciationIntelligibility": 65,
            "codeSwitchingDetected": normalization["code_switching_detected"],
            "normalizedTranscript": normalization["clean_transcript"],
            "detectedLanguages": normalization["detected_languages"],
            "durationSeconds": duration,
        },
        "feedback": {
            "summary": "Analysis ran with limited grading data — record again to get more accurate feedback.",
            "strengths": ["You completed the recording."],
            "improvements": ["Try giving a longer answer next time.", "Aim to use the structure taught in the lesson."],
            "ieltsAdvice": [
                "Keep your answer in English throughout." if normalization["code_switching_detected"]
                else "Focus on staying organized — answer, reason, example."
            ],
            "bestSentence": "", "weakerSentence": "", "strongerVersion": "",
            "nextPracticeFocus": "Try the prompt again with one more example.",
        },
        "transcriptFeedback": [], "nextRecommendedLessonId": None, "transcript": transcript,
    }


async def analyze_ielts_core(
    part: str, prompt_text: str, transcript: str,
    duration: int | None, acoustic_metrics: dict | None,
) -> dict:
    normalization = await normalize_ielts_transcript(transcript)
    filler_count = count_fillers(normalization["clean_transcript"])

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": IELTS_GRADING_PROMPT.format(
                part=part, prompt=prompt_text,
                raw_transcript=transcript,
                normalized_transcript=normalization["clean_transcript"],
                code_switching_detected=normalization["code_switching_detected"],
                duration_seconds=duration if duration else "unknown",
                acoustic_metrics=json.dumps(acoustic_metrics or {}),
            )}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))
    except Exception:
        return ielts_fallback_report(part, transcript, duration, filler_count, normalization)

    criteria = data.get("criteriaScores", {})
    for k in ("fluencyCoherence", "lexicalResource", "grammarRangeAccuracy", "pronunciation"):
        criteria[k] = clamp(int(criteria.get(k, 50)))

    overall_band, per_band = compute_practice_band(criteria)
    koras_metrics = data.get("korasMetrics", {}) or {}
    koras_metrics["fillerCount"] = max(filler_count, int(koras_metrics.get("fillerCount", 0)))
    if duration and duration > 0:
        koras_metrics["fillerRatePerMinute"] = round(koras_metrics["fillerCount"] / (duration / 60), 1)
        koras_metrics["wordsPerMinute"] = round(
            (len(normalization["clean_transcript"].split()) / duration) * 60, 1
        )
    koras_metrics.update({
        "codeSwitchingDetected": normalization["code_switching_detected"],
        "normalizedTranscript": normalization["clean_transcript"],
        "detectedLanguages": normalization["detected_languages"],
        "durationSeconds": duration,
    })

    feedback = data.get("feedback", {}) or {}
    if normalization["code_switching_detected"]:
        advice = list(feedback.get("ieltsAdvice") or [])
        advice.insert(0, "In IELTS practice, try to keep your full answer in English.")
        feedback["ieltsAdvice"] = advice

    return {
        "criteriaScores": criteria, "criteriaBand": per_band,
        "practiceBandEstimate": overall_band,
        "overallScore": int(sum(criteria.values()) / 4),
        "korasMetrics": koras_metrics, "feedback": feedback,
        "transcriptFeedback": data.get("transcriptFeedback") or [],
        "nextRecommendedLessonId": data.get("nextRecommendedLessonId"),
        "transcript": transcript,
    }
