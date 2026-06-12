import json

from config import anthropic_client, CLAUDE_MODEL
from models.analyze import (
    ConfidenceMarkers,
    FillerAnalysis,
    PhrasingRewrite,
    TranscriptAnalysis,
    VoiceScores,
)
from prompts.analyze import COACH_FEEDBACK_PROMPT, TRANSCRIPT_ANALYSIS_PROMPT
from utils.text import strip_fences


async def analyze_transcript(transcript: str) -> TranscriptAnalysis:
    msg = await anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": TRANSCRIPT_ANALYSIS_PROMPT.format(transcript=transcript)}],
    )
    text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    cleaned = strip_fences("\n".join(text_blocks).strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude returned invalid JSON: {cleaned[:300]}") from e

    fa = data.get("filler_analysis", {})
    cm = data.get("confidence_markers", {})
    return TranscriptAnalysis(
        topic_summary=data.get("topic_summary", ""),
        filler_analysis=FillerAnalysis(
            count=int(fa.get("count", 0)),
            rate_per_minute=float(fa.get("rate_per_minute", 0)),
            fillers_used=fa.get("fillers_used", []),
            worst_sentences=fa.get("worst_sentences", []),
        ),
        phrasing_rewrites=[PhrasingRewrite(**r) for r in data.get("phrasing_rewrites", [])],
        clarity_issues=data.get("clarity_issues", []),
        confidence_markers=ConfidenceMarkers(
            hedging_words=cm.get("hedging_words", []),
            count=int(cm.get("count", 0)),
        ),
        suggested_revision=data.get("suggested_revision", ""),
    )


async def generate_coach_feedback(scores: VoiceScores, transcript_analysis: TranscriptAnalysis) -> str:
    msg = await anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": COACH_FEEDBACK_PROMPT.format(
            pitch=scores.pitch, pace=scores.pace, clarity=scores.clarity,
            resonance=scores.resonance, confidence=scores.confidence, overall=scores.overall,
            filler_count=transcript_analysis.filler_analysis.count,
            hedging_count=transcript_analysis.confidence_markers.count,
            topic_summary=transcript_analysis.topic_summary,
        )}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip().strip('"').strip()
