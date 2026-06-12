from models.analyze import (
    AcousticMetrics,
    TranscriptAnalysis,
    VoiceScores,
)
from utils.text import clamp

SCORE_WEIGHTS = {
    "relevance": 0.15, "eloquence": 0.15, "fluency": 0.15,
    "grammar": 0.10, "tense": 0.10, "fillerControl": 0.10,
    "clarity": 0.10, "confidence": 0.05, "vocabulary": 0.05, "listening": 0.05,
}

INTERVIEW_SCORE_WEIGHTS = {
    "relevance": 0.20, "structure": 0.15, "specificity": 0.15,
    "delivery": 0.10, "confidence": 0.10, "fluency": 0.10,
    "grammar": 0.10, "conciseness": 0.05, "professionalism": 0.05,
}


def weighted_overall(scores: dict) -> int:
    return clamp(sum(scores.get(k, 50) * w for k, w in SCORE_WEIGHTS.items()))


def interview_weighted_overall(scores: dict) -> int:
    return clamp(sum(int(scores.get(k, 50)) * w for k, w in INTERVIEW_SCORE_WEIGHTS.items()))


def score_to_band(score_100: int) -> float:
    if score_100 >= 92: return 9.0
    if score_100 >= 87: return 8.5
    if score_100 >= 82: return 8.0
    if score_100 >= 77: return 7.5
    if score_100 >= 72: return 7.0
    if score_100 >= 65: return 6.5
    if score_100 >= 58: return 6.0
    if score_100 >= 52: return 5.5
    if score_100 >= 45: return 5.0
    if score_100 >= 38: return 4.5
    return 4.0


def round_to_nearest_half_band(value: float) -> float:
    return round(value * 2) / 2


def compute_practice_band(criteria: dict) -> tuple[float, dict]:
    per = {
        "fluencyCoherence": score_to_band(int(criteria.get("fluencyCoherence", 50))),
        "lexicalResource": score_to_band(int(criteria.get("lexicalResource", 50))),
        "grammarRangeAccuracy": score_to_band(int(criteria.get("grammarRangeAccuracy", 50))),
        "pronunciation": score_to_band(int(criteria.get("pronunciation", 50))),
    }
    return round_to_nearest_half_band(sum(per.values()) / 4.0), per


def compute_voice_scores(
    features: dict,
    transcript_data: dict,
    transcript_analysis: TranscriptAnalysis,
) -> tuple[VoiceScores, AcousticMetrics]:
    duration = features["duration"]
    word_count = len(transcript_data["text"].split())
    wpm = (word_count / duration * 60) if duration > 0 else 0

    pitch_score = clamp(int(min(100, (features["std_f0"] / 50) * 100)))
    pace_score = clamp(int(100 - abs(wpm - 130) / 1.3)) if wpm > 0 else 50
    filler_rate = transcript_analysis.filler_analysis.count / max(duration / 60, 0.1)
    clarity_score = clamp(int(100 - filler_rate * 15))
    hnr = features["hnr"]
    resonance_score = clamp(int(min(100, (hnr / 20) * 100))) if hnr > 0 else 50
    confidence_score = clamp(int(100 - transcript_analysis.confidence_markers.count * 8))
    overall = clamp(int(
        pitch_score * 0.20 + pace_score * 0.20 + clarity_score * 0.25 +
        resonance_score * 0.15 + confidence_score * 0.20
    ))

    scores = VoiceScores(
        pitch=pitch_score, pace=pace_score, clarity=clarity_score,
        resonance=resonance_score, confidence=confidence_score, overall=overall,
    )
    metrics = AcousticMetrics(
        duration_seconds=round(duration, 2),
        words_per_minute=round(wpm, 1),
        mean_pitch_hz=round(features["mean_f0"], 1),
        pitch_std_hz=round(features["std_f0"], 1),
        hnr_db=round(hnr, 1),
        pause_count=features["pause_count"],
        long_pause_count=features["long_pause_count"],
    )
    return scores, metrics


def pick_archetype(s: VoiceScores) -> str:
    dims = {"pitch": s.pitch, "pace": s.pace, "clarity": s.clarity,
            "resonance": s.resonance, "confidence": s.confidence}
    top = max(dims, key=dims.get)
    if top == "resonance" and s.resonance >= 80: return "The Warm Communicator"
    if top == "clarity" and s.clarity >= 80: return "The Precise Speaker"
    if top == "pitch" and s.pitch >= 80: return "The Natural Storyteller"
    if top == "confidence" and s.confidence >= 80: return "The Grounded Voice"
    if s.clarity >= 75 and s.pace >= 75: return "The Clear Thinker"
    if s.pitch >= 70 and s.pace >= 70: return "The Energetic Presenter"
    return "The Developing Voice"
