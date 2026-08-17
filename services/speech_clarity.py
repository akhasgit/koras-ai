"""Speech Clarity generation, scoring, and drill selection."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from config import CLAUDE_MODEL, anthropic_client
from models.speech_clarity import (
    AnalyzeClarityReadResponse,
    ClarityScores,
    GenerateClarityDrillsRequest,
    GenerateClarityDrillsResponse,
    GenerateClarityPassageRequest,
    GenerateClarityPassageResponse,
    ScoreClarityDrillResponse,
)
from prompts.speech_clarity import (
    CLARITY_DRILL_COPY_PROMPT,
    CLARITY_PASSAGE_PROMPT,
    CLARITY_READ_PROMPT,
)
from services.phone_client import align_phones
from utils.text import clamp, count_fillers, strip_fences

GENERATOR_VERSION = "v1"

FALLBACK_PASSAGES = [
    "I was not expecting that at all. Could you say it again, a little more slowly? I think I missed the last part. When I hear it once more, I can follow the idea. Then I will try to say it back in my own words. Does that sound fair to you?",
    "Every morning I walk to the corner shop for bread and milk. The street is quiet before the buses start. I like to take my time and look around. Have you ever noticed how the light changes at that hour? I always feel ready for the day after that short walk.",
    "I tried to finish the work before dinner, but I needed a short break. Could we talk about it after I eat? I want to give you a clear answer, not a rushed one. If we wait a little, I think we will both feel better about the plan.",
]

DEFAULT_FEATURES = [
    "rhythm_vowel_reduction",
    "word_stress",
    "sentence_prominence",
    "intonation_terminal",
    "pace",
    "pausing",
]

DRILL_BY_CLASS = {
    "prosodic_rhythm": ["shadowing", "chunking_linking", "targeted_passage"],
    "prosodic_stress": ["stress_tapping", "shadowing"],
    "prosodic_intonation": ["intonation_contours", "shadowing"],
    "segmental": ["minimal_pair_discrimination", "minimal_pair_production"],
    "decoding": ["shadowing"],
    "pace": ["chunking_linking"],
}

STATIC_DRILLS = {
    "shadowing": {
        "id": "shadow-1",
        "drill_type": "shadowing",
        "feature_id": "rhythm_vowel_reduction",
        "title": "Shadow the line",
        "instruction": "Hear the line, then say it back immediately. Keep the same rhythm.",
        "items": [
            {"id": "i1", "text": "I was not expecting that at all.", "ipa": ""},
            {"id": "i2", "text": "Could you say it again, a little more slowly?", "ipa": ""},
        ],
    },
    "chunking_linking": {
        "id": "chunk-1",
        "drill_type": "chunking_linking",
        "feature_id": "linking",
        "title": "Link the words",
        "instruction": "Read the line, joining the marked words. Pause only at the bars.",
        "items": [
            {"id": "i1", "text": "I want_to finish_it before dinner. | Then we can talk."},
        ],
    },
    "stress_tapping": {
        "id": "stress-1",
        "drill_type": "stress_tapping",
        "feature_id": "word_stress",
        "title": "Tap the stress",
        "instruction": "Tap the stressed syllable, then say the word.",
        "items": [
            {"id": "i1", "text": "tomorrow", "choices": ["to", "MOR", "row"], "correct": "MOR"},
            {"id": "i2", "text": "important", "choices": ["im", "POR", "tant"], "correct": "POR"},
        ],
    },
    "minimal_pair_discrimination": {
        "id": "disc-1",
        "drill_type": "minimal_pair_discrimination",
        "feature_id": "v_w_contrast",
        "title": "Which word did you hear?",
        "instruction": "Listen once, then tap the word you heard.",
        "items": [
            {"id": "i1", "text": "vine", "choices": ["vine", "wine"], "correct": "vine"},
            {"id": "i2", "text": "vest", "choices": ["vest", "west"], "correct": "west"},
        ],
    },
    "minimal_pair_production": {
        "id": "prod-1",
        "drill_type": "minimal_pair_production",
        "feature_id": "v_w_contrast",
        "title": "Say both words",
        "instruction": "Say each pair clearly: first word, then second.",
        "items": [
            {"id": "i1", "text": "vine / wine"},
            {"id": "i2", "text": "vest / west"},
        ],
    },
    "intonation_contours": {
        "id": "int-1",
        "drill_type": "intonation_contours",
        "feature_id": "intonation_terminal",
        "title": "Change the tune",
        "instruction": "Say the same words as a statement, then as a question.",
        "items": [
            {"id": "i1", "text": "You are coming with us."},
            {"id": "i2", "text": "You are coming with us?"},
        ],
    },
    "targeted_passage": {
        "id": "targ-1",
        "drill_type": "targeted_passage",
        "feature_id": "rhythm_vowel_reduction",
        "title": "One short line",
        "instruction": "Read this line, reducing the small words.",
        "items": [
            {"id": "i1", "text": "I want to get to the shop before it closes."},
        ],
    },
    "transfer_test": {
        "id": "xfer-1",
        "drill_type": "transfer_test",
        "feature_id": "sentence_prominence",
        "title": "Speak freely",
        "instruction": "For about a minute, tell us about a time you had to explain something clearly.",
        "items": [{"id": "i1", "text": "Tell us about a time you had to explain something clearly."}],
    },
}


def passage_seed(
    user_id: str,
    date: str,
    target_accent_id: str,
    feature_ids: list[str],
    generator_version: str = GENERATOR_VERSION,
) -> str:
    payload = f"{user_id}|{date}|{target_accent_id}|{','.join(sorted(feature_ids))}|{generator_version}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _fallback_passage(seed: str, features: list[str]) -> GenerateClarityPassageResponse:
    idx = int(seed[:8], 16) % len(FALLBACK_PASSAGES)
    seeded = (features or DEFAULT_FEATURES)[:6]
    return GenerateClarityPassageResponse(
        passage_text=FALLBACK_PASSAGES[idx],
        seeded_feature_ids=seeded,
        fallback=True,
    )


async def generate_clarity_passage(req: GenerateClarityPassageRequest) -> GenerateClarityPassageResponse:
    features = req.active_feature_ids or DEFAULT_FEATURES
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": CLARITY_PASSAGE_PROMPT.format(
                    grade_band=req.grade_band or "B1",
                    features_json=json.dumps(features[:6]),
                    passage_seed=req.passage_seed,
                ),
            }],
        )
        raw = strip_fences(msg.content[0].text if msg.content else "")
        data = json.loads(raw)
        text = (data.get("passage_text") or "").strip()
        words = len(re.findall(r"[A-Za-z']+", text))
        if words < 40 or words > 90:
            return _fallback_passage(req.passage_seed, features)
        seeded = data.get("seeded_feature_ids") or features[:6]
        return GenerateClarityPassageResponse(
            passage_text=text,
            seeded_feature_ids=list(seeded)[:6],
            fallback=False,
        )
    except Exception:
        return _fallback_passage(req.passage_seed, features)


def _token_overlap(passage: str, transcript: str) -> float:
    def toks(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[a-zA-Z']+", s or "") if len(t) > 2}
    a, b = toks(passage), toks(transcript)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _fluency_score(transcript: str, features: dict, duration: float) -> tuple[int, float, int]:
    words = len((transcript or "").split())
    wpm = (words / duration * 60) if duration > 0 else 0.0
    fillers = count_fillers(transcript)
    filler_rate = fillers / max(duration / 60, 0.1) if duration > 0 else 0.0
    pace = clamp(int(100 - abs(wpm - 130) / 1.3)) if wpm > 0 else 50
    fluency = clamp(int(pace - filler_rate * 8 - max(0, features.get("long_pause_count", 0) - 2) * 4))
    return fluency, wpm, fillers


def _intonation_score(features: dict) -> int:
    std_f0 = float(features.get("std_f0") or 0.0)
    return clamp(int(min(100, (std_f0 / 40) * 100)))


def _weighted_overall(scores: ClarityScores) -> int:
    parts: list[tuple[int, float]] = []
    if scores.intelligibility is not None:
        parts.append((scores.intelligibility, 0.35))
    if scores.rhythm is not None:
        parts.append((scores.rhythm, 0.20))
    if scores.word_stress is not None:
        parts.append((scores.word_stress, 0.15))
    if scores.intonation is not None:
        parts.append((scores.intonation, 0.10))
    parts.append((scores.fluency, 0.20))
    # accent_alignment is 0% under Accent 5
    if not parts:
        return 50
    weight_sum = sum(w for _, w in parts)
    return clamp(int(sum(v * w for v, w in parts) / weight_sum))


def _default_error_map(fluency: int, intonation: int, wpm: float) -> dict[str, Any]:
    features = []
    dominant = "prosodic_rhythm"
    if wpm > 160:
        dominant = "pace"
        features.append({
            "feature_id": "pace",
            "feature_class": "prosodic",
            "note": "The read was quite fast — slowing a little will help listeners catch each word.",
            "severity": 2,
        })
    if intonation < 55:
        features.append({
            "feature_id": "intonation_terminal",
            "feature_class": "prosodic",
            "note": "The pitch stayed quite flat. Try a clearer rise or fall at the end of a sentence.",
            "severity": 2,
        })
        if dominant != "pace":
            dominant = "prosodic_intonation"
    features.append({
        "feature_id": "rhythm_vowel_reduction",
        "feature_class": "prosodic",
        "note": "Focus on shortening the small words so the important ones stand out.",
        "severity": 1,
    })
    if fluency < 60 and dominant != "pace":
        dominant = "prosodic_rhythm"
    return {"dominant_class": dominant, "features": features[:6]}


async def analyze_clarity_read(
    wav_bytes: bytes,
    transcript_data: dict,
    features: dict,
    passage: str,
    active_features: list[str],
) -> AnalyzeClarityReadResponse:
    transcript = (transcript_data.get("text") or "").strip()
    duration = float(features.get("duration") or transcript_data.get("duration") or 0.0)
    fluency, wpm, fillers = _fluency_score(transcript, features, duration)
    intonation = _intonation_score(features)
    overlap = _token_overlap(passage, transcript)
    off_script = overlap < 0.45

    phone = await align_phones(wav_bytes, passage)
    phone_available = bool(phone.get("available"))
    intelligibility = phone.get("intelligibility_score") if phone_available else None
    rhythm = phone.get("rhythm_score") if phone_available else None
    word_stress = phone.get("word_stress_score") if phone_available else None

    scores = ClarityScores(
        intelligibility=intelligibility if isinstance(intelligibility, int) else None,
        rhythm=rhythm if isinstance(rhythm, int) else None,
        word_stress=word_stress if isinstance(word_stress, int) else None,
        intonation=intonation,
        fluency=fluency,
        accent_alignment=None,
        overall=50,
    )
    scores.overall = _weighted_overall(scores)

    segments = transcript_data.get("segments") or []
    avg_logprob = None
    if segments:
        vals = [float(s.get("avg_logprob") or 0.0) for s in segments]
        avg_logprob = sum(vals) / len(vals)

    acoustic = {
        "duration_seconds": round(duration, 2),
        "words_per_minute": round(wpm, 1),
        "pause_count": features.get("pause_count"),
        "long_pause_count": features.get("long_pause_count"),
        "mean_f0": features.get("mean_f0"),
        "std_f0": features.get("std_f0"),
        "hnr": features.get("hnr"),
        "filler_count": fillers,
        "avg_logprob": avg_logprob,
        "alignment_coverage": round(overlap, 3),
        "phone_reason": phone.get("reason"),
    }

    error_map = _default_error_map(fluency, intonation, wpm)
    feature_findings: list[dict[str, Any]] = []
    feedback = {"coach_note": "Nice work on this read. Next, slow the small words and let the important ones land."}

    if not off_script:
        try:
            msg = await anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=900,
                messages=[{
                    "role": "user",
                    "content": CLARITY_READ_PROMPT.format(
                        passage=passage[:2000],
                        transcript=transcript[:2000],
                        features_json=json.dumps(active_features[:8]),
                        wpm=round(wpm, 1),
                        pause_count=features.get("pause_count"),
                        long_pause_count=features.get("long_pause_count"),
                        mean_f0=round(float(features.get("mean_f0") or 0.0), 1),
                        phone_available=phone_available,
                    ),
                }],
            )
            data = json.loads(strip_fences(msg.content[0].text if msg.content else ""))
            if isinstance(data.get("error_map"), dict):
                error_map = data["error_map"]
            if isinstance(data.get("feature_findings"), list):
                feature_findings = data["feature_findings"]
            if data.get("coach_note"):
                feedback = {"coach_note": data["coach_note"]}
            if data.get("off_script") is True:
                off_script = True
        except Exception:
            pass

    if off_script:
        feedback = {
            "coach_note": "It sounded like a different text from the passage. Fluency is scored; word-level highlights are skipped.",
        }

    return AnalyzeClarityReadResponse(
        transcript=transcript,
        scores=scores,
        acoustic_metrics=acoustic,
        feature_findings=feature_findings,
        error_map=error_map,
        feedback=feedback,
        phone_sequence=phone.get("phone_sequence"),
        reference_phones=phone.get("reference_phones"),
        phone_available=phone_available,
        off_script=off_script,
    )


async def score_clarity_drill(
    wav_bytes: bytes,
    transcript_data: dict,
    features: dict,
    prompt_text: str,
) -> ScoreClarityDrillResponse:
    transcript = (transcript_data.get("text") or "").strip()
    duration = float(features.get("duration") or 0.0)
    fluency, wpm, _fillers = _fluency_score(transcript, features, duration)
    intonation = _intonation_score(features)
    phone = await align_phones(wav_bytes, prompt_text)
    phone_available = bool(phone.get("available"))
    scores = ClarityScores(
        intelligibility=phone.get("intelligibility_score") if phone_available else None,
        rhythm=phone.get("rhythm_score") if phone_available else None,
        word_stress=phone.get("word_stress_score") if phone_available else None,
        intonation=intonation,
        fluency=fluency,
        accent_alignment=None,
        overall=50,
    )
    scores.overall = _weighted_overall(scores)
    matched = _token_overlap(prompt_text, transcript) >= 0.4
    return ScoreClarityDrillResponse(
        scores=scores,
        acoustic_metrics={
            "duration_seconds": round(duration, 2),
            "words_per_minute": round(wpm, 1),
            "hnr": features.get("hnr"),
        },
        drill_result={"matched": matched, "transcript": transcript},
        phone_available=phone_available,
    )


def select_drill_types(error_map: dict, include_transfer: bool) -> list[str]:
    dominant = (error_map or {}).get("dominant_class") or "prosodic_rhythm"
    pool = list(DRILL_BY_CLASS.get(dominant, DRILL_BY_CLASS["prosodic_rhythm"]))
    chosen = pool[:2]
    if include_transfer:
        chosen.append("transfer_test")
    # unique, max 3
    out: list[str] = []
    for t in chosen:
        if t not in out:
            out.append(t)
    return out[:3]


async def generate_clarity_drills(req: GenerateClarityDrillsRequest) -> GenerateClarityDrillsResponse:
    types = select_drill_types(req.error_map, req.include_transfer)
    drills = []
    for t in types:
        base = json.loads(json.dumps(STATIC_DRILLS.get(t, STATIC_DRILLS["shadowing"])))
        drills.append(base)
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": CLARITY_DRILL_COPY_PROMPT.format(
                    dominant_class=(req.error_map or {}).get("dominant_class") or "prosodic_rhythm",
                    drill_types_json=json.dumps(types),
                    features_json=json.dumps(req.active_features[:8]),
                ),
            }],
        )
        data = json.loads(strip_fences(msg.content[0].text if msg.content else ""))
        if isinstance(data.get("drills"), list) and data["drills"]:
            return GenerateClarityDrillsResponse(drills=data["drills"][:3])
    except Exception:
        pass
    return GenerateClarityDrillsResponse(drills=drills)
