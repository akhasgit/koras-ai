"""
Service helpers for the Listening Comprehension programme.

Two responsibilities:
  1. TTS synthesis via OpenAI (`tts-1`), returned as MP3 bytes.
     Provider is behind a single helper so we can swap later.
  2. Claude-graded voice comprehension answer, with a non-fatal
     fallback (partial result beats total failure).
"""
from __future__ import annotations

import json
import re

from config import CLAUDE_MODEL, anthropic_client, openai_client
from models.listening import ListeningVoiceAnalysis, VocabularyNote
from prompts.listening import LISTENING_ANSWER_PROMPT, format_expected_points
from utils.text import clamp, strip_fences

# OpenAI TTS voices we're happy to expose. "alloy" is a neutral,
# natural-sounding default that reads narrative well.
DEFAULT_TTS_VOICE = "alloy"
ALLOWED_TTS_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
TTS_MODEL = "tts-1"


async def synthesize_tts(text: str, voice: str | None, speed: float) -> bytes:
    """
    Turn text into MP3 bytes. Provider-swappable — every other call in
    the codebase should go through this helper, not OpenAI directly.
    """
    picked = voice if voice in ALLOWED_TTS_VOICES else DEFAULT_TTS_VOICE
    speed_clamped = max(0.5, min(1.5, float(speed or 1.0)))

    response = await openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=picked,  # type: ignore[arg-type]
        input=text,
        speed=speed_clamped,
        response_format="mp3",
    )
    # The OpenAI SDK returns an HttpxBinaryResponseContent-like object.
    # `.read()` gives us the raw bytes.
    return await response.aread() if hasattr(response, "aread") else response.read()


# ─── Voice grading ──────────────────────────────────────────────────────────


# Common Singapore English discourse particles. Stripped from the
# transcript BEFORE grading so Claude doesn't waste a token thinking
# about them — belt-and-braces alongside the prompt instructions.
_SG_PARTICLE_RE = re.compile(
    r"\b(lah|lor|leh|meh|hor|sia|ah|liao|hai(y|ah)?)\b",
    re.IGNORECASE,
)


def normalise_transcript_for_grading(transcript: str) -> str:
    """
    Light-touch normalisation. We do NOT rewrite the sentence — we just
    trim SG particles that Whisper transcribed literally, which would
    otherwise nudge Claude toward marking "informal register" that we
    don't care about in a listening test.
    """
    cleaned = _SG_PARTICLE_RE.sub("", transcript or "")
    # Collapse resulting double-spaces.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def listening_answer_fallback(
    transcript: str,
    expected_points: list[str],
) -> ListeningVoiceAnalysis:
    """
    Non-fatal fallback if Claude fails: a partial-but-usable result. We
    make a coarse guess based on how much they said and how many
    expected-point keywords appear in the transcript.
    """
    words = len(transcript.split())
    lower = transcript.lower()
    covered: list[str] = []
    missed: list[str] = []
    for pt in expected_points:
        keywords = [w for w in re.findall(r"[a-z]{4,}", pt.lower())][:3]
        if keywords and any(k in lower for k in keywords):
            covered.append(pt)
        else:
            missed.append(pt)

    hit_ratio = len(covered) / max(1, len(expected_points))
    base = int(round(hit_ratio * 100))
    if words < 15:
        base = min(base, 45)  # too short to demonstrate understanding
    comp = clamp(base)
    rel = clamp(base - 5) if base >= 20 else base

    return ListeningVoiceAnalysis(
        comprehension_score=comp,
        relevance_score=rel,
        captured_meaning=comp >= 60,
        key_points_covered=covered,
        key_points_missed=missed,
        vocabulary=VocabularyNote(range="adequate", notable_words=[]),
        feedback=(
            "We couldn't run the full comprehension grading, so this score is a "
            "rough estimate. Try recording again for a more accurate report."
        ),
    )


async def grade_listening_answer(
    transcript: str,
    question_prompt: str,
    passage_gist: str,
    expected_points: list[str],
    target_skill: str,
) -> ListeningVoiceAnalysis:
    """
    Ask Claude to grade a student's spoken answer for comprehension.
    Falls back to a heuristic if Claude fails or returns garbage.
    """
    normalised = normalise_transcript_for_grading(transcript)
    prompt = LISTENING_ANSWER_PROMPT.format(
        question_prompt=question_prompt[:1000],
        target_skill=target_skill or "main_idea",
        passage_gist=(passage_gist or "")[:2000],
        expected_points_block=format_expected_points(expected_points or []),
        transcript=normalised[:6000],
    )

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = strip_fences("\n".join(text_blocks).strip())
        data = json.loads(raw)
    except Exception:
        return listening_answer_fallback(normalised, expected_points or [])

    # Coerce + clamp everything defensively — Claude is helpful, but we
    # don't trust structural output blindly.
    try:
        vocab_raw = data.get("vocabulary") or {}
        vocab_range = str(vocab_raw.get("range") or "adequate").strip().lower()
        if vocab_range not in ("limited", "adequate", "strong"):
            vocab_range = "adequate"
        notable = [
            str(w).strip()
            for w in (vocab_raw.get("notable_words") or [])
            if isinstance(w, (str, int, float)) and str(w).strip()
        ][:8]

        return ListeningVoiceAnalysis(
            comprehension_score=clamp(int(data.get("comprehension_score", 50))),
            relevance_score=clamp(int(data.get("relevance_score", 50))),
            captured_meaning=bool(data.get("captured_meaning", False)),
            key_points_covered=[
                str(x).strip()
                for x in (data.get("key_points_covered") or [])
                if isinstance(x, (str, int, float)) and str(x).strip()
            ],
            key_points_missed=[
                str(x).strip()
                for x in (data.get("key_points_missed") or [])
                if isinstance(x, (str, int, float)) and str(x).strip()
            ],
            vocabulary=VocabularyNote(
                range=vocab_range,  # type: ignore[arg-type]
                notable_words=notable,
            ),
            feedback=str(data.get("feedback") or "").strip()
            or "Your answer was scored.",
        )
    except Exception:
        return listening_answer_fallback(normalised, expected_points or [])
