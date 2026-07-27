"""
Word generation service for the Daily Vocabulary feature.

Wraps the Anthropic Claude call with:
  * a strict-JSON prompt (from prompts/vocabulary.py)
  * `strip_fences` cleanup
  * pydantic validation against `WordObject`
  * a curated fallback list when Claude returns malformed JSON, an
    error, or fewer than `count` words.

Mirrors the shape of `services/daily_plan.py` — build a deterministic
fallback first, attempt the Claude call, return the validated Claude
output if it succeeds, otherwise return the fallback.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Iterable

from config import CLAUDE_MODEL, anthropic_client
from models.analyze import VoiceScores
from models.vocabulary import (
    AnalyzePronunciationResponse,
    AnalyzeVocabularySentenceResponse,
    AnalyzeVocabularySpeechResponse,
    GenerateVocabularyRequest,
    GenerateVocabularyResponse,
    VoiceSubScores,
    WordDetected,
    WordObject,
)
from prompts.vocabulary import (
    VOCABULARY_GENERATE_PROMPT,
    VOCABULARY_SENTENCE_PROMPT,
    VOCABULARY_SPEECH_PROMPT,
)
from services.scoring import compute_voice_scores, pick_archetype
from utils.text import clamp, strip_fences

VALID_REGISTERS = {"formal", "neutral", "informal", "academic", "business"}
VALID_CEFR = {"A1", "A2", "B1", "B2", "C1", "C2"}
VALID_POS = {
    "noun", "verb", "adjective", "adverb", "preposition",
    "conjunction", "interjection", "pronoun", "determiner",
}


# A small curated pool, deterministically filtered by the learner's CEFR
# band and the exclude list. Only used when Claude fails entirely.
_FALLBACK_POOL: list[WordObject] = [
    WordObject(
        word="eloquent", ipa="/ˈɛləkwənt/", part_of_speech="adjective",
        definition="able to speak clearly and persuasively.",
        example_sentence="She gave an eloquent answer in the interview.",
        register="neutral", difficulty="B2",
        why_chosen="A reliable word to describe confident, well-structured speech — useful in interviews and presentations.",
    ),
    WordObject(
        word="resilient", ipa="/rɪˈzɪliənt/", part_of_speech="adjective",
        definition="able to recover quickly from difficulty.",
        example_sentence="The team stayed resilient after the missed deadline.",
        register="neutral", difficulty="B2",
        why_chosen="A common 'soft-skill' word that lands well in workplace and study settings.",
    ),
    WordObject(
        word="concise", ipa="/kənˈsaɪs/", part_of_speech="adjective",
        definition="short and clear, without extra words.",
        example_sentence="Keep your update concise — three sentences is enough.",
        register="neutral", difficulty="B1",
        why_chosen="Anchors the habit of cutting filler from your sentences.",
    ),
    WordObject(
        word="acknowledge", ipa="/əkˈnɒlɪdʒ/", part_of_speech="verb",
        definition="to accept or admit that something is true.",
        example_sentence="I want to acknowledge the work the team did this week.",
        register="neutral", difficulty="B1",
        why_chosen="A practical verb for handling feedback or recognising contributions in conversation.",
    ),
    WordObject(
        word="nuance", ipa="/ˈnjuːɑːns/", part_of_speech="noun",
        definition="a small but important difference in meaning or tone.",
        example_sentence="There's a nuance in her answer that we shouldn't miss.",
        register="academic", difficulty="C1",
        why_chosen="Lets you describe subtle differences — useful when discussing arguments or texts.",
    ),
    WordObject(
        word="rigorous", ipa="/ˈrɪɡərəs/", part_of_speech="adjective",
        definition="thorough and careful.",
        example_sentence="They followed a rigorous testing process.",
        register="academic", difficulty="B2",
        why_chosen="A useful word in academic and professional writing or speech.",
    ),
    WordObject(
        word="ambition", ipa="/æmˈbɪʃən/", part_of_speech="noun",
        definition="a strong wish to achieve something.",
        example_sentence="Her ambition is to lead a research team one day.",
        register="neutral", difficulty="B1",
        why_chosen="A natural way to talk about goals — common in interviews.",
    ),
    WordObject(
        word="empathy", ipa="/ˈɛmpəθi/", part_of_speech="noun",
        definition="the ability to understand how another person feels.",
        example_sentence="Good managers lead with empathy.",
        register="neutral", difficulty="B2",
        why_chosen="A widely-respected word for talking about teamwork and leadership.",
    ),
    WordObject(
        word="straightforward", ipa="/ˌstreɪtˈfɔːwəd/", part_of_speech="adjective",
        definition="easy to understand or do; not complicated.",
        example_sentence="The setup is straightforward — three steps in total.",
        register="neutral", difficulty="B1",
        why_chosen="A confident alternative to 'easy' that sounds polished without being formal.",
    ),
    WordObject(
        word="diligent", ipa="/ˈdɪlɪdʒənt/", part_of_speech="adjective",
        definition="working hard and being careful with detail.",
        example_sentence="He is diligent about checking his work twice.",
        register="formal", difficulty="B2",
        why_chosen="A respected word employers and teachers respond to.",
    ),
]


def _cefr_lower_or_equal(target: str) -> set[str]:
    """Return the set of CEFR bands at or below the target."""
    order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    if target not in order:
        return set(order)
    idx = order.index(target)
    return set(order[: idx + 1])


def build_fallback(req: GenerateVocabularyRequest) -> GenerateVocabularyResponse:
    """
    Deterministic, level-aware fallback when Claude fails entirely.

    Filters the curated pool to words at or below the learner's CEFR
    target, excludes already-learned words, and returns the first
    `req.count` matches. If we still come up short we top up with the
    earliest remaining pool entries — ensuring the contract
    "exactly count words" is honoured.
    """
    exclude = {w.lower() for w in (req.exclude_words or [])}
    allowed_bands = _cefr_lower_or_equal(req.cefr_target)

    chosen: list[WordObject] = []
    leftovers: list[WordObject] = []
    for w in _FALLBACK_POOL:
        if w.word.lower() in exclude:
            continue
        if w.difficulty in allowed_bands:
            chosen.append(w)
        else:
            leftovers.append(w)

    out: list[WordObject] = chosen[: req.count]
    if len(out) < req.count:
        for w in leftovers:
            if len(out) >= req.count:
                break
            if w.word.lower() not in exclude:
                out.append(w)
    # Last-resort top-up: just take from the pool ignoring exclude, so we
    # always meet the contract count rather than returning fewer words.
    if len(out) < req.count:
        for w in _FALLBACK_POOL:
            if len(out) >= req.count:
                break
            if w not in out:
                out.append(w)

    return GenerateVocabularyResponse(words=out[: req.count])


def _normalise_word_dict(raw: dict) -> WordObject | None:
    """Coerce one raw Claude word entry into a validated WordObject."""
    try:
        word = str(raw.get("word") or "").strip()
        if not word:
            return None
        ipa = str(raw.get("ipa") or "").strip()
        pos = str(raw.get("part_of_speech") or "").strip().lower()
        if pos not in VALID_POS:
            pos = "noun"  # safe default rather than rejecting
        definition = str(raw.get("definition") or "").strip()
        if not definition:
            return None
        example = str(raw.get("example_sentence") or "").strip()
        if not example:
            return None
        register = str(raw.get("register") or "neutral").strip().lower()
        if register not in VALID_REGISTERS:
            register = "neutral"
        difficulty = str(raw.get("difficulty") or "").strip().upper()
        if difficulty not in VALID_CEFR:
            difficulty = "B1"
        why_chosen = str(raw.get("why_chosen") or "").strip()
        if not why_chosen:
            why_chosen = "A useful word at your current level."
        return WordObject(
            word=word, ipa=ipa, part_of_speech=pos,
            definition=definition, example_sentence=example,
            register=register, difficulty=difficulty, why_chosen=why_chosen,
        )
    except Exception:
        return None


def _dedupe_words(words: Iterable[WordObject]) -> list[WordObject]:
    seen: set[str] = set()
    out: list[WordObject] = []
    for w in words:
        key = w.word.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


async def generate_vocabulary_words(
    req: GenerateVocabularyRequest,
) -> GenerateVocabularyResponse:
    """
    Call Claude to generate `req.count` vocabulary words, validated and
    safe to persist. Falls back to a curated pool on any error or when
    the model returns fewer than `req.count` usable items.
    """
    fallback = build_fallback(req)

    try:
        prompt = VOCABULARY_GENERATE_PROMPT.format(
            count=req.count,
            segment=req.segment,
            grade_level=req.grade_level if req.grade_level is not None else "null",
            cefr_target=req.cefr_target,
            profession=req.profession or "null",
            goals_json=json.dumps(req.goals or [], ensure_ascii=False),
            weak_areas_json=json.dumps(req.weak_areas or [], ensure_ascii=False),
            exclude_words_json=json.dumps((req.exclude_words or [])[:80], ensure_ascii=False),
        )
    except Exception:
        return fallback

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        cleaned = strip_fences("\n".join(text_blocks).strip())
        data = json.loads(cleaned)
    except Exception:
        return fallback

    raw_words = data.get("words") if isinstance(data, dict) else None
    if not isinstance(raw_words, list):
        return fallback

    exclude_lc = {w.lower() for w in (req.exclude_words or [])}
    normalised: list[WordObject] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        word = _normalise_word_dict(raw)
        if word is None:
            continue
        if word.word.lower() in exclude_lc:
            continue
        normalised.append(word)

    normalised = _dedupe_words(normalised)
    if len(normalised) < req.count:
        # Top up with fallback entries (still respecting exclude_lc).
        for fw in fallback.words:
            if len(normalised) >= req.count:
                break
            if fw.word.lower() in exclude_lc:
                continue
            if any(fw.word.lower() == n.word.lower() for n in normalised):
                continue
            normalised.append(fw)

    return GenerateVocabularyResponse(words=normalised[: req.count])


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────


_WORD_TOKEN = re.compile(r"[a-zA-Z']+")


def _normalise_word(w: str) -> str:
    """Lower, strip punctuation — cheap normalisation for fuzzy match."""
    return "".join(ch for ch in (w or "").lower() if ch.isalpha() or ch == "'").strip("'")


def _tokens(text: str) -> list[str]:
    return [_normalise_word(t) for t in _WORD_TOKEN.findall(text or "")]


def _fuzzy_similarity(a: str, b: str) -> float:
    a, b = _normalise_word(a), _normalise_word(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _stem_variants(word: str) -> set[str]:
    """
    Cheap English inflection expansion — enough that "eloquent" matches
    "eloquently" and "acknowledge" matches "acknowledged" without
    dragging in a real stemmer. Not linguistically rigorous; deliberate.
    """
    w = _normalise_word(word)
    if not w:
        return set()
    variants = {w}
    for suffix in ("s", "es", "ed", "d", "ing", "er", "est", "ly", "ies"):
        variants.add(f"{w}{suffix}")
    # Common -y → -ies / -ed forms.
    if w.endswith("y"):
        variants.add(f"{w[:-1]}ies")
        variants.add(f"{w[:-1]}ied")
    if w.endswith("e"):
        variants.add(f"{w[:-1]}ing")
        variants.add(f"{w}d")
    return variants


def _hnr_to_clarity(hnr_db: float) -> int:
    """Map HNR (dB, typically 5–25 for speech) to a 0-100 clarity score."""
    if hnr_db <= 0:
        return 40
    return clamp(int(min(100, (hnr_db / 20) * 100)))


def _fuzzy_word_present(target: str, tokens: list[str]) -> tuple[bool, float]:
    """
    Return (present, best_similarity). "Present" if any token matches the
    target's simple stem variants exactly, OR if the best fuzzy ratio is
    ≥ 0.82 — the threshold Whisper mishears usually fall under.
    """
    if not target or not tokens:
        return False, 0.0
    variants = _stem_variants(target)
    for t in tokens:
        if t in variants:
            return True, 1.0
    best = max((_fuzzy_similarity(target, t) for t in tokens), default=0.0)
    return best >= 0.82, best


def _pronunciation_feedback(matched: bool, similarity: float, clarity: int) -> str:
    if matched and similarity >= 0.95 and clarity >= 75:
        return "Clean, well-shaped pronunciation — hold that shape."
    if matched and clarity >= 65:
        return "Recognised — nice work. Try opening the vowel a touch more for a crisper finish."
    if matched:
        return "We caught the word. Try again with a little more breath and clearer consonants."
    if similarity >= 0.6:
        return "Very close — we heard a similar shape. Slow down slightly and emphasise each syllable."
    return "We didn't catch the target word — try recording again, unhurried."


# ─── Pronounce ──────────────────────────────────────────────────────────────


def analyze_pronunciation_sync(
    target_word: str,
    transcript_data: dict,
    features: dict,
) -> AnalyzePronunciationResponse:
    """
    Synchronous scoring — no Claude, no I/O beyond what the caller
    already fetched. Wrapped in an async controller.
    """
    transcript_text = (transcript_data.get("text") or "").strip()
    tokens = _tokens(transcript_text)
    matched, similarity = _fuzzy_word_present(target_word, tokens)

    clarity = _hnr_to_clarity(float(features.get("hnr") or 0.0))
    # Score blends fuzzy match (dominant when transcript is right) with
    # acoustic clarity (dominant when the learner nailed the shape but
    # Whisper confused an accent). 70/30 weighting.
    pron_score = clamp(int(similarity * 70 + clarity * 0.3))

    return AnalyzePronunciationResponse(
        target_word=target_word,
        transcript=transcript_text,
        matched=matched,
        similarity=round(similarity, 3),
        pronunciation_score=float(pron_score),
        clarity=float(clarity),
        feedback=_pronunciation_feedback(matched, similarity, clarity),
    )


# ─── Sentence ───────────────────────────────────────────────────────────────


def _compute_sentence_subscores(transcript_text: str, features: dict) -> VoiceSubScores:
    duration = float(features.get("duration") or 0.0)
    word_count = len(transcript_text.split())
    wpm = (word_count / duration * 60) if duration > 0 else 0.0

    # Import lazily to avoid a top-level circular with utils.text at test time.
    from utils.text import count_fillers

    filler_count = count_fillers(transcript_text)
    filler_rate = filler_count / max(duration / 60, 0.1) if duration > 0 else 0.0

    clarity = _hnr_to_clarity(float(features.get("hnr") or 0.0))
    clarity = clamp(int(clarity - filler_rate * 6))

    pace = clamp(int(100 - abs(wpm - 130) / 1.3)) if wpm > 0 else 50
    # Confidence heuristic: fewer fillers + reasonable pitch spread = confident.
    std_f0 = float(features.get("std_f0") or 0.0)
    confidence = clamp(int(80 - filler_count * 5 + min(20, std_f0 / 5)))

    return VoiceSubScores(clarity=clarity, pace=pace, confidence=confidence)


def _sentence_fallback(
    target_word: str,
    transcript_text: str,
    features: dict,
) -> AnalyzeVocabularySentenceResponse:
    tokens = _tokens(transcript_text)
    present, _ = _fuzzy_word_present(target_word, tokens)
    scores = _compute_sentence_subscores(transcript_text, features)
    return AnalyzeVocabularySentenceResponse(
        target_word=target_word,
        transcript=transcript_text,
        word_present=present,
        used_correctly=present,   # optimistic: we couldn't verify, so give the benefit of the doubt
        usage_feedback=(
            "Nice attempt — we heard the word."
            if present
            else "We didn't catch the target word this time — try again."
        ),
        grammar_notes="",
        scores=scores,
        coach_feedback="Recording captured — try another take to hear how it improves.",
    )


async def analyze_vocabulary_sentence(
    target_word: str,
    part_of_speech: str,
    transcript_data: dict,
    features: dict,
) -> AnalyzeVocabularySentenceResponse:
    transcript_text = (transcript_data.get("text") or "").strip()
    fallback = _sentence_fallback(target_word, transcript_text, features)

    if not transcript_text:
        return fallback

    duration = float(transcript_data.get("duration") or features.get("duration") or 0.0)
    try:
        prompt = VOCABULARY_SENTENCE_PROMPT.format(
            target_word=target_word,
            part_of_speech=part_of_speech or "unspecified",
            transcript=transcript_text[:2000],
            duration_seconds=round(duration, 1),
        )
    except Exception:
        return fallback

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        cleaned = strip_fences("\n".join(text_blocks).strip())
        data = json.loads(cleaned)
    except Exception:
        return fallback

    if not isinstance(data, dict):
        return fallback

    scores = _compute_sentence_subscores(transcript_text, features)
    return AnalyzeVocabularySentenceResponse(
        target_word=target_word,
        transcript=transcript_text,
        word_present=bool(data.get("word_present", fallback.word_present)),
        used_correctly=bool(data.get("used_correctly", fallback.used_correctly)),
        usage_feedback=(str(data.get("usage_feedback") or fallback.usage_feedback)).strip()[:400],
        grammar_notes=(str(data.get("grammar_notes") or "")).strip()[:280],
        scores=scores,
        coach_feedback=(
            "Good take — hold onto that confidence when you try again."
            if bool(data.get("used_correctly", False))
            else "Not quite there yet — read the usage note and give it another go."
        ),
    )


# ─── Speech (end-of-day) ────────────────────────────────────────────────────


def _speech_fallback(
    words: list[str],
    transcript_text: str,
    scores: VoiceScores,
    archetype: str,
) -> AnalyzeVocabularySpeechResponse:
    tokens = _tokens(transcript_text)
    detected: list[WordDetected] = []
    for w in words:
        present, sim = _fuzzy_word_present(w, tokens)
        detected.append(WordDetected(
            word=w,
            used=present,
            # If we only fuzzy-matched (not exact), we can't be sure it
            # was used correctly — leave `correct=False` to be honest.
            correct=present and sim >= 0.95,
            note=(
                "Used." if present and sim >= 0.95
                else "Roughly matched — try emphasising this one next time." if present
                else "Not detected in your speech."
            ),
        ))
    used_count = sum(1 for d in detected if d.used)
    return AnalyzeVocabularySpeechResponse(
        transcript=transcript_text,
        words_detected=detected,
        scores=scores,
        coach_feedback=(
            f"You used {used_count} of {len(words)} target words. "
            "Try weaving the rest into your next take."
        ),
        archetype=archetype,
    )


async def analyze_vocabulary_speech(
    words: list[str],
    transcript_data: dict,
    features: dict,
) -> AnalyzeVocabularySpeechResponse:
    from models.analyze import ConfidenceMarkers, FillerAnalysis, TranscriptAnalysis
    from utils.text import count_fillers

    transcript_text = (transcript_data.get("text") or "").strip()

    # Compute the acoustic-only scores up front so we always have a
    # valid `VoiceScores` to hand back, even if Claude fails.
    filler_count = count_fillers(transcript_text)
    stub_transcript_analysis = TranscriptAnalysis(
        topic_summary="",
        filler_analysis=FillerAnalysis(
            count=filler_count,
            rate_per_minute=0.0,
            fillers_used=[],
            worst_sentences=[],
        ),
        phrasing_rewrites=[],
        clarity_issues=[],
        confidence_markers=ConfidenceMarkers(hedging_words=[], count=0),
        suggested_revision="",
    )
    scores, _metrics = compute_voice_scores(features, transcript_data, stub_transcript_analysis)
    archetype = pick_archetype(scores)

    fallback = _speech_fallback(list(words), transcript_text, scores, archetype)

    if not transcript_text or not words:
        return fallback

    duration = float(transcript_data.get("duration") or features.get("duration") or 0.0)
    try:
        prompt = VOCABULARY_SPEECH_PROMPT.format(
            words_json=json.dumps(list(words), ensure_ascii=False),
            transcript=transcript_text[:4000],
            duration_seconds=round(duration, 1),
        )
    except Exception:
        return fallback

    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        cleaned = strip_fences("\n".join(text_blocks).strip())
        data = json.loads(cleaned)
    except Exception:
        return fallback

    raw_detected = data.get("words_detected") if isinstance(data, dict) else None
    if not isinstance(raw_detected, list):
        return fallback

    # Rebuild `words_detected` in the *input order* so callers can map
    # positions 1..5 directly. Missing entries fall back to detection.
    by_word: dict[str, dict] = {}
    for raw in raw_detected:
        if not isinstance(raw, dict):
            continue
        key = _normalise_word(str(raw.get("word") or ""))
        if key:
            by_word[key] = raw

    tokens = _tokens(transcript_text)
    detected: list[WordDetected] = []
    for w in words:
        key = _normalise_word(w)
        raw = by_word.get(key)
        if raw is not None:
            detected.append(WordDetected(
                word=w,
                used=bool(raw.get("used", False)),
                correct=bool(raw.get("correct", False)),
                note=(str(raw.get("note") or "")).strip()[:220],
            ))
            continue
        present, sim = _fuzzy_word_present(w, tokens)
        detected.append(WordDetected(
            word=w,
            used=present,
            correct=present and sim >= 0.95,
            note=(
                "Used." if present and sim >= 0.95
                else "Roughly matched — try emphasising this one next time." if present
                else "Not detected in your speech."
            ),
        ))

    coach_feedback = (str(data.get("coach_feedback") or fallback.coach_feedback)).strip()[:600]
    return AnalyzeVocabularySpeechResponse(
        transcript=transcript_text,
        words_detected=detected,
        scores=scores,
        coach_feedback=coach_feedback,
        archetype=archetype,
    )
