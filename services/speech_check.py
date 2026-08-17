"""Speech Check V1 scoring — Whisper segment logprob mapped onto words + acoustics."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from config import CLAUDE_MODEL, anthropic_client
from models.speech_check import (
    AnalyzeSpeechCheckResponse,
    FlaggedWord,
    RescoreSpeechCheckRequest,
    RescoreSpeechCheckResponse,
    SpeechCheckScores,
    WordToken,
)
from prompts.speech_check import SPEECH_CHECK_COACH_PROMPT
from utils.text import clamp, count_fillers, strip_fences

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "is", "it", "be",
    "as", "at", "by", "or", "we", "you", "i", "my", "me", "our", "your", "that",
    "this", "with", "from", "was", "were", "are", "been", "have", "has", "had",
    "do", "does", "did", "not", "but", "if", "so", "than", "then", "too", "very",
    "can", "will", "just", "about", "into", "up", "out", "over", "after",
    "before", "also", "there", "their", "they", "them", "he", "she", "his",
    "her", "its", "i'm", "don't", "it's", "that's",
}

UNCERTAIN_LOGPROB = -1.0
FLAG_LOGPROB = -0.45

_IPA: dict[str, str] = {
    "morning": "/ˈmɔːnɪŋ/", "house": "/haʊs/", "water": "/ˈwɔːtə/",
    "people": "/ˈpiːpl/", "because": "/bɪˈkɒz/", "thought": "/θɔːt/",
    "through": "/θruː/", "enough": "/ɪˈnʌf/", "another": "/əˈnʌðə/",
    "important": "/ɪmˈpɔːtnt/", "comfortable": "/ˈkʌmftəbl/",
    "usually": "/ˈjuːʒuəli/", "probably": "/ˈprɒbəbli/",
    "different": "/ˈdɪfrənt/", "together": "/təˈɡeðə/",
    "question": "/ˈkwestʃən/", "answer": "/ˈɑːnsə/",
}


def _hnr_clarity(hnr: float) -> int:
    if hnr <= 0:
        return 40
    return clamp(int(min(100, (hnr / 20) * 100)))


def _logprob_to_score(logprob: float | None) -> int:
    if logprob is None:
        return 70
    return clamp(int(100 + float(logprob) * 80))


def _token_overlap(passage: str, transcript: str) -> float:
    def toks(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[a-zA-Z']+", s or "") if len(t) > 2}
    a, b = toks(passage), toks(transcript)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _is_content(word: str, index: int) -> bool:
    raw = (word or "").strip()
    if not raw:
        return False
    clean = re.sub(r"[^a-zA-Z']", "", raw)
    if not clean:
        return False
    lower = clean.lower()
    if lower in STOPWORDS:
        return False
    if any(ch.isdigit() for ch in raw):
        return False
    if clean.isupper() and len(clean) > 1:
        return False
    if index > 0 and clean[:1].isupper() and clean[1:].islower() and len(clean) > 2:
        return False
    return True


def _ipa_for(word: str) -> str:
    key = re.sub(r"[^a-zA-Z']", "", word or "").lower()
    return _IPA.get(key, "")


def _word_issue(logprob: float | None, duration: float, median: float) -> str:
    if logprob is not None and logprob < UNCERTAIN_LOGPROB:
        return "uncertain"
    if median > 0 and duration < median * 0.55:
        return "rushed"
    return "unclear"


def _flag_from_token(
    token: dict[str, Any],
    hnr_score: int,
    median_dur: float,
    existing_id: str | None = None,
) -> FlaggedWord:
    start = float(token.get("start_s") or token.get("start") or 0.0)
    end = float(token.get("end_s") or token.get("end") or 0.0)
    duration = max(0.0, end - start)
    logprob = token.get("segment_logprob")
    if logprob is None:
        logprob = token.get("avg_logprob")
    logprob_f = float(logprob) if logprob is not None else None
    intel = clamp(int(_logprob_to_score(logprob_f) * 0.7 + hnr_score * 0.3))
    word = str(token.get("word") or "").strip()
    return FlaggedWord(
        id=existing_id or token.get("id") or str(uuid.uuid4()),
        word=word,
        ipa=_ipa_for(word),
        start_s=round(start, 3),
        end_s=round(end, 3),
        start_ms=int(start * 1000),
        end_ms=int(end * 1000),
        segment_logprob=logprob_f,
        intelligibility=intel,
        issue=_word_issue(logprob_f, duration, median_dur),  # type: ignore[arg-type]
    )


def _rank_flags(
    tokens: list[dict[str, Any]],
    hnr_score: int,
    mode: str,
    existing_by_key: dict[str, str] | None = None,
) -> tuple[list[FlaggedWord], list[FlaggedWord]]:
    durs = []
    for t in tokens:
        start = float(t.get("start_s") or t.get("start") or 0.0)
        end = float(t.get("end_s") or t.get("end") or 0.0)
        if end > start:
            durs.append(end - start)
    durs_sorted = sorted(durs)
    median = durs_sorted[len(durs_sorted) // 2] if durs_sorted else 0.0

    candidates: list[FlaggedWord] = []
    for i, t in enumerate(tokens):
        word = str(t.get("word") or "").strip()
        if not _is_content(word, i):
            continue
        logprob = t.get("segment_logprob")
        if logprob is None:
            logprob = t.get("avg_logprob")
        if mode != "read" and logprob is not None and float(logprob) > FLAG_LOGPROB:
            continue
        if mode != "read" and logprob is None:
            continue
        key = f"{word.lower()}|{round(float(t.get('start_s') or t.get('start') or 0.0), 2)}"
        existing_id = (existing_by_key or {}).get(key)
        flagged = _flag_from_token(t, hnr_score, median, existing_id)
        if mode == "read" and (logprob is None or float(logprob) > FLAG_LOGPROB) and flagged.issue != "rushed":
            continue
        candidates.append(flagged)

    def sort_key(f: FlaggedWord) -> tuple:
        lp = f.segment_logprob if f.segment_logprob is not None else 0.0
        return (lp, -len(f.word))

    candidates.sort(key=sort_key)
    top = candidates[:6]
    rest = candidates[6:12]
    return top, rest


def _session_scores(
    tokens: list[dict[str, Any]],
    flagged: list[FlaggedWord],
    features: dict[str, Any],
    duration: float,
    transcript: str,
) -> SpeechCheckScores:
    content = [t for i, t in enumerate(tokens) if _is_content(str(t.get("word") or ""), i)]
    proxies = []
    hnr_score = _hnr_clarity(float(features.get("hnr") or 0.0))
    for t in content:
        lp = t.get("segment_logprob")
        if lp is None:
            lp = t.get("avg_logprob")
        proxies.append(_logprob_to_score(float(lp) if lp is not None else None) * 0.7 + hnr_score * 0.3)
    if flagged:
        intel = clamp(int(sum(f.intelligibility for f in flagged) / len(flagged)))
    elif proxies:
        intel = clamp(int(sum(proxies) / len(proxies)))
    else:
        intel = 80

    words = max(1, len((transcript or "").split()))
    wpm = (words / duration * 60) if duration > 0 else 0.0
    fillers = count_fillers(transcript)
    long_pauses = int(features.get("long_pause_count") or 0)
    pace = clamp(int(100 - abs(wpm - 130) / 1.3)) if wpm > 0 else 50
    fluency = clamp(int(pace - max(0, long_pauses - 2) * 4 - min(8, fillers * 2)))
    clarity = clamp(int(hnr_score - max(0, long_pauses - 2) * 3))
    overall = clamp(int(intel * 0.40 + fluency * 0.25 + pace * 0.20 + clarity * 0.15))
    return SpeechCheckScores(
        intelligibility=intel,
        fluency=fluency,
        pace=pace,
        clarity_acoustic=clarity,
        overall=overall,
    )


def _normalise_tokens(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, WordToken):
            if item.deleted:
                continue
            out.append({
                "id": item.id,
                "word": item.word,
                "start": item.start_s,
                "start_s": item.start_s,
                "end": item.end_s,
                "end_s": item.end_s,
                "segment_logprob": item.segment_logprob,
                "avg_logprob": item.segment_logprob,
                "no_speech_prob": item.no_speech_prob,
            })
            continue
        if not isinstance(item, dict):
            continue
        if item.get("deleted"):
            continue
        start = float(item.get("start_s") or item.get("start") or 0.0)
        end = float(item.get("end_s") or item.get("end") or 0.0)
        lp = item.get("segment_logprob")
        if lp is None:
            lp = item.get("avg_logprob")
        out.append({
            "id": item.get("id"),
            "word": str(item.get("word") or "").strip(),
            "start": start,
            "start_s": start,
            "end": end,
            "end_s": end,
            "segment_logprob": lp,
            "avg_logprob": lp,
            "no_speech_prob": item.get("no_speech_prob"),
        })
    return out


def score_from_tokens(
    tokens: list[Any],
    features: dict[str, Any],
    mode: str,
    passage: str | None,
    existing_flagged: list[FlaggedWord] | None = None,
) -> tuple[str, SpeechCheckScores, list[FlaggedWord], list[FlaggedWord], bool, dict[str, Any]]:
    norm = _normalise_tokens(tokens)
    transcript = " ".join(t["word"] for t in norm if t["word"]).strip()
    duration = float(features.get("duration") or 0.0)
    if duration <= 0 and norm:
        duration = max(float(t["end_s"]) for t in norm)
    overlap = _token_overlap(passage or "", transcript) if mode == "read" and passage else 1.0
    off_script = mode == "read" and bool(passage) and overlap < 0.45
    hnr_score = _hnr_clarity(float(features.get("hnr") or 0.0))
    existing_by_key: dict[str, str] = {}
    for f in existing_flagged or []:
        existing_by_key[f"{f.word.lower()}|{round(f.start_s, 2)}"] = f.id
    if off_script:
        flagged, rest = [], []
    else:
        flagged, rest = _rank_flags(norm, hnr_score, mode, existing_by_key)
    scores = _session_scores(norm, flagged, features, duration, transcript)
    acoustic = {
        "duration_seconds": round(duration, 2),
        "words_per_minute": round((len(transcript.split()) / duration * 60) if duration > 0 else 0.0, 1),
        "pause_count": features.get("pause_count"),
        "long_pause_count": features.get("long_pause_count"),
        "hnr": features.get("hnr"),
        "mean_f0": features.get("mean_f0"),
        "alignment_coverage": round(overlap, 3) if mode == "read" else None,
        "words": norm,
    }
    return transcript, scores, flagged, rest, off_script, acoustic


async def _coach_and_ipa(
    mode: str,
    prompt: str,
    transcript: str,
    flagged: list[FlaggedWord],
) -> str:
    if not flagged:
        return "Those words landed clearly. Keep that pace when you speak a little longer."
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": SPEECH_CHECK_COACH_PROMPT.format(
                    mode=mode,
                    prompt=(prompt or "")[:1500],
                    transcript=transcript[:2000],
                    flagged_json=json.dumps([{"word": f.word, "issue": f.issue} for f in flagged]),
                ),
            }],
        )
        data = json.loads(strip_fences(msg.content[0].text if msg.content else ""))
        ipa_map = data.get("ipa") if isinstance(data.get("ipa"), dict) else {}
        for f in flagged:
            key = re.sub(r"[^a-zA-Z']", "", f.word).lower()
            got = ipa_map.get(f.word) or ipa_map.get(key)
            if got and not f.ipa:
                f.ipa = str(got)
        return str(data.get("coach_note") or "").strip()[:240]
    except Exception:
        return "Words that may not have landed clearly — practise them below."


def analyze_speech_check(
    transcript_data: dict[str, Any],
    features: dict[str, Any],
    mode: str,
    passage: str | None,
) -> AnalyzeSpeechCheckResponse:
    words = transcript_data.get("words") or []
    text = (transcript_data.get("text") or "").strip()
    duration = float(features.get("duration") or transcript_data.get("duration") or 0.0)
    word_count = len(text.split())
    segments = transcript_data.get("segments") or []
    no_speech = 0.0
    if segments:
        no_speech = max(float(s.get("no_speech_prob") or 0.0) for s in segments)

    too_short = word_count < 10 or duration < 8 or not text or no_speech > 0.85
    if too_short:
        return AnalyzeSpeechCheckResponse(
            transcript=text,
            words=words,
            scores=SpeechCheckScores(
                intelligibility=0, fluency=0, pace=0, clarity_acoustic=0, overall=0,
            ),
            flagged_words=[],
            acoustic_metrics={
                "duration_seconds": round(duration, 2),
                "no_speech_prob": no_speech,
                "words": words,
            },
            too_short=True,
        )

    transcript, scores, flagged, rest, off_script, acoustic = score_from_tokens(
        words, {**features, "duration": duration}, mode, passage,
    )
    if segments:
        acoustic["avg_logprob"] = sum(float(s.get("avg_logprob") or 0.0) for s in segments) / len(segments)
        acoustic["no_speech_prob"] = no_speech
    return AnalyzeSpeechCheckResponse(
        transcript=transcript or text,
        words=acoustic.get("words") or words,
        scores=scores,
        flagged_words=flagged,
        also_noticed=rest,
        acoustic_metrics=acoustic,
        off_script=off_script,
        too_short=False,
    )


async def analyze_speech_check_with_coach(
    transcript_data: dict[str, Any],
    features: dict[str, Any],
    mode: str,
    passage: str | None,
    prompt: str,
) -> AnalyzeSpeechCheckResponse:
    out = analyze_speech_check(transcript_data, features, mode, passage)
    if out.too_short:
        return out
    if out.off_script:
        out.coach_note = (
            "It sounded like a different text from the passage. "
            "Fluency is scored; word-level highlights are skipped."
        )
        return out
    out.coach_note = await _coach_and_ipa(mode, prompt, out.transcript, out.flagged_words)
    return out


def rescore_speech_check(req: RescoreSpeechCheckRequest) -> RescoreSpeechCheckResponse:
    features = dict(req.acoustic_metrics or {})
    transcript, scores, flagged, rest, off_script, _acoustic = score_from_tokens(
        req.words,
        features,
        req.mode,
        req.passage_text,
        req.existing_flagged,
    )
    for f in flagged:
        if not f.ipa:
            f.ipa = _ipa_for(f.word)
    return RescoreSpeechCheckResponse(
        transcript=transcript,
        scores=scores,
        flagged_words=flagged,
        also_noticed=rest,
        off_script=off_script,
    )
