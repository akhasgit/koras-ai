import io
import os
import tempfile

import ffmpeg
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from config import openai_client, WHISPER_MODEL


def normalize_audio(audio_bytes: bytes, filename: str) -> bytes:
    suffix = os.path.splitext(filename)[1] or ".bin"
    in_path: str | None = None
    out_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        out_path = in_path + ".wav"
        (
            ffmpeg
            .input(in_path)
            .output(out_path, ac=1, ar=16000, format="wav", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        with open(out_path, "rb") as f:
            return f.read()
    except ffmpeg.Error as e:  # type: ignore[attr-defined]
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed: {stderr[:400]}") from e
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def extract_features(wav_bytes: bytes) -> dict:
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        snd = parselmouth.Sound(wav_path)
        duration = snd.duration

        pitch_obj = snd.to_pitch()
        pitch_values = pitch_obj.selected_array["frequency"]
        voiced = pitch_values[pitch_values > 0]
        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
        std_f0 = float(np.std(voiced)) if len(voiced) > 0 else 0.0

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        hnr = float(hnr) if hnr == hnr else 0.0  # nan guard

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        intervals = librosa.effects.split(y, top_db=30)
        pause_count = max(0, len(intervals) - 1)
        long_pause_count = 0
        if len(intervals) > 1:
            gap_starts = intervals[:-1, 1]
            gap_ends = intervals[1:, 0]
            gaps = (gap_ends - gap_starts) / sr
            long_pause_count = int(np.sum(gaps > 0.5))

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        return {
            "duration": duration,
            "mean_f0": mean_f0,
            "std_f0": std_f0,
            "hnr": hnr,
            "pause_count": int(pause_count),
            "long_pause_count": int(long_pause_count),
            "spectral_centroid": centroid,
        }
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


async def transcribe(wav_bytes: bytes) -> dict:
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"

    result = await openai_client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        temperature=0,
    )

    segments_raw = getattr(result, "segments", None) or []
    segments = []
    for s in segments_raw:
        if hasattr(s, "avg_logprob"):
            segments.append({
                "avg_logprob": float(getattr(s, "avg_logprob", 0.0) or 0.0),
                "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0) or 0.0),
                "start": float(getattr(s, "start", 0.0) or 0.0),
                "end": float(getattr(s, "end", 0.0) or 0.0),
            })
        elif isinstance(s, dict):
            segments.append({
                "avg_logprob": float(s.get("avg_logprob") or 0.0),
                "no_speech_prob": float(s.get("no_speech_prob") or 0.0),
                "start": float(s.get("start") or 0.0),
                "end": float(s.get("end") or 0.0),
            })

    def _segment_for_word(start: float, end: float) -> dict | None:
        best = None
        best_overlap = -1.0
        for seg in segments:
            overlap = min(end, seg["end"]) - max(start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best = seg
        return best if best is not None and best_overlap >= 0 else None

    words_raw = getattr(result, "words", None) or []
    words = []
    for w in words_raw:
        if hasattr(w, "word"):
            token = w.word
            start = float(w.start)
            end = float(w.end)
        else:
            token = w["word"]
            start = float(w["start"])
            end = float(w["end"])
        seg = _segment_for_word(start, end)
        words.append({
            "word": token,
            "start": start,
            "end": end,
            "avg_logprob": None if seg is None else seg["avg_logprob"],
            "no_speech_prob": None if seg is None else seg["no_speech_prob"],
        })

    return {
        "text": (result.text or "").strip(),
        "words": words,
        "duration": float(getattr(result, "duration", 0.0) or 0.0),
        "segments": segments,
    }
