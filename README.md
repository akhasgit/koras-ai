# Koras Demo API

A Modal-hosted FastAPI service that powers the `/demo` page on the Koras
marketing site. It takes a short voice clip, extracts acoustic features,
transcribes it with Whisper, analyzes the transcript with Claude, and returns
scores plus written feedback.

## Layout

```
backend/
├── app.py            # single-file Modal app
├── requirements.txt  # for local tooling only (Modal builds from the image spec in app.py)
├── .env.example
└── README.md         # you are here
```

Everything the service needs at runtime is declared in the Modal image at the
top of `app.py`. The `requirements.txt` here is only for local linting /
IDE type hints.

## Prerequisites

```bash
pip install modal
modal token new
```

Then create the secrets Modal will inject at runtime:

```bash
modal secret create openai-api-key       OPENAI_API_KEY=sk-...
modal secret create anthropic-api-key    ANTHROPIC_API_KEY=sk-ant-...
```

## Deploy

```bash
cd backend
modal deploy app.py
```

Modal will print a URL that looks like:

```
https://<your-username>--koras-demo-fastapi-app.modal.run
```

Copy that and drop it into `koras-web/.env.local`:

```
NEXT_PUBLIC_DEMO_API_URL=https://<your-username>--koras-demo-fastapi-app.modal.run
```

Then restart the Next.js dev server so it picks up the new env var.

## Dev loop (hot reload)

```bash
modal serve app.py
```

Every time you save `app.py`, Modal redeploys the function. Use this while
iterating on prompts and scoring formulas — much faster than `modal deploy`.

## Endpoints

### `GET /health`

Liveness probe.

```bash
curl https://<your-url>/health
# {"status":"ok"}
```

### `POST /analyze`

Multipart upload. Field name: `audio`. Accepts webm, mp3, m4a, wav — anything
ffmpeg can decode. 10 MB limit.

```bash
curl -X POST -F "audio=@clip.webm" https://<your-url>/analyze
```

Response shape (see `AnalyzeResponse` in `app.py`):

```jsonc
{
  "scores":   { "pitch": 82, "pace": 74, "clarity": 88, "resonance": 79, "confidence": 70, "overall": 79 },
  "metrics":  { "duration_seconds": 28.4, "words_per_minute": 152.3, "mean_pitch_hz": 189.1, "pitch_std_hz": 42.7, "hnr_db": 18.2, "pause_count": 4, "long_pause_count": 1 },
  "transcript": "Hi, my name is...",
  "transcript_analysis": { ... },
  "coach_feedback": "There's a real warmth in how you land on key words...",
  "archetype": "The Warm Communicator"
}
```

## Cost (approx. per request)

| Item | Cost |
|---|---|
| Whisper `whisper-1` for a 30 s clip | ~$0.003 |
| Claude transcript analysis (~1.5k in / 600 out) | ~$0.010 |
| Claude coach feedback (~500 in / 100 out) | ~$0.003 |
| Modal compute (~10 s @ 2 CPU) | ~$0.002 |
| **Total** | **~$0.018** |

## Tuning

A few named constants at the top of `app.py`:

| Constant | Default | When to change |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Bump to `claude-sonnet-4-6` when available |
| `MAX_UPLOAD_BYTES` | 10 MB | Raise if you extend demo beyond 30 s |
| `MIN_TRANSCRIPT_WORDS` | 10 | Lower to 5 if real 30 s clips get rejected |
| `ALLOWED_ORIGINS` | localhost + koras.com | Add staging / custom domains here |

## Troubleshooting

- **CORS errors** — add your origin to `ALLOWED_ORIGINS` and redeploy.
- **"Audio too short or unclear"** on real 30 s clips — Whisper sometimes
  returns short transcripts on unclear audio. Drop `MIN_TRANSCRIPT_WORDS` to
  5 in `app.py`.
- **Cold start ≥ 30 s** — add `keep_warm=1` to the `@app.function(...)`
  decorator. Costs ~$5/month, eliminates cold starts.
- **Claude returns invalid JSON** — `_strip_fences` handles markdown fences
  and stray prose; if you see failures, the error message includes a 300-char
  preview of the response to help you iterate on the prompt.
- **Scores look off** — `print(features)` at the top of `compute_scores` and
  check the Modal dashboard logs for the raw measurements.

## Out of scope (for now)

- No auth, rate limiting, or persistence.
- No audio storage — every clip is analyzed and forgotten within the request.
- No payment wiring.
