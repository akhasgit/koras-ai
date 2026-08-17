"""Optional client for the koras-phone Cloud Run service."""
from __future__ import annotations

import httpx

from config import KORAS_INTERNAL_SECRET, KORAS_PHONE_URL


async def align_phones(wav_bytes: bytes, reference_text: str) -> dict:
    """Call koras-phone. Returns {available: bool, ...} and never raises."""
    if not KORAS_PHONE_URL:
        return {"available": False, "reason": "KORAS_PHONE_URL unset"}
    url = f"{KORAS_PHONE_URL.rstrip('/')}/align-phones"
    headers = {"x-koras-secret": KORAS_INTERNAL_SECRET} if KORAS_INTERNAL_SECRET else {}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                url,
                files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
                data={"reference_text": reference_text},
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            data.setdefault("available", False)
            return data
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)[:300]}
