from fastapi import Header, HTTPException

from config import KORAS_INTERNAL_SECRET


def verify_internal_secret(x_koras_secret: str = Header(...)):
    if not KORAS_INTERNAL_SECRET or x_koras_secret != KORAS_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
