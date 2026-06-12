import json
import time


def log(endpoint: str, req_id: str, user_id: str | None, **extra) -> None:
    print(json.dumps({
        "ts": time.time(),
        "req_id": req_id,
        "endpoint": endpoint,
        "user_id": user_id,
        **extra,
    }))
