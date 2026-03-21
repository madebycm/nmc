import asyncio
import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "gemini")  # "gemini" or "claude"

if AGENT_BACKEND == "claude":
    from agent_claude import solve_task_sync
else:
    from agent import solve_task_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tripletex")

app = FastAPI()

API_KEY = os.environ.get("TRIPLETEX_API_KEY", "")
TASK_LOG_FILE = os.environ.get("TASK_LOG_FILE", "/opt/tripletex/task_log.jsonl")

# Thread pool for concurrent task execution (sync requests blocks event loop)
_executor = ThreadPoolExecutor(max_workers=8)


@app.post("/solve")
async def solve(request: Request):
    # Optional bearer token auth
    if API_KEY:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_KEY}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.json()
    prompt = body.get("prompt", "")
    log.info("Task received: %s", prompt[:120])
    t0 = time.time()

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, solve_task_sync, body)
        log.info("Task completed successfully")
    except Exception:
        log.error("Task failed:\n%s", traceback.format_exc())
        # Log crashed tasks so they don't disappear
        try:
            crash_entry = {
                "prompt": prompt,
                "outcome": "crash",
                "error": traceback.format_exc()[-500:],
                "elapsed_s": round(time.time() - t0, 1),
                "logged_at": datetime.utcnow().isoformat() + "Z",
            }
            with open(TASK_LOG_FILE, "a") as f:
                f.write(json.dumps(crash_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Always return completed — partial work still earns partial credit
    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
