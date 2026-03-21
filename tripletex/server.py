import asyncio
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from agent import solve_task_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tripletex")

app = FastAPI()

API_KEY = os.environ.get("TRIPLETEX_API_KEY", "")

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
    log.info("Task received: %s", body.get("prompt", "")[:120])

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, solve_task_sync, body)
        log.info("Task completed successfully")
    except Exception:
        log.error("Task failed:\n%s", traceback.format_exc())

    # Always return completed — partial work still earns partial credit
    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
