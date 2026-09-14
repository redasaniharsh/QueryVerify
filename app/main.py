"""
FastAPI entrypoint.
"""

from fastapi import FastAPI, HTTPException, Request
from app.db.connection import get_engine
from app.agents.orchestrator import handle_question
from app.models.schemas import QuestionRequest, AnswerResponse
from app.config import settings
from app.rate_limiter import SlidingWindowRateLimiter

app = FastAPI(title="QueryVerify")

rate_limiter = SlidingWindowRateLimiter(
    limit=settings.rate_limit_per_minute,
    exempt_loopback=settings.rate_limit_exempt_loopback,
)


def _get_client_ip(request: Request) -> str:
    # Check X-Forwarded-For header first (supports reverse proxies & simulated client IPs)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest, request: Request):
    client_ip = _get_client_ip(request)
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — please wait a moment before asking another question.",
        )
    try:
        engine = get_engine(req.database)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = handle_question(req.question, engine)
    return AnswerResponse(question=req.question, **result)
