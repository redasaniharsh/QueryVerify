"""
FastAPI entrypoint.
"""

from fastapi import FastAPI, HTTPException
from app.db.connection import get_engine
from app.agents.orchestrator import handle_question
from app.models.schemas import QuestionRequest, AnswerResponse

app = FastAPI(title="QueryVerify")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        engine = get_engine(req.database)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = handle_question(req.question, engine)
    return AnswerResponse(question=req.question, **result)
