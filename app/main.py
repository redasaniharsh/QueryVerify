"""
FastAPI entrypoint.
"""

from fastapi import FastAPI
from app.db.connection import engine
from app.agents.orchestrator import handle_question
from app.models.schemas import QuestionRequest, AnswerResponse

app = FastAPI(title="QueryVerify")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    result = handle_question(req.question, engine)
    return AnswerResponse(question=req.question, **result)
