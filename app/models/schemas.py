"""
Request/response shapes for the FastAPI endpoints.
"""

from typing import Optional
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    # Which database to answer against. None -> the fixed sample dataset.
    # Otherwise a bare file name like "user_upload_<session_id>.db" (resolved
    # by the backend inside its data/ directory).
    database: Optional[str] = None


class AnswerResponse(BaseModel):
    question: str
    needs_clarification: bool
    clarifying_question: Optional[str] = None
    sql: Optional[str] = None
    result: Optional[dict] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
    attempts: Optional[int] = None
    success: Optional[bool] = None
    blocked: Optional[bool] = None
    detected_language: Optional[str] = None
    translated_question: Optional[str] = None
    trace: list = []
