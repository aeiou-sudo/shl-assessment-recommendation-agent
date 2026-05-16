"""FastAPI application for the assessment recommendation agent."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.services.assessment_agent import AssessmentRecommendationAgent
from app.state.conversation_state import ConversationState


app = FastAPI(title="SHL Assessment Recommendation Agent")
agent = AssessmentRecommendationAgent.from_catalogue()
sessions: dict[str, ConversationState] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    query: str


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    state = sessions.setdefault(request.session_id, ConversationState())
    response = agent.handle(request.query, state)
    payload: dict[str, object] = {
        "message": response.message,
        "recommendations": response.recommendations or None,
        "assessment_plan": response.assessment_plan,
        "end_of_conversation": response.end_of_conversation,
    }
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
