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
    return {
        "status": response.status.value,
        "message": response.message,
        "assessment_plan": response.assessment_plan,
        "closest_matches": response.closest_matches,
        "state": {
            "domains": response.state.domains,
            "skills": response.state.skills,
            "specializations": response.state.specializations,
            "negative_constraints": response.state.negative_constraints,
            "turns": response.state.turns,
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
