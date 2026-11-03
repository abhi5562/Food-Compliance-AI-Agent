from langgraph.graph.message import add_messages
from enum import Enum
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

class InspectionCategory(str, Enum):
    FOOD_STORAGE = "Food Storage"
    KITCHEN_CLEANLINESS = "Kitchen Cleanliness"
    FOOD_HANDLER_HYGIENE = "Food Handler Hygiene"
    EQUIPMENT_CLEANLINESS = "Equipment Cleanliness"


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class InspectionFinding(BaseModel):
    observation: str = Field(min_length=1, description="Visible evidence of a potential violation in the image.")
    sop_requirement: str = Field(min_length=1)
    citation_reference: str = Field(min_length=1, description="Exact reference_id from rag_tool.")
    advisory_correction: str = Field(
        min_length=1,
        description="Suggested corrective action based on the cited SOP.",
    )


class InspectionResult(BaseModel):
    assessment: Literal["NO_ISSUE_OBSERVED", "POTENTIAL_VIOLATION", "INSUFFICIENT_EVIDENCE"]
    summary: str = Field(min_length=1)
    findings: list[InspectionFinding]
    evidence_limitations: list[str]


class InspectionState(TypedDict):
    image_bytes: bytes
    mime_type: str
    category: InspectionCategory
    analysis: dict | None
    retrieved_evidence: list[dict]
    image_b64: str


class CorrectiveProposal(BaseModel):
    decision: Literal["NEW_PROPOSAL", "POSSIBLE_DUPLICATE"]
    title: str | None
    description: str | None
    existing_task_id: str | None
    reason: str

    @model_validator(mode="after")
    def validate_decision(self):
        if self.decision == "NEW_PROPOSAL" and (not self.title or not self.description):
            raise ValueError("A new proposal requires a title and description.")
        if self.decision == "POSSIBLE_DUPLICATE" and not self.existing_task_id:
            raise ValueError("A possible duplicate requires an existing task ID.")
        return self


class ResolutionVerificationResult(BaseModel):
    assessment: Literal["APPEARS_RESOLVED", "NOT_RESOLVED", "INSUFFICIENT_EVIDENCE"]
    summary: str = Field(min_length=1)
    observations: list[str]
    evidence_limitations: list[str]


class ResolutionVerificationState(TypedDict):
    task_context: dict
    original_image_bytes: bytes
    original_mime_type: str
    correction_image_bytes: bytes
    correction_mime_type: str
    description: str | None
    verification: dict
