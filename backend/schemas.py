from pydantic import BaseModel, Field
from typing import Literal, Optional
from uuid import uuid4


class ResumeProfile(BaseModel):
    """Structured profile extracted from a resume (PDF/text)."""
    name: str = ""
    email: str | None = None
    skills: list[str] = Field(default_factory=list)
    years_experience: int = Field(ge=0, default=0)
    seniority: Literal["junior", "mid", "senior", "lead"] = "mid"
    domain: str = ""
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    summary: str = ""


# ── Input schemas ──
class CandidateProfile(BaseModel):
    """Structured candidate profile stored in PostgreSQL."""
    skills: list[str]
    years_experience: int = Field(ge=0)
    seniority: Literal["junior", "mid", "senior", "lead"] = "mid"
    domain: str = ""
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    summary: str = ""
    raw_text: str = ""  # Raw resume text for semantic matching


class JDRequirements(BaseModel):
    """Structured extraction from job description."""
    required_skills: list[str]
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority_level: str = ""
    domain: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class PrioritizedGap(BaseModel):
    """Single ranked skill gap entry."""
    skill: str
    priority_rank: int = Field(ge=1)
    estimated_match_gain_pct: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=5)


# ── Request models ──
class CreateCandidateRequest(BaseModel):
    name: str = "Candidate"
    email: str = ""
    skills: list[str]
    years_experience: int = Field(ge=0)
    seniority: Literal["junior", "mid", "senior", "lead"] = "mid"
    domain: str = ""
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    summary: str = ""


class CreateMatchesRequest(BaseModel):
    candidate_id: str
    jd_inputs: list[str] = Field(min_length=1, max_length=10)


# ── Output schema ──
class ResourceItem(BaseModel):
    title: str
    url: str
    estimated_hours: int = Field(ge=1, le=100)
    type: Literal["course", "project", "cert", "certification", "doc"]
    relevance_score: float = Field(ge=0.0, le=1.0)


class LearningPlanItem(BaseModel):
    skill: str
    priority_rank: int = Field(ge=1)
    estimated_match_gain_pct: int = Field(ge=0, le=100)
    resources: list[ResourceItem] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=5)


class ToolCallTrace(BaseModel):
    tool: str
    status: Literal["success", "failed", "timeout", "fallback"]
    latency_ms: int = Field(ge=0)


class AgentTrace(BaseModel):
    tool_calls: list[ToolCallTrace] = []
    total_llm_calls: int = 0
    fallbacks_triggered: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_ms: int = 0


class DimensionScores(BaseModel):
    """Typed dimension scores matching the spec."""
    skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    seniority_fit: int = Field(ge=0, le=100)


class AgentOutputForLLM(BaseModel):
    """Schema for the agent's final JSON output (without agent_trace).

    The agent produces this via output_schema. The orchestrator (JobRunner)
    then parses the final_text from the event stream and adds agent_trace.
    """
    job_id: str
    overall_score: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    dimension_scores: DimensionScores = Field(default_factory=DimensionScores)
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=10, max_length=1000)
    learning_plan: list[LearningPlanItem] = Field(min_length=0, max_length=10)


class AgentOutput(BaseModel):
    """Final structured output. Stored in match_jobs.result JSONB."""
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    overall_score: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    dimension_scores: DimensionScores = Field(default_factory=DimensionScores)
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=10, max_length=1000)
    learning_plan: list[LearningPlanItem] = Field(min_length=0, max_length=10)
    agent_trace: Optional[AgentTrace] = None
