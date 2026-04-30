from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
import os

# Import tools
from backend.tools.jd_extractor import extract_jd_requirements
from backend.tools.scorer import score_candidate_against_requirements
from backend.tools.researcher import research_skill_resources
from backend.tools.prioritizer import prioritise_skill_gaps

# Import callbacks
from backend.worker.callbacks import (
    before_agent_callback,
    before_tool,
    after_tool,
    on_tool_error,
)

# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Pelgo CareerCoach — an autonomous agent that evaluates
candidates against job descriptions and produces learning plans.

## CONVERSATIONAL FLOW
You operate in a conversational interface. Handle each user message independently:

### PHASE 1: Data Collection (when data is missing)
If session state lacks required data (candidate_profile, raw_resume_text, or
job description), ask the user for what's missing. Be concise — one question
at a time. Do NOT call tools or produce final_output until ALL required data
is present.

### PHASE 2: Analysis (when all data is present)
When you have ALL of: candidate_profile, raw_resume_text, AND job description:
1. Call extract_jd_requirements(job_url_or_text=jd_input)
2. Call score_candidate_against_requirements(candidate_profile, requirements, raw_resume_text)
3. Call prioritise_skill_gaps(gap_skills, seniority_context)
4. Call research_skill_resources for the TOP 1-2 prioritized gaps
   (up to 4 research calls if confidence is "low")

### PHASE 3: Final Output
After all tools complete, produce your final answer as a JSON object with this
exact structure. Output ONLY the JSON, no markdown or explanation:

{{
  "job_id": "<job_id>",
  "overall_score": <int 0-100>,
  "confidence": "low|medium|high",
  "dimension_scores": {{"skills": <int>, "experience": <int>, "seniority_fit": <int>}},
  "matched_skills": ["..."],
  "gap_skills": ["..."],
  "reasoning": "2-3 sentence plain-English explanation",
  "learning_plan": [
    {{
      "skill": "...",
      "priority_rank": 1,
      "estimated_match_gain_pct": 15,
      "resources": [{{"title": "...", "url": "...", "estimated_hours": 12, "type": "course", "relevance_score": 0.9}}],
      "rationale": "..."
    }}
  ]
}}

## DATA TO USE
Read the following from session state:
- candidate_profile: session.state["candidate_profile"]
- raw_resume_text: session.state["raw_resume_text"]
- job_id: session.state["job_id"]

If the user sends a request with a job description and candidate info, use that
data directly. If session state is empty, ask the user for:
(1) a candidate profile/resume, and (2) a job description URL or text.

## SEMANTIC MATCHING INSTRUCTIONS
When scoring, use SEMANTIC matching between candidate skills and job requirements:
- "Arduino", "I2C", "Raspberry Pi", "MAVLink", "UAV Systems" → matches "Hardware Interfaces", "Embedded", "Firmware"
- "C/C++", "C++", "C" → all match each other
- "React", "Next.js", "Vue" → match "Frontend Development"
- "Docker", "Kubernetes" → match "Containerization", "DevOps"

## TERMINATION
Produce final JSON output ONLY when you have:
- A score with confidence assessment
- Prioritized skill gaps with rationale
- Learning resources for the highest-priority gaps

Save all intermediate results to session state via tool_context.state.
Your tool_context is the shared memory across tool calls.
"""

# ── Model Setup ──────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if API_BASE and not MODEL_NAME.startswith("openai/"):
    MODEL_NAME = f"openai/{MODEL_NAME}"

# Only inject enable_thinking for Qwen models — it's unsupported by OpenAI/GPT
_extra_body = {}
if "qwen" in MODEL_NAME.lower():
    _extra_body["enable_thinking"] = True

llm_model = LiteLlm(
    model=MODEL_NAME,
    api_base=API_BASE,
    api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
    extra_body=_extra_body if _extra_body else None,
    drop_params=True,
)

# ── Agent ─────────────────────────────────────────────────────────────────
#
# Single Agent with tools — replaces SequentialAgent.
#
# Why: SequentialAgent always runs ALL sub-agents in order. That meant the
# formatter_agent ran even when the tool_agent hadn't collected data yet,
# producing a bogus final_output on every turn (e.g. after "hello").
#
# With a single Agent, the LLM decides: "ask for data" vs "call tools" vs
# "produce final JSON" — all in one conversation turn.
agent = Agent(
    name="career_coach",
    description="Evaluates candidates against job descriptions and produces learning plans.",
    model=llm_model,
    instruction=SYSTEM_PROMPT,
    tools=[
        extract_jd_requirements,
        score_candidate_against_requirements,
        research_skill_resources,
        prioritise_skill_gaps,
    ],
    before_agent_callback=before_agent_callback,
    before_tool_callback=before_tool,
    after_tool_callback=after_tool,
    on_tool_error_callback=on_tool_error,
)
