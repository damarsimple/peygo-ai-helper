from google.adk import Agent
from google.adk.agents import SequentialAgent
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

# Import schema
from backend.schemas import AgentOutputForLLM

# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TOOL = """You are Pelgo CareerCoach — an autonomous agent that evaluates
candidates against job descriptions and produces learning plans.

## CANDIDATE TO EVALUATE
{candidate_profile}

## RAW RESUME TEXT (use for semantic skill matching)
{raw_resume_text}

## JOB ID
{job_id}

## YOUR DECISION-DRIVEN WORKFLOW
You decide the tool-call sequence at runtime based on the data you discover.
Follow these decision rules:

1. ALWAYS START: Call extract_jd_requirements(job_url_or_text=jd_input)
   to parse the job description into structured requirements.

2. ALWAYS SCORE: Call score_candidate_against_requirements(
   candidate_profile=<candidate above>,
   requirements=<step 1 output>,
   raw_resume_text=<raw resume text above>
   ) to compute match score and identify skill gaps.

3. ALWAYS PRIORITIZE: Call prioritise_skill_gaps(
   gap_skills=<step 2 gap_skills>,
   seniority_context=<candidate seniority>
   ) to rank which gaps matter most. This ensures you focus on critical gaps.

4. FOCUSED RESEARCH: 
   - Based on the priority ranking from step 3, call research_skill_resources
     for the TOP 1-2 prioritized gaps.
   - IF the original score.confidence was "low", you may research 1 additional
     gap to gain better signal (max 4 total research calls).

## SEMANTIC MATCHING INSTRUCTIONS
When scoring, use SEMANTIC matching between candidate skills and job requirements:
- "Arduino", "I2C", "Raspberry Pi", "MAVLink", "UAV Systems" → matches "Hardware Interfaces", "Embedded", "Firmware"
- "C/C++", "C++", "C" → all match each other
- "React", "Next.js", "Vue" → match "Frontend Development"
- "Docker", "Kubernetes" → match "Containerization", "DevOps"

## TERMINATION
Produce a final answer when you have:
- A score with confidence assessment
- Prioritized skill gaps with rationale
- Learning resources for the highest-priority gaps

Save all intermediate results to session state via tool_context.state.
Your tool_context is the shared memory across tool calls.
"""

SYSTEM_PROMPT_FORMATTER = '''You are a JSON Formatter for Pelgo CareerCoach.
Construct the final structured output JSON using the data in session state:

Score data: {score}
Gap skills: {gap_skills}
Job ID: {job_id}

Build the learning_plan from the prioritized_gaps and research results.
Each learning_plan item MUST have:
- skill: the skill name
- priority_rank: integer (1 = highest priority)
- estimated_match_gain_pct: integer 0-100
- resources: array of {title, url, estimated_hours, type, relevance_score}
- rationale: why this skill should be learned first

Return ONLY this exact JSON structure with ALL required fields:
{{
  "job_id": "<from job_id>",
  "overall_score": <score.overall_score>,
  "confidence": "<score.confidence: low|medium|high>",
  "dimension_scores": {{
    "skills": <score.dimension_scores.skills>,
    "experience": <score.dimension_scores.experience>,
    "seniority_fit": <score.dimension_scores.seniority_fit>
  }},
  "matched_skills": <score.matched_skills>,
  "gap_skills": <gap_skills list>,
  "reasoning": "2-3 sentence plain-English explanation of the overall match",
  "learning_plan": [
    {{
      "skill": "skill_name",
      "priority_rank": 1,
      "estimated_match_gain_pct": 15,
      "resources": [
        {{"title": "...", "url": "...", "estimated_hours": 12, "type": "course", "relevance_score": 0.9}}
      ],
      "rationale": "Why this skill first"
    }}
  ]
}}
'''

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

# ── Agents ───────────────────────────────────────────────────────────────

# Agent 1: Executes tools and populates session state
tool_agent = Agent(
    name="career_tool_agent",
    description="Executes career coaching tools and saves intermediate results.",
    model=llm_model,
    instruction=SYSTEM_PROMPT_TOOL,
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

# Agent 2: Formats session state into structured JSON (output_schema works without tools!)
formatter_agent = Agent(
    name="career_formatter",
    description="Formats intermediate results into structured JSON output.",
    model=llm_model,
    instruction=SYSTEM_PROMPT_FORMATTER,
    output_schema=AgentOutputForLLM,
    output_key="final_output",
)

# ADK Canonical Pattern: SequentialAgent pipeline
agent = SequentialAgent(
    name="career_coach",
    sub_agents=[tool_agent, formatter_agent],
)
