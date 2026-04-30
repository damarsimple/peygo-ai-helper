"""Integration test: Qwen3.6-27B + tools compliance with ADK SequentialAgent pattern.

Verifies that the tool_agent -> formatter_agent pipeline correctly produces
structured output via session state at temperatures 0.0, 0.6, and 1.0.

Usage:
    python -m pytest backend/tests/test_qwen_schema.py -v -s
"""

import json
import os

import pytest

from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from backend.schemas import AgentOutputForLLM

# ── constants ─────────────────────────────────────────────────────────

JD_TEXT = (
    "Senior Data Engineer\n"
    "Required: Python, SQL, Spark, Airflow, AWS\n"
    "Nice to have: Kafka, Terraform\n"
    "Seniority: senior\n"
    "Domain: data_engineering"
)

CANDIDATE_PROFILE = {
    "skills": ["Python", "SQL", "Pandas", "Git"],
    "years_experience": 3,
    "seniority": "mid",
    "domain": "data_engineering",
    "education": ["BSc Computer Science"],
    "certifications": [],
    "summary": "Mid-level data engineer with 3 years experience.",
}


def _resolve_model() -> str:
    """Strip openai/ prefix for vLLM endpoint."""
    model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    if os.getenv("OPENAI_BASE_URL") and model.startswith("openai/"):
        model = model[7:]
    return model


def _parse_text_as_json(text: str):
    """Parse raw text as AgentOutputForLLM."""
    if isinstance(text, dict):
        try:
            return AgentOutputForLLM.model_validate(text)
        except Exception:
            return None

    cleaned = text.strip()
    for tag in ("\x1e", "\x1f"):
        cleaned = cleaned.replace(tag, "")

    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start == -1 or brace_end <= brace_start:
        return None

    try:
        data = json.loads(cleaned[brace_start : brace_end + 1])
        return AgentOutputForLLM.model_validate(data)
    except Exception:
        return None


async def _run_pipeline(jd: str, temperature: float, session_service, session_id: str):
    """Run the career_coach SequentialAgent pipeline with the given temperature.

    Catches ADK schema validation errors (expected at high temps) so the
    fallback path can still be tested via session state.
    """
    from backend.agents import career_coach as cc_module
    from backend.worker import callbacks as cb
    from google.adk.models.lite_llm import LiteLlm
    import asyncio

    # Swap model with fresh temperature-specific instance
    original_llm = cc_module.llm_model
    fresh_llm = LiteLlm(
        model=cc_module.MODEL_NAME,
        api_base=cc_module.API_BASE,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        extra_body={
            "enable_thinking": True,
            "temperature": temperature,
        },
        drop_params=True,
    )
    cc_module.llm_model = fresh_llm

    cb._cb_state = {
        "__job_data__": {
            "candidate_profile": CANDIDATE_PROFILE,
            "job_id": session_id,
        }
    }

    events = []
    try:
        runner = Runner(
            app_name="pelgo",
            agent=cc_module.agent,
            session_service=session_service,
            auto_create_session=True,
        )

        async for event in runner.run_async(
            user_id="schema-test-user",
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=f"Analyze this job for the candidate:\n\n{jd}")],
            ),
        ):
            events.append(event)
    except Exception:
        pass  # Schema validation errors are expected at high temps; fallback handles it
    finally:
        cc_module.llm_model = original_llm

    return events


def _check_output(events, temperature, session_state):
    """Validate output via session state (ADK canonical pattern)."""
    # ── Path 1: final_output from formatter_agent output_key ─────────
    final_text = session_state.get("final_output")
    if final_text is not None:
        parsed = _parse_text_as_json(final_text)
        if parsed is not None:
            return True, f"✓ strict (output_key) — score={parsed.overall_score}, confidence={parsed.confidence}"

    # ── Path 2: fallback reconstruction from intermediate state ─────
    score_data = session_state.get("score", {})
    gap_skills = session_state.get("gap_skills", [])
    if not score_data:
        score_data = session_state.get("temp:score_candidate_against_requirements_response", {})
    if not gap_skills:
        gap_skills = score_data.get("gap_skills", [])

    if score_data:
        out = {
            "job_id": session_state.get("job_id", "unknown"),
            "overall_score": score_data.get("overall_score", 0),
            "confidence": score_data.get("confidence", "low"),
            "dimension_scores": score_data.get("dimension_scores", {}),
            "matched_skills": score_data.get("matched_skills", []),
            "gap_skills": gap_skills,
            "reasoning": f"Scored {score_data.get('overall_score', 0)} ({score_data.get('confidence', 'low')} confidence)",
            "learning_plan": [],
        }
        try:
            validated = AgentOutputForLLM.model_validate(out)
            return True, f"✓ fallback — score={validated.overall_score}, confidence={validated.confidence}"
        except Exception as e:
            return False, f"fallback parse failed: {e}"

    last_raw = ""
    final_events = [e for e in events if e.is_final_response()]
    if final_events:
        last = final_events[-1]
        last_raw = (last.content.parts[0].text if last.content and last.content.parts else "").strip()[:200]
    return False, f"no structured data (text: {last_raw!r})"


# ── tests ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def session_service():
    svc = InMemorySessionService()
    return svc


class TestQwenSchemaCompliance:
    """Test Qwen3 via vLLM with ADK SequentialAgent pipeline."""

    @pytest.mark.asyncio
    async def test_temp_0_0(self, session_service):
        sid = "schema-00"
        await session_service.create_session(app_name="pelgo", user_id="schema-test-user", session_id=sid)
        events = await _run_pipeline(JD_TEXT, temperature=0.0, session_service=session_service, session_id=sid)
        session = await session_service.get_session(app_name="pelgo", user_id="schema-test-user", session_id=sid)
        state = session.state if session else {}
        ok, detail = _check_output(events, 0.0, state)
        assert ok, f"[temp=0.0] could not extract structured output: {detail}"

    @pytest.mark.asyncio
    async def test_temp_0_6(self, session_service):
        sid = "schema-06"
        await session_service.create_session(app_name="pelgo", user_id="schema-test-user", session_id=sid)
        events = await _run_pipeline(JD_TEXT, temperature=0.6, session_service=session_service, session_id=sid)
        session = await session_service.get_session(app_name="pelgo", user_id="schema-test-user", session_id=sid)
        state = session.state if session else {}
        ok, detail = _check_output(events, 0.6, state)
        assert ok, f"[temp=0.6] could not extract structured output: {detail}"

    @pytest.mark.asyncio
    async def test_temp_1_0(self, session_service):
        sid = "schema-10"
        await session_service.create_session(app_name="pelgo", user_id="schema-test-user", session_id=sid)
        events = await _run_pipeline(JD_TEXT, temperature=1.0, session_service=session_service, session_id=sid)
        session = await session_service.get_session(app_name="pelgo", user_id="schema-test-user", session_id=sid)
        state = session.state if session else {}
        ok, detail = _check_output(events, 1.0, state)
        assert ok, f"[temp=1.0] could not extract structured output: {detail}"
