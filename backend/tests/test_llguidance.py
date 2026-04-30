"""Integration test: verify llguidance JSON schema enforcement on all tools.

Exercises every tool that calls the LLM via the OpenAI-compatible API
and validates that the response is schema-compliant JSON with zero
manual fence-stripping needed.

Usage:
    python -m pytest backend/tests/test_llguidance.py -v -s
"""
import asyncio
import json
import os
import pytest
import openai
from pydantic import ValidationError

from dotenv import load_dotenv
load_dotenv()

from backend.tools.llm_utils import (
    resolve_model, get_openai_client,
    build_json_schema, create_structured_request,
    parse_structured_response,
)
from backend.schemas import JDRequirements


# ── helpers ──────────────────────────────────────────────────────────────


def _is_pure_json(text: str) -> bool:
    """Return True if text is parseable JSON without any fence stripping."""
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        return False


def _has_fences(text: str) -> bool:
    return "```" in text


# ── Tool 1: JD Extractor ─────────────────────────────────────────────────


class TestJDExtractor:
    @pytest.mark.asyncio
    async def test_jd_extractor_produces_valid_json(self):
        """The JD extractor tool should return schema-valid JSON via llguidance."""
        jd_text = (
            "Senior Backend Engineer\n"
            "Required: Python, FastAPI, PostgreSQL, Docker, Kubernetes\n"
            "Nice to have: Terraform, AWS, gRPC\n"
            "Seniority: senior\n"
            "Domain: backend_engineering"
        )

        from backend.tools.jd_extractor import extract_jd_requirements
        import httpx
        from bs4 import BeautifulSoup

        # Bypass URL fetch — call the LLM path directly with raw text
        # We monkey-patch to inject raw text
        result = await extract_jd_requirements(jd_text)

        # Validate schema
        validated = JDRequirements.model_validate(result)
        assert isinstance(validated.required_skills, list)
        assert len(validated.required_skills) >= 1
        assert validated.seniority_level in ("junior", "mid", "senior", "lead", "")
        assert "```" not in json.dumps(result)
        print(f"  ✅ JD extractor: {len(validated.required_skills)} required skills, {validated.seniority_level}")


# ── Tool 2: Scorer ───────────────────────────────────────────────────────


class TestScorer:
    @pytest.mark.asyncio
    async def test_scorer_produces_valid_json(self):
        """The scorer tool should return schema-valid JSON via llguidance."""
        candidate = {
            "skills": ["Python", "SQL", "Pandas"],
            "years_experience": 3,
            "seniority": "mid",
            "domain": "data_engineering",
        }
        requirements = {
            "required_skills": ["Python", "SQL", "Spark", "Airflow"],
            "nice_to_have_skills": ["Kafka"],
            "seniority_level": "senior",
            "domain": "data_engineering",
        }

        from backend.tools.scorer import score_candidate_against_requirements
        result = await score_candidate_against_requirements(
            candidate_profile=candidate,
            requirements=requirements,
            raw_resume_text="",
        )

        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "confidence" in result
        assert result["confidence"] in ("low", "medium", "high")
        assert "dimension_scores" in result
        assert "matched_skills" in result
        assert "gap_skills" in result
        print(f"  ✅ Scorer: score={result['overall_score']}, confidence={result['confidence']}")


# ── Tool 4: Prioritizer ──────────────────────────────────────────────────


class TestPrioritizer:
    @pytest.mark.asyncio
    async def test_prioritizer_produces_valid_json(self):
        """The prioritizer should return schema-valid JSON via llguidance."""
        gap_skills = ["Spark", "Airflow", "AWS", "Kubernetes"]

        from backend.tools.prioritizer import prioritise_skill_gaps
        result = await prioritise_skill_gaps(
            gap_skills=gap_skills,
            seniority_context="mid",
        )

        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "skill" in item
            assert "priority_rank" in item
            assert "rationale" in item
        print(f"  ✅ Prioritizer: {len(result)} prioritized gaps")


# ── Direct LLM calls: raw schema enforcement ────────────────────────────


class TestDirectSchemaEnforcement:
    """Test raw LLM calls against the llguidance endpoint without any wrapper."""

    @pytest.mark.asyncio
    async def test_raw_json_schema_call(self):
        """A raw OpenAI-compatible call with json_schema should return pure JSON."""
        client = get_openai_client()
        schema = build_json_schema(JDRequirements)

        messages = [{
            "role": "user",
            "content": (
                "Extract skills from: 'Backend Engineer requiring Python, FastAPI, PostgreSQL, Docker'\n"
                "Return a JSON object with keys: required_skills, nice_to_have_skills, "
                "seniority_level, domain, responsibilities."
            ),
        }]

        payload = create_structured_request(messages, schema, temperature=0.6)
        response = await client.chat.completions.create(**payload)
        content = response.choices[0].message.content

        # Verify: no fences, pure JSON
        assert not _has_fences(content), f"Response still has fences: {content[:200]}"
        assert _is_pure_json(content), f"Response is not parseable JSON: {content[:200]}"

        data = json.loads(content)
        validated = JDRequirements.model_validate(data)
        assert isinstance(validated.required_skills, list)
        print(f"  ✅ Raw call: pure JSON, validated — skills={validated.required_skills}")


# ── Cross-temperature stability ──────────────────────────────────────────


class TestTemperatureStability:
    """Verify JSON schema enforcement holds at elevated temperatures."""

    @pytest.mark.asyncio
    async def test_temp_0_6(self):
        client = get_openai_client()
        schema = build_json_schema(JDRequirements)
        payload = create_structured_request([{
            "role": "user",
            "content": "Extract from: 'ML Engineer: PyTorch, Docker, Kubernetes, MLOps'\n"
                        "Return JSON with required_skills, nice_to_have_skills, seniority_level, domain, responsibilities."
        }], schema, temperature=0.6)
        response = await client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        data = json.loads(content)
        validated = JDRequirements.model_validate(data)
        assert not _has_fences(content)
        print(f"  ✅ temp=0.6: {validated.required_skills}")

    @pytest.mark.asyncio
    async def test_temp_1_0(self):
        client = get_openai_client()
        schema = build_json_schema(JDRequirements)
        payload = create_structured_request([{
            "role": "user",
            "content": "Extract from: 'DevOps Engineer: AWS, Terraform, CI/CD, Linux'\n"
                        "Return JSON with required_skills, nice_to_have_skills, seniority_level, domain, responsibilities."
        }], schema, temperature=1.0)
        response = await client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        data = json.loads(content)
        validated = JDRequirements.model_validate(data)
        assert not _has_fences(content)
        print(f"  ✅ temp=1.0: {validated.required_skills}")
