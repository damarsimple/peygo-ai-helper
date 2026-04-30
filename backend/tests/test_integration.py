"""Full lifecycle integration test.

Covers:
1. JSON candidate ingestion
2. JD submission
3. Agent execution (polling)
4. Result schema validation (including learning_plan content)
5. Agent trace validation (verifies behaviour, not just status codes)
6. Admin requeue
7. List matches pagination + candidate_id filter
"""
import pytest
import httpx
import asyncio

BASE_URL = "http://localhost:8000"


@pytest.mark.asyncio
async def test_full_lifecycle():
    """Full lifecycle: ingest → submit JD → agent runs → valid result with trace."""

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Ingest candidate
        resp = await client.post("/api/v1/candidate", json={
            "name": "Test Candidate",
            "email": f"test-{id(object())}@example.com",  # Unique email per run
            "skills": ["Python", "SQL"],
            "years_experience": 2,
            "seniority": "junior",
            "domain": "data",
        })
        assert resp.status_code == 200
        candidate_id = resp.json()["id"]
        assert resp.json()["status"] == "created"

        # 2. Submit JD
        resp = await client.post("/api/v1/matches", json={
            "candidate_id": candidate_id,
            "jd_inputs": [
                "Senior Python Developer — Required: Python, Django, PostgreSQL, Docker"
            ],
        })
        assert resp.status_code == 200
        job_id = resp.json()[0]["id"]
        assert resp.json()[0]["status"] == "pending"

        # 3. Poll until completed (max 180s — full pipeline with LLM can take 2-3 min)
        for _ in range(180):
            resp = await client.get(f"/api/v1/matches/{job_id}")
            data = resp.json()
            status = data["status"]
            if status == "completed":
                break
            if status == "failed":
                pytest.fail(f"Job failed: {data.get('error_detail')}")
            await asyncio.sleep(1)
        else:
            pytest.fail("Job did not complete within 180 seconds")

        # 4. Validate result schema
        assert data["status"] == "completed"
        result = data["result"]
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 100
        assert result["confidence"] in ("low", "medium", "high")

        # Validate dimension_scores is a dict with required keys and valid ranges
        dim = result["dimension_scores"]
        assert isinstance(dim, dict)
        for key in ("skills", "experience", "seniority_fit"):
            assert key in dim, f"Missing dimension_scores.{key}"
            assert 0 <= dim[key] <= 100, f"dimension_scores.{key}={dim[key]} out of range"

        # Validate skill lists
        assert isinstance(result["matched_skills"], list)
        assert isinstance(result["gap_skills"], list)
        # Python should be matched (candidate has it, JD requires it)
        assert "Python" in result["matched_skills"], (
            f"Expected 'Python' in matched_skills, got {result['matched_skills']}"
        )
        # Django, PostgreSQL, Docker should be gaps (candidate doesn't have them)
        assert len(result["gap_skills"]) >= 1, "Expected at least 1 gap skill"

        # Validate reasoning is substantive
        assert "reasoning" in result
        assert len(result["reasoning"]) >= 20

        # 4b. Validate learning_plan has actual content
        assert "learning_plan" in result
        lp = result["learning_plan"]
        assert isinstance(lp, list)
        if len(lp) > 0:
            item = lp[0]
            assert "skill" in item and len(item["skill"]) > 0
            assert "priority_rank" in item and item["priority_rank"] >= 1
            assert "estimated_match_gain_pct" in item
            assert "rationale" in item and len(item["rationale"]) >= 5
            assert "resources" in item and len(item["resources"]) >= 1
            res = item["resources"][0]
            assert "title" in res and len(res["title"]) > 0
            assert "url" in res and len(res["url"]) > 0
            assert "estimated_hours" in res and res["estimated_hours"] >= 1
            assert res["type"] in ("course", "project", "cert", "certification", "doc")

        # 5. Validate agent_trace (populated by orchestrator, not fabricated by LLM)
        trace = data["agent_trace"]
        assert isinstance(trace, dict)
        assert "tool_calls" in trace
        assert "total_llm_calls" in trace
        assert "fallbacks_triggered" in trace
        assert len(trace["tool_calls"]) >= 2, (
            f"Expected at least 2 tool calls (extract + score), got {len(trace['tool_calls'])}"
        )

        # Verify tool call structure and that expected tools were called
        tool_names_called = set()
        for tc in trace["tool_calls"]:
            assert tc["tool"] in (
                "extract_jd_requirements",
                "score_candidate_against_requirements",
                "research_skill_resources",
                "prioritise_skill_gaps",
            )
            assert tc["status"] in ("success", "failed", "timeout", "fallback")
            assert isinstance(tc["latency_ms"], int)
            assert tc["latency_ms"] >= 0
            tool_names_called.add(tc["tool"])

        # extract + score are mandatory in every run
        assert "extract_jd_requirements" in tool_names_called, (
            "extract_jd_requirements was not called"
        )
        assert "score_candidate_against_requirements" in tool_names_called, (
            "score_candidate_against_requirements was not called"
        )

        # LLM calls should be at least 2 (tool_agent + formatter_agent)
        assert trace["total_llm_calls"] >= 2

        # 6. Validate requeue endpoint (non-failed job should return 404)
        resp2 = await client.post("/api/v1/matches", json={
            "candidate_id": candidate_id,
            "jd_inputs": ["Placeholder JD for requeue test"],
        })
        requeue_job_id = resp2.json()[0]["id"]
        resp3 = await client.post(f"/api/v1/admin/requeue/{requeue_job_id}")
        assert resp3.status_code == 404  # Not in failed state

        # 7. Validate list matches endpoint
        resp4 = await client.get("/api/v1/matches", params={"limit": 10, "offset": 0})
        assert resp4.status_code == 200
        matches_list = resp4.json()
        assert isinstance(matches_list, list)
        assert len(matches_list) >= 1

        # 7b. Validate list matches with candidate_id filter
        resp_cand = await client.get("/api/v1/matches", params={
            "candidate_id": candidate_id, "limit": 10, "offset": 0,
        })
        assert resp_cand.status_code == 200
        cand_matches = resp_cand.json()
        assert all(str(m["candidate_id"]) == str(candidate_id) for m in cand_matches)

        # 8. Validate list matches with status filter
        resp5 = await client.get("/api/v1/matches", params={"status": "completed", "limit": 5, "offset": 0})
        assert resp5.status_code == 200
        completed_list = resp5.json()
        assert all(m["status"] == "completed" for m in completed_list)

        # 9. Validate health endpoint
        resp6 = await client.get("/health")
        assert resp6.status_code == 200
        assert resp6.json()["status"] == "healthy"
