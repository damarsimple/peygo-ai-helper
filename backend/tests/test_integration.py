"""
Integration test: Full lifecycle — ingest candidate → submit JD → agent runs → result returned.

Requires:
- Running API server (localhost:8000)
- Running worker with access to vLLM
- PostgreSQL database
- Example files: example/my-resume.pdf, example/target-job.txt

Run with: pytest backend/tests/test_integration.py -v -s
"""
import asyncio
import os
import time

import pytest
import httpx

pytest_plugins = ["pytest_asyncio"]

API_BASE = "http://localhost:8000"
EXAMPLE_DIR = "/home/damar/pelgo-ai/example"


@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_full_lifecycle():
    """Test: ingest PDF → submit JD → poll until complete → verify agent_trace."""
    candidate_id = None
    job_id = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        # ── Step 1: Ingest candidate from PDF ──
        pdf_path = os.path.join(EXAMPLE_DIR, "my-resume.pdf")
        assert os.path.exists(pdf_path), f"PDF not found: {pdf_path}"

        with open(pdf_path, "rb") as f:
            resp = await client.post(
                f"{API_BASE}/api/v1/candidate/pdf",
                files={"file": ("resume.pdf", f, "application/pdf")},
            )

        assert resp.status_code == 200, f"Candidate creation failed: {resp.text}"
        data = resp.json()
        candidate_id = data["id"]
        print(f"✓ Candidate created: {candidate_id}")

        # ── Step 2: Submit job description ──
        jd_path = os.path.join(EXAMPLE_DIR, "target-job.txt")
        assert os.path.exists(jd_path), f"JD file not found: {jd_path}"

        with open(jd_path, "r") as f:
            jd_text = f.read().strip()

        resp = await client.post(
            f"{API_BASE}/api/v1/matches",
            json={"candidate_id": candidate_id, "jd_inputs": [jd_text]},
        )

        assert resp.status_code == 200, f"Match submission failed: {resp.text}"
        jobs = resp.json()
        assert len(jobs) == 1
        job_id = jobs[0]["id"]
        assert jobs[0]["status"] == "pending"
        print(f"✓ Job enqueued: {job_id}")

    # ── Step 3: Poll until completion (with timeout) ──
    max_wait = 300  # 5 minutes
    poll_interval = 10
    elapsed = 0

    print(f"⏳ Waiting for job {job_id} to complete...")
    while elapsed < max_wait:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{API_BASE}/api/v1/matches/{job_id}")
            assert resp.status_code == 200, f"Failed to get job: {resp.text}"
            job = resp.json()
            print(f"  [{elapsed}s] Status: {job['status']}")

            if job["status"] == "completed":
                break
            elif job["status"] == "failed":
                error_detail = job.get("error_detail", "Unknown error")
                pytest.fail(f"Job failed: {error_detail}")

    assert job["status"] == "completed", f"Job did not complete within {max_wait}s"

    # ── Step 4: Verify structured output ──
    result = job.get("result", {})
    assert result, "No result field in completed job"
    assert "overall_score" in result, "Missing overall_score"
    assert 0 <= result["overall_score"] <= 100, f"Score out of range: {result['overall_score']}"
    assert result["confidence"] in ["low", "medium", "high"], f"Invalid confidence: {result['confidence']}"
    assert "dimension_scores" in result, "Missing dimension_scores"
    assert "matched_skills" in result, "Missing matched_skills"
    assert "gap_skills" in result, "Missing gap_skills"
    assert "reasoning" in result, "Missing reasoning"
    assert "learning_plan" in result, "Missing learning_plan"

    print(f"✓ Score: {result['overall_score']} ({result['confidence']} confidence)")
    print(f"✓ Matched skills: {result['matched_skills']}")
    print(f"✓ Gap skills: {result['gap_skills']}")

    # ── Step 5: Verify agent_trace (orchestrator-populated, not fabricated) ──
    trace = result.get("agent_trace")
    assert trace, "Missing agent_trace"
    assert "tool_calls" in trace, "Missing tool_calls in agent_trace"
    assert len(trace["tool_calls"]) > 0, "No tool calls recorded"
    assert "total_llm_calls" in trace, "Missing total_llm_calls"

    for tc in trace["tool_calls"]:
        assert "tool" in tc, "Tool call missing 'tool' field"
        assert "status" in tc, "Tool call missing 'status' field"
        assert "latency_ms" in tc, "Tool call missing 'latency_ms' field"
        print(f"  - {tc['tool']}: {tc['status']} ({tc['latency_ms']}ms)")

    print(f"✓ agent_trace verified: {trace['total_llm_calls']} LLM calls, {len(trace['tool_calls'])} tools")

    # ── Step 6: Verify learning plan has resources ──
    if result["learning_plan"]:
        for lp in result["learning_plan"]:
            assert "skill" in lp, "Learning plan missing skill"
            assert "priority_rank" in lp, "Learning plan missing priority_rank"
            assert "estimated_match_gain_pct" in lp, "Learning plan missing estimated_match_gain_pct"
            assert "resources" in lp, "Learning plan missing resources"
            if lp["resources"]:
                for res in lp["resources"]:
                    assert "title" in res, "Resource missing title"
                    assert "url" in res, "Resource missing url"
                    assert "estimated_hours" in res, "Resource missing estimated_hours"
        print(f"✓ Learning plan verified: {len(result['learning_plan'])} items")

    # ── Step 7: Verify list endpoint works ──
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{API_BASE}/api/v1/matches?candidate_id={candidate_id}&limit=10")
        assert resp.status_code == 200
        matches = resp.json()
        match_ids = [m["id"] for m in matches]
        assert job_id in match_ids, "Job not found in list endpoint"
        print(f"✓ List endpoint verified: {len(matches)} matches for candidate")

    print("\n✅ Full integration test passed!")


if __name__ == "__main__":
    asyncio.run(test_full_lifecycle())