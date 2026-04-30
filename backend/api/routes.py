from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from uuid import uuid4
import json
from backend.schemas import CreateCandidateRequest, CreateMatchesRequest, ResumeProfile
import backend.db.engine
import structlog

log = structlog.get_logger()

router = APIRouter()


def _parse_jsonb_columns(row: dict, jsonb_keys: list[str]) -> dict:
    """Parse JSONB columns that asyncpg may return as strings."""
    for key in jsonb_keys:
        if key in row and isinstance(row[key], str):
            row[key] = json.loads(row[key])
    return row


@router.post("/api/v1/candidate")
async def ingest_candidate_json(request: CreateCandidateRequest):
    """Ingest candidate profile as structured JSON."""
    return await _create_candidate_from_data(
        name=request.name,
        email=request.email,
        skills=request.skills,
        years_experience=request.years_experience,
        seniority=request.seniority,
        domain=request.domain,
        education=request.education,
        certifications=request.certifications,
        summary=request.summary,
    )


@router.post("/api/v1/candidate/pdf")
async def ingest_candidate_pdf(file: UploadFile = File(...)):
    """Ingest candidate resume from PDF. Extract text and parse structured profile."""
    import pdfplumber
    import tempfile
    import os as _os

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(content)
            tmp.flush()

        with pdfplumber.open(tmp_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        raise HTTPException(400, f"PDF parsing failed: {str(e)}")
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            _os.unlink(tmp_path)

    # Extract structured data from PDF text using LLM
    from backend.tools.llm_utils import get_openai_client, build_json_schema, create_structured_request, parse_structured_response

    prompt = f"""Extract the candidate's COMPLETE profile from this resume text.
CRITICAL: Extract ALL skills mentioned, including:
- Programming languages (C/C++, C++, C, Python, Java, etc.)
- Hardware skills (Arduino, Raspberry Pi, I2C, MAVLink, UAV Systems, IoT)
- Frameworks, tools, databases, platforms

Resume text:
{full_text[:6000]}
"""

    try:
        client = get_openai_client()
        schema = build_json_schema(ResumeProfile)
        payload = create_structured_request(
            messages=[{"role": "user", "content": prompt}],
            schema=schema,
            temperature=0.1,
        )
        response = await client.chat.completions.create(**payload)
        extracted = parse_structured_response(response)
        profile = ResumeProfile.model_validate(extracted)
    except Exception as e:
        log.error("resume_parsing_failed", error=str(e))
        raise HTTPException(400, f"Resume parsing failed: {str(e)}")

    log.info(
        "candidate_pdf_ingested",
        filename=file.filename,
        skills=len(profile.skills),
    )

    return await _create_candidate_from_data(
        name=profile.name or "Candidate",
        email=profile.email,
        skills=profile.skills,
        years_experience=profile.years_experience,
        seniority=profile.seniority,
        domain=profile.domain,
        education=profile.education,
        certifications=profile.certifications,
        summary=profile.summary,
        raw_text=full_text,
    )


async def _create_candidate_from_data(
    name: str, email: str, skills: list[str],
    years_experience: int, seniority: str, domain: str,
    education: list[str], certifications: list[str], summary: str,
    raw_text: str = "",
) -> dict:
    """Core candidate creation logic shared by JSON and PDF endpoints."""
    from backend.schemas import CandidateProfile
    structured = CandidateProfile(
        skills=skills,
        years_experience=years_experience,
        seniority=seniority,
        domain=domain,
        education=education,
        certifications=certifications,
        summary=summary,
    )
    async with backend.db.engine.db_pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT id FROM candidates WHERE email = $1", email
        )
        candidate_id = str(uuid4())
        row = await conn.fetchrow(
            """INSERT INTO candidates (id, name, email, structured_profile, raw_text)
               VALUES ($1, $2, $3, $4::jsonb, $5)
               ON CONFLICT (email) DO UPDATE 
               SET structured_profile = EXCLUDED.structured_profile,
                   name = EXCLUDED.name,
                   raw_text = EXCLUDED.raw_text
               RETURNING id""",
            candidate_id, name, email,
            structured.model_dump_json(), raw_text
        )
    return {"id": row["id"], "status": "updated" if existing else "created"}


@router.post("/api/v1/matches")
async def create_matches(request: CreateMatchesRequest):
    """Accept ≤10 JDs. Enqueue one agent run per JD."""
    results = []
    async with backend.db.engine.db_pool.acquire() as conn:
        for jd in request.jd_inputs:
            job_id = str(uuid4())
            await conn.execute(
                """INSERT INTO match_jobs (id, candidate_id, jd_input, status, attempt_count)
                   VALUES ($1, $2, $3, 'pending', 0)""",
                job_id, request.candidate_id, jd
            )
            results.append({"id": job_id, "status": "pending"})
    return results


@router.get("/api/v1/matches")
async def list_matches(
    status: str | None = None,
    candidate_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    """Paginated list, filterable by status and candidate_id."""
    async with backend.db.engine.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, candidate_id, jd_input, status, result,
                      created_at, agent_trace
               FROM match_jobs
               WHERE ($1::text IS NULL OR status = $1)
                 AND ($2::text IS NULL OR candidate_id::text = $2)
               ORDER BY created_at DESC
               LIMIT $3 OFFSET $4""",
            status, candidate_id, limit, offset
        )
    return [_parse_jsonb_columns(dict(r), ["result", "agent_trace"]) for r in rows]


@router.get("/api/v1/matches/{job_id}")
async def get_match(job_id: str):
    """Return status + full structured output for one job."""
    async with backend.db.engine.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, candidate_id, jd_input, status, result, agent_trace, "
            "error_detail, attempt_count FROM match_jobs WHERE id = $1", job_id
        )
    if not row:
        raise HTTPException(404, "Job not found")
    return _parse_jsonb_columns(dict(row), ["result", "agent_trace"])


@router.post("/api/v1/admin/requeue/{job_id}")
async def requeue_job(job_id: str):
    """Admin: reset a failed job back to pending."""
    async with backend.db.engine.db_pool.acquire() as conn:
        result = await conn.execute(
            """UPDATE match_jobs
               SET status='pending', attempt_count=0, error_detail=NULL, updated_at=NOW()
               WHERE id=$1 AND status='failed'""",
            job_id
        )
        if result == "UPDATE 0":
            raise HTTPException(404, "Job not found or not in failed state")
    return {"id": job_id, "status": "pending"}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(503, f"Unhealthy: {str(e)}")


@router.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "index.html")
    return FileResponse(path)
