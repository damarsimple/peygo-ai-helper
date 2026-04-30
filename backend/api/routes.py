from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from uuid import uuid4, UUID
import json
import asyncio
from functools import partial
import structlog

from backend.schemas import CreateCandidateRequest, CreateMatchesRequest, ResumeProfile
import backend.db.engine
from backend.tools.llm_utils import (
    get_openai_client, build_json_schema,
    create_structured_request, parse_structured_response,
)

log = structlog.get_logger()

router = APIRouter()

MAX_JD_BATCH = 10
MAX_LIST_LIMIT = 100


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_jsonb_columns(row: dict, jsonb_keys: list[str]) -> dict:
    """Parse JSONB columns that asyncpg may return as strings.
    
    Logs a warning and leaves the value as-is if parsing fails,
    rather than silently corrupting the response or raising a 500.
    """
    for key in jsonb_keys:
        if key in row and isinstance(row[key], str):
            try:
                row[key] = json.loads(row[key])
            except json.JSONDecodeError:
                log.warning("jsonb_parse_failed", key=key, value_preview=row[key][:80])
    return row


def _validate_uuid(value: str, field: str) -> str:
    """Raise HTTP 400 if value is not a valid UUID string."""
    try:
        UUID(value)
    except ValueError:
        raise HTTPException(400, f"Invalid {field}: '{value}' is not a valid UUID")
    return value


# ── Candidate endpoints ───────────────────────────────────────────────────


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
    """Ingest candidate resume from PDF. Extracts text then parses a structured profile."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")

    # Run blocking pdfplumber in a thread pool so we don't block the event loop
    try:
        full_text = await asyncio.get_event_loop().run_in_executor(
            None, partial(_extract_pdf_text, content)
        )
    except Exception as e:
        log.warning("pdf_extraction_failed", filename=file.filename, error=str(e))
        raise HTTPException(400, f"PDF text extraction failed: {e}")

    if not full_text.strip():
        raise HTTPException(422, (
            "No readable text found in this PDF. "
            "It may be a scanned image — please upload a text-based PDF."
        ))

    prompt = f"""You are a technical recruiter parsing a candidate's resume.
Extract a COMPLETE, structured profile from the resume text below.

EXTRACTION RULES:
- skills: Extract ALL technical skills mentioned — languages, frameworks, tools, platforms,
  hardware, protocols, methodologies. Normalise aliases:
  • "C/C++" → list as BOTH "C" and "C++"
  • "Node" → "Node.js"
  • "Postgres" / "PG" → "PostgreSQL"
  • "K8s" → "Kubernetes"
  Do NOT deduplicate across aliases — list each normalised form separately.
- years_experience: Sum total years of professional engineering experience.
  If not explicitly stated, estimate from earliest to most recent role dates.
- seniority: Map years to one of: "junior" (<2yr), "mid" (2–5yr), "senior" (5–9yr), "lead" (9+yr).
  Override with explicit title if present (e.g. "Staff Engineer" → "lead").
- domain: Single slug describing primary discipline:
  "backend", "frontend", "fullstack", "embedded", "data_engineering",
  "devops", "mobile", "ml_ai", "security", or "unknown".
- education / certifications: Include all degrees and professional certs found.
- summary: 1–2 sentence characterisation of the candidate's background. Do not copy
  the resume's own summary verbatim — synthesise from the full text.

Return ONLY a valid JSON object — no markdown, no explanation:
{{
  "name": "Full Name",
  "email": "email@example.com or null",
  "skills": ["skill1", "skill2"],
  "years_experience": 4,
  "seniority": "mid",
  "domain": "backend",
  "education": ["BSc Computer Science, MIT 2018"],
  "certifications": ["AWS Solutions Architect"],
  "summary": "2-sentence background summary."
}}

RESUME TEXT:
{full_text[:6000]}
"""

    try:
        client = get_openai_client()
        schema = build_json_schema(ResumeProfile)
        payload = create_structured_request(
            messages=[{"role": "user", "content": prompt}],
            schema=schema,
            temperature=1   # Low — structured extraction, not creative generation
        )
        response = await client.chat.completions.create(**payload)
        raw_content = response.choices[0].message.content
        log.info(
            "llm_resume_parsed",
            filename=file.filename,
            raw_content=raw_content,
        )
        extracted = parse_structured_response(response)
        profile = ResumeProfile.model_validate(extracted)
    except Exception as e:
        log.error(
            "resume_parsing_failed",
            filename=file.filename,
            error=str(e),
            raw_content=getattr(response, "choices", [None])[0].message.content if 'response' in dir() else "no response",
        )
        raise HTTPException(422, f"Resume parsing failed: {e}")

    log.info(
        "candidate_pdf_ingested",
        filename=file.filename,
        skills_found=len(profile.skills),
        seniority=profile.seniority,
        domain=profile.domain,
    )

    return await _create_candidate_from_data(
        name=profile.name or "Candidate",
        email=profile.email or "",
        skills=profile.skills,
        years_experience=profile.years_experience,
        seniority=profile.seniority,
        domain=profile.domain,
        education=profile.education,
        certifications=profile.certifications,
        summary=profile.summary,
        raw_text=full_text,
    )


def _extract_pdf_text(content: bytes) -> str:
    """Blocking PDF text extraction — call via run_in_executor."""
    import pdfplumber
    import tempfile
    import os as _os

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(content)
            tmp.flush()

        full_text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            _os.unlink(tmp_path)


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

    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM candidates WHERE email = $1", email
            )
            new_id = str(uuid4())
            row = await conn.fetchrow(
                """INSERT INTO candidates (id, name, email, structured_profile, raw_text)
                   VALUES ($1, $2, $3, $4::jsonb, $5)
                   ON CONFLICT (email) DO UPDATE
                   SET structured_profile = EXCLUDED.structured_profile,
                       name               = EXCLUDED.name,
                       raw_text           = EXCLUDED.raw_text
                   RETURNING id""",
                new_id, name, email,
                structured.model_dump_json(), raw_text,
            )
    except Exception as e:
        log.error("candidate_db_write_failed", email=email, error=str(e))
        raise HTTPException(500, "Failed to save candidate — please try again")

    return {"id": row["id"], "status": "updated" if existing else "created"}


# ── Match endpoints ────────────────────────────────────────────────────────


@router.post("/api/v1/matches")
async def create_matches(request: CreateMatchesRequest):
    """Accept 1–10 JDs and enqueue one agent run per JD.
    
    Returns a list of job objects with initial status 'pending'.
    """
    if not request.jd_inputs:
        raise HTTPException(400, "jd_inputs must contain at least one job description")
    if len(request.jd_inputs) > MAX_JD_BATCH:
        raise HTTPException(
            400,
            f"Maximum {MAX_JD_BATCH} job descriptions per request; "
            f"got {len(request.jd_inputs)}"
        )

    results = []
    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            for jd in request.jd_inputs:
                job_id = str(uuid4())
                await conn.execute(
                    """INSERT INTO match_jobs (id, candidate_id, jd_input, status, attempt_count)
                       VALUES ($1, $2, $3, 'pending', 0)""",
                    job_id, request.candidate_id, jd,
                )
                results.append({"id": job_id, "status": "pending"})
    except Exception as e:
        log.error("match_enqueue_failed", candidate_id=request.candidate_id, error=str(e))
        raise HTTPException(500, "Failed to enqueue jobs — please try again")

    log.info("matches_enqueued", count=len(results), candidate_id=request.candidate_id)
    return results


@router.get("/api/v1/matches")
async def list_matches(
    status: str | None = None,
    candidate_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    """Paginated list of match jobs, filterable by status and candidate_id."""
    if candidate_id:
        _validate_uuid(candidate_id, "candidate_id")

    VALID_STATUSES = {"pending", "processing", "completed", "failed"}
    if status and status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of: {', '.join(sorted(VALID_STATUSES))}")

    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, candidate_id, jd_input, status, result,
                          created_at, agent_trace
                   FROM match_jobs
                   WHERE ($1::text IS NULL OR status = $1)
                     AND ($2::text IS NULL OR candidate_id::text = $2)
                   ORDER BY created_at DESC
                   LIMIT $3 OFFSET $4""",
                status, candidate_id, limit, offset,
            )
    except Exception as e:
        log.error("list_matches_failed", error=str(e))
        raise HTTPException(500, "Failed to fetch matches")

    return [_parse_jsonb_columns(dict(r), ["result", "agent_trace"]) for r in rows]


@router.get("/api/v1/matches/{job_id}")
async def get_match(job_id: str):
    """Return status and full structured output for one match job."""
    _validate_uuid(job_id, "job_id")

    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, candidate_id, jd_input, status, result, agent_trace,
                          error_detail, attempt_count
                   FROM match_jobs WHERE id = $1""",
                job_id,
            )
    except Exception as e:
        log.error("get_match_failed", job_id=job_id, error=str(e))
        raise HTTPException(500, "Failed to fetch match")

    if not row:
        raise HTTPException(404, f"Job '{job_id}' not found")

    return _parse_jsonb_columns(dict(row), ["result", "agent_trace"])


# ── Admin endpoints ───────────────────────────────────────────────────────


@router.post("/api/v1/admin/requeue/{job_id}")
async def requeue_job(job_id: str):
    """Admin: reset any job back to pending regardless of current status."""
    _validate_uuid(job_id, "job_id")

    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE match_jobs
                   SET status = 'pending',
                       attempt_count = 0,
                       error_detail  = NULL,
                       updated_at    = NOW()
                   WHERE id = $1::uuid""",
                job_id,
            )
    except Exception as e:
        log.error("requeue_failed", job_id=job_id, error=str(e))
        raise HTTPException(500, "Requeue failed")

    if result == "UPDATE 0":
        raise HTTPException(404, f"Job '{job_id}' not found")

    log.info("job_requeued", job_id=job_id)
    return {"id": job_id, "status": "pending"}


# ── Infrastructure endpoints ──────────────────────────────────────────────


@router.get("/health")
async def health_check():
    """Liveness + DB connectivity check."""
    try:
        async with backend.db.engine.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        log.error("health_check_failed", error=str(e))
        raise HTTPException(503, f"Database unreachable: {e}")
    return {"status": "healthy"}


@router.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "frontend", "index.html",
    )
    if not os.path.exists(path):
        raise HTTPException(404, "Frontend not found")
    return FileResponse(path)