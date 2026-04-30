"""Database seed script: ensures schema exists and seeds sample data.

Run standalone:
    python -m backend.db.seed

Or via docker-compose (auto-run at container boot).

Note: Schema is also managed by Alembic migrations. The CREATE TABLE IF NOT EXISTS
statements here are a safety net for fresh databases where Alembic hasn't been run.
In production, use `alembic upgrade head` as the canonical migration path.
"""
import asyncpg
import json
import asyncio
import os


async def ensure_schema(conn):
    """Create tables if they don't exist (safety net for fresh databases)."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL DEFAULT 'Candidate',
            email TEXT UNIQUE,
            structured_profile JSONB NOT NULL,
            raw_text TEXT DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS match_jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            candidate_id UUID REFERENCES candidates(id),
            jd_input TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
            worker_id TEXT,
            processing_started_at TIMESTAMPTZ,
            result JSONB,
            agent_trace JSONB,
            error_detail TEXT,
            attempt_count INT DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ
        );
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_match_jobs_candidate
            ON match_jobs(candidate_id);
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_match_jobs_status
            ON match_jobs(status);
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_match_jobs_created
            ON match_jobs(created_at);
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_match_jobs_status_created
            ON match_jobs(status, created_at);
    """)


async def insert_seed_data(conn):
    """Insert sample candidates only. match_jobs are created via the API."""
    c1_profile = {
        "skills": ["Python", "SQL", "Pandas", "Git"],
        "years_experience": 3,
        "seniority": "mid",
        "domain": "data_engineering",
        "education": ["BSc Computer Science"],
        "certifications": [],
        "summary": "Mid-level data engineer with 3 years experience.",
    }
    await conn.execute(
        """INSERT INTO candidates (id, name, email, structured_profile)
           VALUES ('a0000000-0000-0000-0000-000000000001',
                   'Alice', 'alice@test.com', $1::jsonb)
           ON CONFLICT DO NOTHING""",
        json.dumps(c1_profile)
    )

    c2_profile = {
        "skills": ["Java", "Spring", "AWS", "Docker", "Kubernetes"],
        "years_experience": 6,
        "seniority": "senior",
        "domain": "software_engineering",
        "education": ["MSc Software Engineering"],
        "certifications": ["AWS Solutions Architect"],
        "summary": "Senior software engineer, cloud-native specialist.",
    }
    await conn.execute(
        """INSERT INTO candidates (id, name, email, structured_profile)
           VALUES ('a0000000-0000-0000-0000-000000000002',
                   'Bob', 'bob@test.com', $1::jsonb)
           ON CONFLICT DO NOTHING""",
        json.dumps(c2_profile)
    )


async def seed_database(dsn: str = None):
    """Seed: schema creation + sample candidates."""
    if dsn is None:
        dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://pelgo:pelgo@localhost:5432/pelgo",
        )

    pool = await asyncpg.create_pool(dsn=dsn)

    async with pool.acquire() as conn:
        await ensure_schema(conn)
        await insert_seed_data(conn)

    await pool.close()
    print("Seed data inserted successfully.")


if __name__ == "__main__":
    asyncio.run(seed_database())
