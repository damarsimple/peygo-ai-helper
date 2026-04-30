import asyncio
import json
import structlog
import asyncpg
import os
import signal
import time
import uuid
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from backend.agents.career_coach import agent
from backend.worker.job_runner import JobRunner

# ── Structlog configuration ──────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if not os.getenv("PRETTY_LOGS")
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Worker identity for multi-worker observability
WORKER_ID = os.getenv("WORKER_ID", str(uuid.uuid4())[:8])
log = structlog.get_logger().bind(worker_id=WORKER_ID)

# ── Orphan recovery thresholds ───────────────────────────────────────────
ORPHAN_RECOVERY_INTERVAL = 60  # seconds between cleanup sweeps
JOB_STALE_THRESHOLD = 360      # seconds (6 min) — reset jobs stuck longer than this


async def _orphan_recovery(db_pool, stale_threshold: int, worker_id: str) -> int:
    """Reset jobs stuck in 'processing' for longer than stale_threshold seconds.

    Returns the number of jobs recovered.
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetch(f"""
            UPDATE match_jobs
               SET status = 'pending',
                   worker_id = NULL,
                   processing_started_at = NULL,
                   error_detail = 'Worker died — recovered by orphan sweep',
                   updated_at = NOW()
               WHERE status = 'processing'
                 AND processing_started_at < NOW() - INTERVAL '{stale_threshold} seconds'
               RETURNING id
        """)
    return len(result)


async def worker_loop():
    """Out-of-process worker. Scales to 2+ workers without duplicates.

    Features:
    - Graceful shutdown on SIGTERM/SIGINT (docker stop)
    - Race-condition-safe job claiming via FOR UPDATE SKIP LOCKED
    - 3-retry dead-letter with partial agent_trace
    - Per-worker identity in all log lines
    - Orphan recovery: resets jobs abandoned by crashed workers
    """
    db_url = os.getenv("DATABASE_URL", "postgresql://pelgo:pelgo@postgres:5432/pelgo")
    adk_db_url = os.getenv("ADK_DATABASE_URL", "postgresql+asyncpg://pelgo:pelgo@postgres:5432/pelgo_adk")

    db_pool = await asyncpg.create_pool(dsn=db_url, min_size=2, max_size=10)

    session_service = DatabaseSessionService(db_url=adk_db_url)
    runner = Runner(
        app_name="pelgo",
        agent=agent,
        session_service=session_service,
        auto_create_session=True,
    )

    log.info("adk_db_init", message="Pre-initializing ADK database tables")
    try:
        await session_service.list_sessions(app_name="pelgo", user_id="init")
    except Exception as e:
        log.warning("adk_db_init_warning", error=str(e)[:200])
    log.info("adk_db_ready", message="ADK database tables initialized")

    job_runner = JobRunner(runner, session_service, db_pool)

    max_retries = 3
    poll_interval = 2

    # ── Graceful shutdown ─────────────────────────────────────────────
    shutdown_event = asyncio.Event()

    def _signal_handler():
        log.info("worker_shutdown_requested")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    log.info(
        "worker_started",
        db_url=db_url.replace("://pelgo:pelgo@", "://***@"),
        orphan_recovery_interval=ORPHAN_RECOVERY_INTERVAL,
        job_stale_threshold=JOB_STALE_THRESHOLD,
    )

    # Track last orphan cleanup
    last_cleanup = time.monotonic()

    while not shutdown_event.is_set():
        # ── Orphan recovery sweep ───────────────────────────────────
        now = time.monotonic()
        if now - last_cleanup >= ORPHAN_RECOVERY_INTERVAL:
            recovered = await _orphan_recovery(db_pool, JOB_STALE_THRESHOLD, WORKER_ID)
            if recovered:
                log.info("orphan_recovery", recovered=recovered)
            last_cleanup = now

        # ── Claim a pending job (with worker_id tracking) ─────────
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """WITH candidate_job AS (
                     SELECT id, candidate_id, jd_input
                     FROM match_jobs
                     WHERE status = 'pending'
                       AND attempt_count < $1
                     ORDER BY created_at ASC
                     LIMIT 1
                     FOR UPDATE SKIP LOCKED
                   )
                   UPDATE match_jobs
                   SET status = 'processing',
                       worker_id = $2,
                       processing_started_at = NOW(),
                       updated_at = NOW(),
                       attempt_count = attempt_count + 1
                   FROM candidate_job
                   WHERE match_jobs.id = candidate_job.id
                   RETURNING match_jobs.id,
                             match_jobs.candidate_id,
                             match_jobs.jd_input,
                             match_jobs.attempt_count""",
                max_retries, WORKER_ID
            )

        if not row:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
            continue

        job_id = row["id"]
        candidate_id = row["candidate_id"]
        jd_input = row["jd_input"]
        attempt_count = row["attempt_count"]

        log.info(
            "job_claimed",
            job_id=job_id,
            candidate_id=candidate_id,
            attempt=attempt_count,
            status="processing",
        )

        result = None
        try:
            result = await asyncio.wait_for(
                job_runner.run_job(job_id, candidate_id, jd_input),
                timeout=300,
            )

            result_json = result.model_dump_json()
            trace_json = result.agent_trace.model_dump_json()

            async with db_pool.acquire() as conn:
                await conn.execute(
                    """UPDATE match_jobs SET
                       status = 'completed',
                       result = $1::jsonb,
                       agent_trace = $2::jsonb,
                       updated_at = NOW()
                       WHERE id = $3""",
                    result_json, trace_json, job_id
                )

            log.info(
                "job_completed",
                job_id=job_id,
                status="completed",
                score=result.overall_score,
                confidence=result.confidence,
            )

        except asyncio.TimeoutError:
            error_msg = "Agent run exceeded 300s timeout"
            partial_trace = {
                "tool_calls": [], "total_llm_calls": 0, "fallbacks_triggered": 0,
                "total_input_tokens": 0, "total_output_tokens": 0, "elapsed_ms": 0
            }

            async with db_pool.acquire() as conn:
                if attempt_count >= max_retries:
                    await conn.execute(
                        """UPDATE match_jobs SET
                           status = 'failed',
                           error_detail = $1,
                           agent_trace = $2::jsonb,
                           updated_at = NOW()
                           WHERE id = $3""",
                        error_msg, json.dumps(partial_trace), job_id
                    )
                    log.error(
                        "job_failed_timeout",
                        job_id=job_id,
                        status="failed",
                        attempts=attempt_count,
                        reason=error_msg,
                    )
                else:
                    await conn.execute(
                        """UPDATE match_jobs SET
                           status = 'pending',
                           error_detail = $1,
                           updated_at = NOW()
                           WHERE id = $2""",
                        error_msg, job_id
                    )
                    log.warning(
                        "job_retry_timeout",
                        job_id=job_id,
                        status="pending",
                        attempt=attempt_count,
                        reason=error_msg,
                    )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error("job_exception_traceback", job_id=job_id, traceback=tb)
            error_msg = str(e)

            partial_trace = {
                "tool_calls": [], "total_llm_calls": 0, "fallbacks_triggered": 0,
                "total_input_tokens": 0, "total_output_tokens": 0, "elapsed_ms": 0
            }
            if result and hasattr(result, "agent_trace") and result.agent_trace:
                partial_trace = result.agent_trace.model_dump()

            async with db_pool.acquire() as conn:
                if attempt_count >= max_retries:
                    await conn.execute(
                        """UPDATE match_jobs SET
                           status = 'failed',
                           error_detail = $1,
                           agent_trace = $2::jsonb,
                           updated_at = NOW()
                           WHERE id = $3""",
                        error_msg, json.dumps(partial_trace), job_id
                    )
                    log.error(
                        "job_failed",
                        job_id=job_id,
                        status="failed",
                        attempts=attempt_count,
                        error=error_msg[:500],
                    )
                else:
                    await conn.execute(
                        """UPDATE match_jobs SET
                           status = 'pending',
                           error_detail = $1,
                           updated_at = NOW()
                           WHERE id = $2""",
                        error_msg, job_id
                    )
                    log.warning(
                        "job_retry",
                        job_id=job_id,
                        status="pending",
                        attempt=attempt_count,
                        error=error_msg[:500],
                    )

        await asyncio.sleep(0.5)

    # ── Cleanup ───────────────────────────────────────────────────────
    log.info("worker_shutting_down")
    await db_pool.close()
    log.info("worker_stopped")


if __name__ == "__main__":
    asyncio.run(worker_loop())
