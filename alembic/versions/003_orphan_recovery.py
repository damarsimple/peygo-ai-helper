"""003 — orphan recovery: add worker_id, reset stuck jobs

Revision ID: 003_orphan_recovery
Revises: 002_add_raw_text
Create Date: 2026-04-30

Adds worker_id to match_jobs so we can track which worker owns a job.
Resets any jobs orphaned in 'processing' state back to 'pending'.
"""
from alembic import op
import sqlalchemy as sa

revision = "003_orphan_recovery"
down_revision = "002_add_raw_text"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Track which worker process claimed this job (pid / worker identity)
    op.add_column("match_jobs", sa.Column("worker_id", sa.Text, nullable=True))

    # Add processing_started_at so we can compute stale durations precisely
    op.add_column("match_jobs", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True))

    # Reset any jobs currently stuck in 'processing' back to 'pending'
    op.execute("""
        UPDATE match_jobs
        SET status = 'pending',
            error_detail = 'Reset by migration 003: worker died mid-execution',
            updated_at = NOW()
        WHERE status = 'processing'
    """)


def downgrade() -> None:
    op.drop_column("match_jobs", "processing_started_at")
    op.drop_column("match_jobs", "worker_id")
