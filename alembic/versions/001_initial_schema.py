"""001 — initial schema

Creates candidates and match_jobs tables with all required indexes.

Revision ID: 001_initial
Revises: —
Create Date: 2026-04-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Revision identifiers
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidates",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text, nullable=False, server_default="Candidate"),
        sa.Column("email", sa.Text, unique=True),
        sa.Column("structured_profile", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table(
        "match_jobs",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("candidate_id", UUID, sa.ForeignKey("candidates.id")),
        sa.Column("jd_input", sa.Text, nullable=False),
        sa.Column(
            "status", sa.Text, nullable=False,
            server_default="pending",
        ),
        sa.Column("result", JSONB),
        sa.Column("agent_trace", JSONB),
        sa.Column("error_detail", sa.Text),
        sa.Column("attempt_count", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="ck_match_jobs_status",
        ),
    )

    # Indexes for common query patterns
    op.create_index("idx_match_jobs_candidate", "match_jobs", ["candidate_id"])
    op.create_index("idx_match_jobs_status", "match_jobs", ["status"])
    op.create_index("idx_match_jobs_created", "match_jobs", ["created_at"])
    op.create_index("idx_match_jobs_status_created", "match_jobs", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_match_jobs_status_created", table_name="match_jobs")
    op.drop_index("idx_match_jobs_created", table_name="match_jobs")
    op.drop_index("idx_match_jobs_status", table_name="match_jobs")
    op.drop_index("idx_match_jobs_candidate", table_name="match_jobs")
    op.drop_table("match_jobs")
    op.drop_table("candidates")
