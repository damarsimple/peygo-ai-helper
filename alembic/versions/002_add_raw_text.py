"""002 — add raw_text column to candidates

Revision ID: 002_add_raw_text
Revises: 001_initial
Create Date: 2026-04-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TEXT

# Revision identifiers
revision = "002_add_raw_text"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("candidates", sa.Column("raw_text", TEXT, server_default=""))


def downgrade() -> None:
    op.drop_column("candidates", "raw_text")
