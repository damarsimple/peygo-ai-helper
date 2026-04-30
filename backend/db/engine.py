import asyncpg
import os

db_pool = None

async def init_db():
    """Initialize database pool."""
    global db_pool
    db_url = os.getenv("DATABASE_URL", "postgresql://pelgo:pelgo@postgres:5432/pelgo")
    db_pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=2, max_size=10,
    )