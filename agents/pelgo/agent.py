"""ADK Web UI entry point for the Pelgo CareerCoach agent.

This file is discovered by `adk web agents` from the `agents/` directory.
ADK imports it as `pelgo.agent` where the working directory is the parent of `agents/`.
We add the project root to sys.path so the `backend` package is importable.
"""

import os
import sys
# __file__ = agents/pelgo/agent.py → parent of parent = project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("MODEL_NAME", os.getenv("MODEL_NAME", "gpt-3.5-turbo"))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "dummy-key"))
os.environ.setdefault("DATABASE_URL", os.getenv("DATABASE_URL", "postgresql://pelgo:pelgo@localhost:5432/pelgo"))
os.environ.setdefault("ADK_DATABASE_URL", os.getenv(
    "ADK_DATABASE_URL",
    "postgresql+asyncpg://pelgo:pelgo@localhost:5432/pelgo_adk",
))

from backend.agents.career_coach import agent as root_agent

# ADK web discovers the root agent via the 'agent' or 'root_agent' export
agent = root_agent

