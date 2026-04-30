from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
from backend.api.routes import router
from backend.db.engine import db_pool

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Pelgo Career Intelligence")

# Mount frontend assets
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found", "path": index_path}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    from backend.db.engine import init_db
    await init_db()


@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()