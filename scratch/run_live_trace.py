import httpx
import asyncio
import json
import os
import subprocess
import time

BASE_URL = "http://localhost:8000"
ARTIFACT_DIR = "/home/damar/pelgo-ai/artifacts/live_run_trace"

async def run_live_trace():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Ingest
        email = f"live-trace-{int(time.time())}@example.com"
        print(f"Ingesting candidate: {email}")
        resp = await client.post("/api/v1/candidate", json={
            "name": "Live Trace Candidate",
            "email": email,
            "skills": ["Python", "Docker", "SQL"],
            "years_experience": 4,
            "seniority": "mid",
            "domain": "backend",
        })
        candidate_id = resp.json()["id"]

        # 2. Submit JD
        jd = "Senior Cloud Engineer. Required: Kubernetes, Terraform, Go, AWS. Nice to have: Python. Seniority: senior."
        print("Submitting job...")
        resp = await client.post("/api/v1/matches", json={
            "candidate_id": candidate_id,
            "jd_inputs": [jd],
        })
        job_id = resp.json()[0]["id"]
        
        print(f"Job submitted: {job_id}. Capturing logs...")
        
        # 3. Start log capture in background
        log_file = open(f"{ARTIFACT_DIR}/worker_live.log", "w")
        log_proc = subprocess.Popen(
            ["sg", "docker", "-c", f"cd /home/damar/pelgo-ai && docker compose logs -f worker worker2"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        # 4. Poll
        start_time = time.time()
        result_data = None
        while time.time() - start_time < 300:
            resp = await client.get(f"/api/v1/matches/{job_id}")
            data = resp.json()
            if data["status"] == "completed":
                result_data = data
                break
            if data["status"] == "failed":
                print("Job failed!")
                result_data = data
                break
            await asyncio.sleep(2)
        
        # 5. Cleanup
        log_proc.terminate()
        log_file.close()
        
        if result_data:
            with open(f"{ARTIFACT_DIR}/raw_result.json", "w") as f:
                json.dump(result_data, f, indent=2)
            print("Trace captured successfully.")
        else:
            print("Timed out.")

if __name__ == "__main__":
    asyncio.run(run_live_trace())
