import httpx
import time
import json
import os

BASE_URL = "http://localhost:8000"
PDF_PATH = "/home/damar/pelgo-ai/ilovepdf_merged-1.pdf"

VALVE_JD = """
Software Engineer - Web
Valve is looking for software engineers to join our web team. We build the systems that power Steam, including the store, community, and partner portals.
Requirements:
- Strong experience with JavaScript/TypeScript and modern frameworks (React, Vue, etc.)
- Experience building large-scale web applications
- Familiarity with backend systems (Node.js, Python, or similar)
- Understanding of cloud infrastructure and service architecture
"""

async def run_test():
    async with httpx.AsyncClient(timeout=300) as client:
        # 1. Upload PDF
        print(f"Uploading PDF: {PDF_PATH}...")
        with open(PDF_PATH, "rb") as f:
            resp = await client.post(f"{BASE_URL}/api/v1/candidate/pdf", files={"file": f})
        
        if resp.status_code != 200:
            print(f"Upload failed: {resp.text}")
            return
            
        candidate_data = resp.json()
        candidate_id = candidate_data["id"]
        print(f"Candidate registered: {candidate_id}")

        # 2. Submit JD
        print("Submitting JD match request...")
        resp = await client.post(f"{BASE_URL}/api/v1/matches", json={
            "candidate_id": candidate_id,
            "jd_inputs": [VALVE_JD]
        })
        
        if resp.status_code != 200:
            print(f"Match submission failed: {resp.text}")
            return
            
        jobs = resp.json()
        job_id = jobs[0]["id"]
        print(f"Job enqueued: {job_id}")

        # 3. Poll for results
        print("Waiting for agent reasoning (this may take 2-3 minutes)...")
        while True:
            resp = await client.get(f"{BASE_URL}/api/v1/matches/{job_id}")
            data = resp.json()
            status = data["status"]
            print(f"Current status: {status}")
            
            if status == "completed":
                print("\n=== SUCCESS! ===")
                print(f"Score: {data['result']['overall_score']}")
                print(f"Confidence: {data['result']['confidence']}")
                print(f"Matched Skills: {data['result']['matched_skills']}")
                print(f"Gaps: {data['result']['gap_skills']}")
                
                print("\n=== TRACE ===")
                for tc in data['agent_trace']['tool_calls']:
                    print(f"- {tc['tool']}: {tc['latency_ms']}ms")
                
                # Save output to artifacts
                with open("/home/damar/pelgo-ai/artifacts/test_result_pdf.json", "w") as out:
                    json.dump(data, out, indent=2)
                break
            elif status == "failed":
                print(f"\n=== FAILED ===\n{data.get('error_detail')}")
                break
                
            time.sleep(10)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_test())
