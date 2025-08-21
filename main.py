import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your crew setup
from crew_test import crew   # your existing code (the big script) should be in crew_agents.py

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AI Knowledge Navigator",
    description="AI-powered project analysis, resource discovery, and code generation API",
    version="1.0.0"
)

# Request model
class AnalysisRequest(BaseModel):
    pdf_path: str

@app.get("/")
def home():
    return {"message": "🚀 AI Knowledge Navigator API is running!"}

@app.post("/run-analysis")
def run_analysis(request: AnalysisRequest):
    """
    Run the full CrewAI pipeline on a given PDF path.
    """
    result = crew.kickoff(inputs={"pdf_path": request.pdf_path})
    return {"status": "completed", "result": str(result)}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and run analysis directly.
    """
    pdf_path = Path(f"./uploads/{file.filename}")
    pdf_path.parent.mkdir(exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    result = crew.kickoff(inputs={"pdf_path": str(pdf_path)})
    return {"status": "completed", "result": str(result), "pdf_used": file.filename}

