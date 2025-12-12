"""
FastAPI application for mood-classifier.

Endpoints:
- POST /analyze/set - Analyze DJ set
- POST /analyze/track - Analyze single track
- GET /jobs/{job_id} - Get job status
- GET /health - Health check
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# Local imports
from .analysis import AnalysisService

app = FastAPI(
    title="Mood Classifier API",
    description="DJ audio analysis service",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (use Redis in production)
jobs: dict = {}


# ============== Models ==============

class AnalyzeURLRequest(BaseModel):
    url: HttpUrl
    callback_url: Optional[HttpUrl] = None


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@app.post("/analyze/url", response_model=AnalyzeResponse)
async def analyze_url(request: AnalyzeURLRequest, background_tasks: BackgroundTasks):
    """
    Analyze audio from URL (SoundCloud, etc.).

    Returns job_id for tracking progress.
    """
    job_id = str(uuid.uuid4())

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    # Queue analysis task
    background_tasks.add_task(
        run_analysis_task,
        job_id=job_id,
        url=str(request.url),
        callback_url=str(request.callback_url) if request.callback_url else None,
    )

    return AnalyzeResponse(
        job_id=job_id,
        status="pending",
        message="Analysis queued. Use /jobs/{job_id} to check status.",
    )


@app.post("/analyze/file", response_model=AnalyzeResponse)
async def analyze_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Analyze uploaded audio file.

    Max file size: 500MB.
    Supported formats: mp3, wav, flac, m4a.
    """
    # Validate file
    max_size = int(os.getenv("MAX_FILE_SIZE_MB", 500)) * 1024 * 1024

    if file.size and file.size > max_size:
        raise HTTPException(400, f"File too large. Max size: {max_size // (1024*1024)}MB")

    allowed_extensions = {".mp3", ".wav", ".flac", ".m4a", ".opus", ".ogg"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed_extensions}")

    # Save file
    job_id = str(uuid.uuid4())
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    file_path = os.path.join(downloads_dir, f"{job_id}{ext}")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    # Queue analysis
    background_tasks.add_task(
        run_analysis_task,
        job_id=job_id,
        file_path=file_path,
    )

    return AnalyzeResponse(
        job_id=job_id,
        status="pending",
        message="Analysis queued. Use /jobs/{job_id} to check status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    return jobs[job_id]


# ============== Background Tasks ==============

async def run_analysis_task(
    job_id: str,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    callback_url: Optional[str] = None,
):
    """Run analysis in background."""
    try:
        jobs[job_id].status = "processing"
        jobs[job_id].updated_at = datetime.utcnow()

        # Download if URL provided
        if url and not file_path:
            from .downloader import download_audio
            downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
            file_path = await download_audio(url, downloads_dir, job_id)

        if not file_path or not os.path.exists(file_path):
            raise ValueError("No audio file available")

        # Run analysis
        service = AnalysisService()
        result = service.analyze_set(file_path)

        jobs[job_id].status = "completed"
        jobs[job_id].result = result.to_dict() if hasattr(result, 'to_dict') else result
        jobs[job_id].updated_at = datetime.utcnow()

        # Cleanup file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Callback if provided
        if callback_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json={
                    "job_id": job_id,
                    "status": "completed",
                    "result": jobs[job_id].result,
                })

    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].updated_at = datetime.utcnow()

        # Cleanup on error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Create directories
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
    cache_dir = os.getenv("CACHE_DIR", "/tmp/cache")
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    pass
