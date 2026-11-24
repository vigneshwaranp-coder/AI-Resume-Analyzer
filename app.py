"""FastAPI service for the AI Resume Analyzer training project."""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from resume_analyzer_core import (
    MAX_UPLOAD_MB,
    EmptyResumeError,
    LLMFailure,
    MissingAPIKeyError,
    ResumeReadError,
    ScannedPDFError,
    analyze_resume_workflow,
    estimate_cost,
    read_resume_bytes,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("resume_analyzer")
app = FastAPI(
    title="AI Resume Analyzer",
    description="Training project: extracts resume details and compares them with a job description.",
    version="1.0.0",
)


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%s", request_id, request.method, request.url.path, response.status_code, round((time.time()-start)*1000, 2))
    return response


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "AI Resume Analyzer"}


@app.post("/analyze", tags=["analysis"])
def analyze(
    request: Request,
    job_description: str = Form(..., min_length=10),
    resume_text: Optional[str] = Form(default=None),
    resume_file: Optional[UploadFile] = File(default=None),
):
    """Accept exactly one resume source: resume_text OR resume_file."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    if bool(resume_text and resume_text.strip()) == bool(resume_file):
        raise HTTPException(status_code=422, detail="Provide exactly one resume source: resume_text or resume_file.")
    if resume_file is not None:
        filename = resume_file.filename or ""
        extension = os.path.splitext(filename)[1].lower()
        if extension not in {".pdf", ".txt"}:
            raise HTTPException(status_code=415, detail="Unsupported file type. Upload a PDF or TXT resume.")
        data = resume_file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
        try:
            text = read_resume_bytes(filename, data)
        except ScannedPDFError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EmptyResumeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ResumeReadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        text = (resume_text or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="Resume text cannot be empty.")

    try:
        return analyze_resume_workflow(text, job_description, request_id=request_id)
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured. Add GEMINI_API_KEY to .env.") from exc
    except ValueError as exc:
        # Includes token-limit overflow. Keep the API contract explicit.
        message = str(exc)
        if "tokens; limit" in message:
            raise HTTPException(status_code=413, detail=message) from exc
        raise HTTPException(status_code=422, detail=message) from exc
    except LLMFailure as exc:
        logger.warning("request_id=%s llm_failure=%s", request_id, str(exc))
        raise HTTPException(status_code=502, detail="The LLM could not return a valid result after retries.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="The LLM request timed out. Please retry later.") from exc
    except Exception as exc:
        # Do not return raw provider details, API keys, or resume content to the caller.
        logger.exception("request_id=%s unexpected_analyzer_error=%s", request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="The analyzer service is temporarily unavailable.") from exc


@app.get("/docs-example", include_in_schema=False)
def docs_example():
    return {
        "endpoint": "POST /analyze",
        "content_type": "multipart/form-data",
        "fields": {
            "job_description": "Job Title: Python Developer\\nRequired: Python, FastAPI, SQL",
            "resume_file": "Upload one .pdf or .txt file OR use resume_text instead.",
        },
        "curl": "curl -X POST http://127.0.0.1:8000/analyze -F 'job_description=Job Title: Python Developer; Required: Python, FastAPI, SQL' -F 'resume_file=@resumes/resume_01.pdf'",
    }
