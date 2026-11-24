"""Core functions for the AI Resume Analyzer training project.

All example resumes are fictional. Do not send confidential personal data to the free API tier.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
RESUME_DIR = PROJECT_DIR / "resumes"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"
PROMPT_DIR = PROJECT_DIR / "prompts"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PRICING_TIER = os.getenv("GEMINI_PRICING_TIER", "free").lower()
MAX_RESUME_TOKENS = int(os.getenv("MAX_RESUME_TOKENS", "12000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
# Current Gemini 2.5 Flash Standard text rates shown on the official pricing page.
# These are used only for a paid-tier reference estimate; check the live page before real billing.
PAID_INPUT_USD_PER_MILLION = 0.30
PAID_OUTPUT_USD_PER_MILLION = 2.50


class ResumeData(BaseModel):
    """Required candidate extraction schema from the assessment."""
    model_config = ConfigDict(extra="forbid")
    candidate_name: str | None
    skills: list[str] = Field(min_length=1)
    years_of_experience: float | None = Field(ge=0, le=50)
    education: list[str]
    strengths: list[str]
    gaps: list[str]


class JobMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_title: str
    score: int = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str


class JobMatchBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: list[JobMatch] = Field(min_length=1)


class ResumeReadError(ValueError):
    pass


class ScannedPDFError(ResumeReadError):
    pass


class EmptyResumeError(ResumeReadError):
    pass


class MissingAPIKeyError(RuntimeError):
    pass


class LLMFailure(RuntimeError):
    pass


def clean_resume_text(text: str) -> str:
    """Remove page labels, repeated blank lines, and common noise without rewriting facts."""
    if not text or not text.strip():
        raise EmptyResumeError("No readable text was found in the resume.")
    kept_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"(?i)(page\s+)?\d+(\s+of\s+\d+)?", line):
            continue
        if re.fullmatch(r"(?i)page\s+\d+(\s+of\s+\d+)?", line):
            continue
        # Collapse repeated internal whitespace, preserving the words themselves.
        line = re.sub(r"\s+", " ", line)
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()
    if not cleaned:
        raise EmptyResumeError("No readable text remained after cleaning the resume.")
    return cleaned


def _remove_repeated_pdf_headers_footers(page_texts: list[str]) -> list[str]:
    if len(page_texts) <= 1:
        return page_texts
    edge_lines = []
    for page_text in page_texts:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        edge_lines.extend(lines[:2] + lines[-2:])
    counts = Counter(edge_lines)
    repeated = {line for line, count in counts.items() if count >= 2 and len(line) <= 100}
    cleaned_pages = []
    for page_text in page_texts:
        lines = []
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped in repeated:
                continue
            lines.append(line)
        cleaned_pages.append("\n".join(lines))
    return cleaned_pages


def read_resume_bytes(filename: str, file_bytes: bytes) -> str:
    """Extract text from PDF/TXT bytes and clean it. Scanned-only PDFs fail clearly."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            raw_text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ResumeReadError("TXT file must be UTF-8 encoded.") from exc
    elif suffix == ".pdf":
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                page_texts = [page.get_text("text") for page in pdf]
        except Exception as exc:
            raise ResumeReadError(f"Could not read the PDF: {exc}") from exc
        page_texts = _remove_repeated_pdf_headers_footers(page_texts)
        raw_text = "\n".join(page_texts)
        if not raw_text.strip():
            raise ScannedPDFError(
                "PDF has no selectable text. It may be scanned/image-only; OCR is not included in this project."
            )
    else:
        raise ResumeReadError("Unsupported file type. Upload a .pdf or .txt file.")
    return clean_resume_text(raw_text)


def read_resume(path: str | Path) -> str:
    """Required read_resume(path): extract PDF text with PyMuPDF or read TXT."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {path}")
    return read_resume_bytes(path.name, path.read_bytes())


def get_gemini_client(client: Any = None):
    if client is not None:
        return client
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your Google AI Studio key."
        )
    from google import genai  # lazy import so offline tests can import this module
    from google.genai import types
    # SDK timeout is in milliseconds; this keeps a single model call bounded.
    http_options = types.HttpOptions(timeout=30_000)
    return genai.Client(api_key=api_key, http_options=http_options)


def count_gemini_tokens(text: str, client: Any = None) -> int:
    """Count tokens using the Gemini API model tokenizer (not a word-count estimate)."""
    from google import genai  # lazy import
    client = get_gemini_client(client)
    result = client.models.count_tokens(model=GEMINI_MODEL, contents=text)
    count = getattr(result, "total_tokens", None)
    if count is None:
        raise RuntimeError("Gemini did not return a total token count.")
    return int(count)


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", input_tokens + output_tokens) or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def estimate_cost(usage: dict[str, int], pricing_tier: str | None = None) -> dict[str, float | str]:
    """Report free-tier estimated charge and a separate paid-standard reference estimate."""
    tier = (pricing_tier or GEMINI_PRICING_TIER).lower()
    paid_reference = (
        usage.get("input_tokens", 0) * PAID_INPUT_USD_PER_MILLION / 1_000_000
        + usage.get("output_tokens", 0) * PAID_OUTPUT_USD_PER_MILLION / 1_000_000
    )
    # This project is configured for Free tier. Requests beyond the active free quota
    # should be refused by the service; the user must not upgrade billing for this exercise.
    actual_estimate = 0.0 if tier == "free" else paid_reference
    return {
        "pricing_tier": tier,
        "estimated_cost_usd": round(actual_estimate, 8),
        "paid_standard_reference_usd": round(paid_reference, 8),
        "note": "Free-tier estimate assumes the request is eligible and within current quota; verify account tier and live pricing.",
    }


def _append_attempt_log(record: dict[str, Any], log_path: str | Path | None = None) -> None:
    """Log metadata only. Never write resume text or API credentials to the log."""
    target = Path(log_path) if log_path else LOG_DIR / "api_attempts.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_record = dict(record)
    safe_record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def _generate_json(prompt: str, response_model: type[BaseModel], client: Any = None):
    from google.genai import types  # lazy import
    client = get_gemini_client(client)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_model,
            max_output_tokens=2048,
        ),
    )
    response_text = getattr(response, "text", None)
    if not response_text:
        raise LLMFailure("Gemini returned an empty response.")
    return response_text, _usage_dict(response)


def validate_json_with_retry(
    prompt: str,
    response_model: type[BaseModel],
    *,
    client: Any = None,
    max_attempts: int = 3,
    generate_fn: Callable | None = None,
    pause_seconds: float = 1.0,
    log_path: str | Path | None = None,
    stage: str = "validation",
) -> dict[str, Any]:
    """Generate JSON, validate it with Pydantic, and retry with the validation error."""
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_error = None
    for attempt in range(1, max_attempts + 1):
        retry_prompt = prompt
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if last_error:
            retry_prompt += (
                "\n\nCORRECTION REQUIRED. Your previous response failed validation. "
                "Fix only the reported issue and return valid JSON. Validation error: "
                + last_error[:1800]
            )
        try:
            raw_text, usage = (
                generate_fn(retry_prompt, response_model)
                if generate_fn is not None
                else _generate_json(retry_prompt, response_model, client)
            )
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0) or 0)
            validated = response_model.model_validate_json(raw_text)
            _append_attempt_log({
                "stage": stage,
                "attempt": attempt,
                "success": True,
                **usage,
                "error_type": None,
            }, log_path)
            return {
                "success": True,
                "data": validated.model_dump(),
                "usage": usage_total,
                "attempts": attempt,
                "error": None,
            }
        except ValidationError as exc:
            last_error = str(exc)
            error_type = "ValidationError"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            error_type = type(exc).__name__
        _append_attempt_log({
            "stage": stage,
            "attempt": attempt,
            "success": False,
            **usage,
            "error_type": error_type,
            "error": last_error[:1200],
        }, log_path)
        if attempt < max_attempts and pause_seconds:
            time.sleep(pause_seconds * attempt)
    return {
        "success": False,
        "data": None,
        "usage": usage_total,
        "attempts": max_attempts,
        "error": last_error or "Unknown validation failure.",
    }


def extract_resume_with_retry(
    resume_text: str,
    *,
    prompt_template: str | None = None,
    client: Any = None,
    max_attempts: int = 3,
    generate_fn: Callable | None = None,
    pause_seconds: float = 1.0,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if not resume_text or not resume_text.strip():
        return {"success": False, "data": None, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "attempts": 0, "error": "Resume text is empty."}
    template = prompt_template or (PROMPT_DIR / "extraction_prompt_v1.txt").read_text(encoding="utf-8")
    prompt = template.replace("{resume_text}", resume_text)
    return validate_json_with_retry(
        prompt, ResumeData, client=client, max_attempts=max_attempts,
        generate_fn=generate_fn, pause_seconds=pause_seconds,
        log_path=log_path, stage="resume_extraction",
    )


def match_resume_to_jobs_with_retry(
    candidate_data: dict[str, Any],
    jobs: list[dict[str, str]],
    *,
    prompt_template: str | None = None,
    client: Any = None,
    max_attempts: int = 3,
    generate_fn: Callable | None = None,
    pause_seconds: float = 1.0,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if not jobs:
        return {"success": False, "data": None, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "attempts": 0, "error": "At least one job description is required."}
    template = prompt_template or (PROMPT_DIR / "matching_prompt_v1.txt").read_text(encoding="utf-8")
    prompt = template.replace("{candidate_json}", json.dumps(candidate_data, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{jobs_json}", json.dumps(jobs, ensure_ascii=False, indent=2))
    return validate_json_with_retry(
        prompt, JobMatchBatch, client=client, max_attempts=max_attempts,
        generate_fn=generate_fn, pause_seconds=pause_seconds,
        log_path=log_path, stage="job_matching",
    )


def analyze_resume_workflow(resume_text: str, job_description: str, request_id: str | None = None) -> dict[str, Any]:
    """Run extraction + one-job matching for the FastAPI endpoint."""
    client = get_gemini_client()
    token_count = count_gemini_tokens(resume_text, client=client)
    if token_count > MAX_RESUME_TOKENS:
        raise ValueError(f"Resume is {token_count} tokens; limit is {MAX_RESUME_TOKENS}.")
    extraction = extract_resume_with_retry(resume_text, client=client)
    if not extraction["success"]:
        raise LLMFailure(f"Resume extraction failed after {extraction['attempts']} attempts: {extraction['error']}")
    title_match = re.search(r"(?im)^\s*(?:job title|position)\s*:\s*(.+)$", job_description)
    job_title = title_match.group(1).strip() if title_match else "Target role"
    matching = match_resume_to_jobs_with_retry(
        extraction["data"], [{"title": job_title, "description": job_description}], client=client
    )
    if not matching["success"]:
        raise LLMFailure(f"Job matching failed after {matching['attempts']} attempts: {matching['error']}")
    usage = {
        key: extraction["usage"].get(key, 0) + matching["usage"].get(key, 0)
        for key in ["input_tokens", "output_tokens", "total_tokens"]
    }
    return {
        "candidate": extraction["data"],
        "match": matching["data"]["matches"][0],
        "token_usage": usage,
        "cost": estimate_cost(usage),
        "request_id": request_id or str(uuid.uuid4()),
    }
