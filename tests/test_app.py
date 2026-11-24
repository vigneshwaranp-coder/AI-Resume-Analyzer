import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from resume_analyzer_core import ResumeData, validate_json_with_retry

client = TestClient(app_module.app)


def fake_success(text, job_description, request_id=None):
    return {
        "candidate": {
            "candidate_name": "Fictional Candidate",
            "skills": ["Python", "SQL"],
            "years_of_experience": 2.0,
            "education": ["BSc Computer Science"],
            "strengths": ["Built reports"],
            "gaps": [],
        },
        "match": {
            "job_title": "Python Developer",
            "score": 78,
            "matched_skills": ["Python", "SQL"],
            "missing_skills": ["FastAPI"],
            "reasoning": "The fictional resume explicitly lists Python and SQL.",
        },
        "token_usage": {"input_tokens": 100, "output_tokens": 60, "total_tokens": 160},
        "cost": {"pricing_tier": "free", "estimated_cost_usd": 0.0, "paid_standard_reference_usd": 0.00018, "note": "test"},
        "request_id": request_id or "test-request-id",
    }


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_successful_text_analysis(monkeypatch):
    monkeypatch.setattr(app_module, "analyze_resume_workflow", fake_success)
    response = client.post("/analyze", data={"resume_text": "Python SQL developer with two years of experience.", "job_description": "Job Title: Python Developer. Required: Python, SQL."})
    assert response.status_code == 200
    assert response.json()["match"]["score"] == 78


def test_invalid_json_then_success_retry():
    outputs = iter([
        ('{"candidate_name":"Bad","skills":["Python"],"years_of_experience":60,"education":[],"strengths":[],"gaps":[]}', {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}),
        ('{"candidate_name":"Good","skills":["Python"],"years_of_experience":2,"education":[],"strengths":[],"gaps":[]}', {"input_tokens": 12, "output_tokens": 9, "total_tokens": 21}),
    ])
    def fake_generator(prompt, response_model):
        return next(outputs)
    result = validate_json_with_retry("test prompt", ResumeData, generate_fn=fake_generator, pause_seconds=0)
    assert result["success"] is True
    assert result["attempts"] == 2


def test_empty_resume_file_returns_422(monkeypatch):
    monkeypatch.setattr(app_module, "analyze_resume_workflow", fake_success)
    response = client.post("/analyze", data={"job_description": "Python Developer role"}, files={"resume_file": ("empty.txt", b"   ", "text/plain")})
    assert response.status_code == 422


def test_wrong_file_type_returns_415():
    response = client.post("/analyze", data={"job_description": "Python Developer role"}, files={"resume_file": ("resume.docx", b"dummy", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert response.status_code == 415


def test_missing_resume_source_returns_422():
    response = client.post("/analyze", data={"job_description": "Python Developer role"})
    assert response.status_code == 422


def test_scanned_pdf_returns_422():
    scanned_path = Path(__file__).resolve().parents[1] / "resumes" / "scanned_resume_test.pdf"
    response = client.post("/analyze", data={"job_description": "Python Developer role"}, files={"resume_file": ("scanned.pdf", scanned_path.read_bytes(), "application/pdf")})
    assert response.status_code == 422


def test_file_too_large_returns_413(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_MB", 1)
    response = client.post("/analyze", data={"job_description": "Python Developer role"}, files={"resume_file": ("large.txt", b"a" * (1024 * 1024 + 1), "text/plain")})
    assert response.status_code == 413


def test_request_id_header_is_added(monkeypatch):
    monkeypatch.setattr(app_module, "analyze_resume_workflow", fake_success)
    response = client.post("/analyze", headers={"X-Request-ID": "my-test-id"}, data={"resume_text": "Python developer", "job_description": "Python role requirements"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "my-test-id"


def test_llm_invalid_json_after_retries_returns_502(monkeypatch):
    def fail_analysis(text, job_description, request_id=None):
        raise app_module.LLMFailure("invalid JSON after three attempts")
    monkeypatch.setattr(app_module, "analyze_resume_workflow", fail_analysis)
    response = client.post("/analyze", data={"resume_text": "Python developer with SQL", "job_description": "Python Developer role requirements"})
    assert response.status_code == 502


def test_timeout_returns_503(monkeypatch):
    def timeout_analysis(text, job_description, request_id=None):
        raise TimeoutError("simulated timeout")
    monkeypatch.setattr(app_module, "analyze_resume_workflow", timeout_analysis)
    response = client.post("/analyze", data={"resume_text": "Python developer with SQL", "job_description": "Python Developer role requirements"})
    assert response.status_code == 503


def test_rate_limit_or_unavailable_returns_502(monkeypatch):
    def unavailable_analysis(text, job_description, request_id=None):
        raise app_module.LLMFailure("simulated rate limit")
    monkeypatch.setattr(app_module, "analyze_resume_workflow", unavailable_analysis)
    response = client.post("/analyze", data={"resume_text": "Python developer with SQL", "job_description": "Python Developer role requirements"})
    assert response.status_code == 502


def test_token_limit_returns_413(monkeypatch):
    def over_limit_analysis(text, job_description, request_id=None):
        raise ValueError("Resume is 12001 tokens; limit is 12000.")
    monkeypatch.setattr(app_module, "analyze_resume_workflow", over_limit_analysis)
    response = client.post("/analyze", data={"resume_text": "Python developer with SQL", "job_description": "Python Developer role requirements"})
    assert response.status_code == 413
