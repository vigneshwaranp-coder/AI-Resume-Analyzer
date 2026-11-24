# AI Resume Analyzer — Week 4 Project

This training project is completed in four connected parts: resume input handling, extraction and schema, validation and job matching, FastAPI serving, and evaluation. All included resumes are fictional examples.

## Project files

- `resume_analyzer.ipynb` — guided notebook for Parts 1–4.
- `resume_analyzer_core.py` — reusable parsing, Gemini, validation, retry, token and cost helpers.
- `app.py` — FastAPI application exposing `/analyze`, `/health`, `/docs`, and `/docs-example`.
- `resumes/` — eight fictional sample resumes (four PDF and four TXT), plus empty/scanned test fixtures.
- `data/golden_set.json` — expected skill labels for evaluation.
- `prompts/` — versioned extraction and matching prompts.
- `tests/` — automated FastAPI and retry tests using a mocked LLM.
- `requirements.txt` — Python dependencies.
- `.env.example` — environment variable template; copy it to `.env` and add your own key.

## Recommended environment

Use Python 3.12. The model configured by default is `gemini-2.5-flash`. The free tier has quotas and terms that can change. Do not enable paid billing for this training project unless you intentionally choose to. Free-tier input may be used to improve Google's products, so use only dummy or consented data.

## Setup

1. Open this project folder in VS Code.
2. Create a virtual environment if desired: `python -m venv .venv`.
3. Install packages: `python -m pip install -r requirements.txt`.
4. Copy `.env.example` to `.env`.
5. Create a Gemini API key in Google AI Studio and place it only in `.env` as `GEMINI_API_KEY=...`. Never paste it into the notebook or commit `.env`.
6. Open `resume_analyzer.ipynb` and run the cells in order.

## Run the API (after Part 3 and after configuring `.env`)

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. `POST /analyze` accepts multipart form data with a `job_description` and exactly one of `resume_file` (`.pdf`/`.txt`) or `resume_text`.

## Run tests

```bash
python -m pytest -q
python -m pytest --cov=app --cov-report=term-missing
```

Tests mock the LLM, so the test suite does not need a Gemini API key and does not incur API usage.

## Limits and responsible use

- Scanned/image-only PDFs are detected and return a clear error. OCR is not included.
- Upload size defaults to 5 MB; resume token limit defaults to 12,000. Both can be configured in `.env`.
- Matching scores are an assistive comparison only, not an employment decision. Human review is required.
- The system is instructed not to use protected or sensitive attributes. It can still make mistakes; review evidence and correct errors.
- The provided names, emails, organizations and work histories are fictional.

## 5-minute demo outline

1. Show a resume being read and cleaned.
2. Show the structured JSON candidate profile.
3. Show Pydantic validation and retry after an invalid value.
4. Submit the resume and a job description to `POST /analyze`.
5. Show token usage, estimated cost, error handling and the test results.

## Assessment mapping

The notebook includes explicit headings for every mandatory bullet in the four supplied assessment pages. Optional bonus challenges are separated from mandatory work.
