[![CI](https://github.com/ashutosh8950/ai-resume-screener/actions/workflows/ci.yml/badge.svg)](https://github.com/ashutosh8950/ai-resume-screener/actions/workflows/ci.yml)

# AI Resume Screener

Compare a resume with a job description using local semantic embeddings, requirement-level evidence, skill coverage, keyword matching, and TF-IDF. The app runs locally and does not require an API key for analysis.

[Production demo](https://ai-resume-screener-w6h3.onrender.com) | [Staging demo](https://ai-resume-screener-staging.onrender.com)

The Render free-tier instance may take 1-2 minutes to wake up after idle on its first request.

Gemini suggestions are optional. Add a free Gemini API key in the sidebar for the current session; without a key, rule-based suggestions are available every time. A `GEMINI_API_KEY` environment variable is also supported.

## Features

- **Free semantic embeddings:** FastEmbed runs the lite MiniLM profile locally through ONNX Runtime, without a paid embedding API.
- **Requirement-to-evidence matching:** requirements are classified into Strong, Partial, and Gaps tables with the best matching resume evidence and match rationale.
- **Skill-gap analysis:** see required technical skills found in the resume and skills that appear to be missing.
- **Suggestions that always work:** deterministic rule-based suggestions are the default and remain available when Gemini is not configured or a request fails. Optional Gemini suggestions use a key provided per session.
- **Batch Mode:** use one resume, pasted or uploaded once, against 2 to 6 job descriptions. The app ranks the jobs by overall score and shows a skill comparison and deeper skill-gap view for the best match.
- **Bulk Resumes:** upload up to 10 PDF resumes (5 MB each) against one job description to get a ranked table with overall and semantic scores, skill coverage, strong/partial/gap counts, and top missing skills; inspect per-resume evidence details and export CSV with spreadsheet-formula-injection protection.
- **PDF input:** upload a resume PDF up to the configured size and page limits, or paste resume text.
- **Memory-safe on Render's free 512 MB tier:** a 192 MB memory guard falls back to skill/keyword/TF-IDF scoring instead of crashing.

### UI

- Dark theme with a purple accent, animated score ring, and animated score tiles.
- Evidence tables with score bars and a styled Bulk Resumes leaderboard.
- Per-job skill chips in Batch Mode.
- Scanning spinner during analysis.
- Job-description text cleanup for pasted bullet glyphs.

## How scoring works

When semantic embeddings are available, the overall score combines:

| Signal | Weight |
|---|---:|
| Semantic requirement coverage | 55% |
| Technical skill coverage | 30% |
| Keyword match | 7.5% |
| TF-IDF similarity | 7.5% |

Semantic matching compares job requirements with resume evidence. Skill coverage checks required technical skill groups, keyword matching measures shared terms, and TF-IDF measures term-weighted similarity over filtered job-description requirements. Batch Mode ranks jobs by unrounded overall score, then skill coverage, then semantic score.

If embeddings are disabled, unavailable, or skipped because of memory pressure, the app uses the fallback score:

| Signal | Fallback weight |
|---|---:|
| Technical skill coverage | 50% |
| Keyword match | 25% |
| TF-IDF similarity | 25% |

## Tech stack

- Python 3.11
- Streamlit
- FastEmbed and ONNX Runtime for local CPU embeddings
- scikit-learn for TF-IDF and cosine similarity
- pandas and Plotly for result tables and charts
- Gemini API via `requests` for optional suggestions
- pytest for automated tests

## Run locally

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/ashutosh8950/ai-resume-screener.git
cd ai-resume-screener
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the application dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run tests

Install the development dependencies and run pytest:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite currently has 83 passing tests and 1 skipped.

## Project structure

```text
app.py
resume_screener/
	__init__.py
	analyzer.py
	bulk.py  # Upload limits, ranking, and safe CSV export
	config.py
	embeddings.py
	llm.py
	logger.py
	pdf_parser.py
	skills.py
	suggestions.py
	text_utils.py
	ui_helpers.py
	visualizer.py
	skills.json
	suggestions.json
tests/
	fixtures/
	test_analyzer.py
	test_app_compiles.py
	test_bulk.py
	test_fixture_screener.py
	test_pdf_parser.py
	test_suggestions.py
	test_text_utils.py
	test_ui_helpers.py
	test_ui_helpers_leaderboard.py
scripts/
	calibrate.py
	measure_memory.py
.github/workflows/ci.yml
requirements.txt
requirements-dev.txt
