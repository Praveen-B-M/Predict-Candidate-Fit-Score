# Predict-Candidate-Fit-Score
Build a machine learning model to predict how well a candidate matches a given job description.
-------------------------------------------------------------------------------------------------
# Resume Job Matcher

## Overview
Predict how well a candidate matches a job description (0-100). Includes data processing, feature engineering, model training, evaluation, a FastAPI endpoint, and resume file parsing (PDF/DOCX/TXT).

## Folder structure
See root of repo.

## Quick setup
1. Create a virtualenv and activate:
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

2. Install:
   pip install -r requirements.txt

3. Place your CSVs:
   - data/candidates.csv  (has candidate_id, name, experience, skills, summary)
   - data/jobs.csv        (has job_id, title, required_experience, required_skills, description)

4. Train:
   python src/main.py --train --candidates data/candidates.csv --jobs data/jobs.csv --model-path models/matcher.joblib

5. Serve UI & API:
   python src/main.py --serve --model-path models/matcher.joblib --port 8000
   Open http://localhost:8000/ui

6. Predict single pair:
   python src/main.py --predict --model-path models/matcher.joblib --candidate "..." --job "..."

## How it handles uploaded resumes
- Upload resume via UI or POST to `/predict-file` (multipart file).
- `src/file_parser.py` extracts text from PDF/DOCX/TXT.
- Combined with job text (either select job from jobs.csv or paste JD), the pipeline computes features and returns score and explanation.

## Customization
- Replace heuristic labeling by providing a labeled pairs csv (candidate_id, job_id, score).
- Enable BERT text similarity by installing `sentence-transformers` — training automatically uses it if available and flagged.

## Deliverables
- src/: full code
- models/matcher.joblib: trained model (after you run training)
- README with usage

