# src/api.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
from src.predict import predict_from_text
from src.model import load_model
import pdfplumber
import docx

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Load model once at startup
MODEL_BUNDLE = load_model("models/matcher.joblib")


def extract_text(path):
    """Reads PDF/DOCX/TXT safely"""
    ext = path.lower()
    
    if ext.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])

    elif ext.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        with open(path, "r", errors="ignore") as f:
            return f.read()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/score")
async def score_endpoint(
    request: Request,
    resume_file: UploadFile = File(None),
    resume_text: str = Form(""),
    job_description: str = Form("")
):
    try:
        # -------------------------------
        # VALIDATION CHECKS
        # -------------------------------
        if not resume_file and not resume_text.strip():
            return {"error": "Please upload a resume or paste resume text."}

        if not job_description.strip():
            return {"error": "Please enter a job description."}

        resume_content = ""

        # -------------------------------
        # CASE 1: User uploaded resume file
        # -------------------------------
        if resume_file:
            temp_path = os.path.join(BASE_DIR, "uploaded_" + resume_file.filename)

            with open(temp_path, "wb") as f:
                f.write(await resume_file.read())

            resume_content = extract_text(temp_path)

            # IMPORTANT: clean temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # -------------------------------
        # CASE 2: User pasted resume text
        # -------------------------------
        else:
            resume_content = resume_text.strip()

        # -------------------------------
        # PREDICT SCORE — ONLY RETURN SCORE
        # -------------------------------
        result = predict_from_text(resume_content, job_description, MODEL_BUNDLE)

        return {"fit_score": result["score"]}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "detail": str(e)},
        )
