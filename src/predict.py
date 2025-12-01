# src/predict.py
import pandas as pd
from src.features import FeatureBuilder
from src.utils import extract_skills

def predict_from_text(candidate_text, job_text, model_bundle):

    df = pd.DataFrame([{
        'candidate_text': candidate_text,
        'job_text': job_text,
        'candidate_title': '',
        'job_title': ''
    }])

    fb = FeatureBuilder()
    fb.tfidf = model_bundle['tfidf']

    features, _ = fb.build(df, fit_tfidf=False)

    score = float(model_bundle['model'].predict(features)[0])
    score = max(0, min(100, score))

    return {"score": score}
