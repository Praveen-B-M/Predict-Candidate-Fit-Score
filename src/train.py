# src/train.py
import pandas as pd
import numpy as np
from .features import FeatureBuilder
from .model import create_default_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_pairs(candidates_csv: str, jobs_csv: str, max_pairs_per_candidate: int = 1):
    cand = pd.read_csv(candidates_csv)
    jobs = pd.read_csv(jobs_csv)

    def cand_text(r):
        parts = []
        for c in ['summary', 'skills', 'experience', 'name']:
            if c in r and pd.notna(r[c]):
                parts.append(str(r[c]))
        return '. '.join(parts)

    def job_text(r):
        parts = []
        for c in ['title', 'required_skills', 'required_experience', 'description']:
            if c in r and pd.notna(r[c]):
                parts.append(str(r[c]))
        return '. '.join(parts)

    cand['candidate_text'] = cand.apply(cand_text, axis=1)
    jobs['job_text'] = jobs.apply(job_text, axis=1)

    pairs = []
    rng = np.random.RandomState(1)
    for _, c in cand.iterrows():
        sampled_jobs = jobs.sample(n=max_pairs_per_candidate, random_state=int(c.get('candidate_id',0)+1))
        for _, j in sampled_jobs.iterrows():
            c_sk = set(c['candidate_text'] and c['candidate_text'] and [] or [])
            # We'll compute heuristic labeling using FeatureBuilder to keep consistency
            pairs.append({
                'candidate_id': c.get('candidate_id', None),
                'job_id': j.get('job_id', None),
                'candidate_text': c['candidate_text'],
                'job_text': j['job_text'],
                'candidate_title': c.get('name',''),
                'job_title': j.get('title','')
            })
    df_pairs = pd.DataFrame(pairs)
    return df_pairs

def generate_labels(df_pairs):
    # Use simple heuristic for labels if no human labels available
    from .utils import extract_skills, extract_years, extract_edu_level
    labels = []
    for _, row in df_pairs.iterrows():
        c_sk = set(extract_skills(row['candidate_text']))
        j_sk = set(extract_skills(row['job_text']))
        overlap = len(c_sk.intersection(j_sk))
        sim = overlap / (len(j_sk) + 1e-6)
        c_exp = extract_years(row['candidate_text'])
        j_exp = extract_years(row['job_text'])
        exp_score = max(0, 1 - max(0, (j_exp - c_exp)) / (j_exp + 1e-6)) if j_exp > 0 else 1.0
        c_edu = extract_edu_level(row['candidate_text'])
        j_edu = extract_edu_level(row['job_text'])
        edu_score = 1.0 if c_edu >= j_edu else 0.8
        base = 0.6 * sim + 0.25 * exp_score + 0.15 * edu_score
        labels.append(float(np.clip((base + np.random.normal(0, 0.04)) * 100, 0, 100)))
    df_pairs['score'] = labels
    return df_pairs

def train(candidates_csv: str, jobs_csv: str, model_out_path: str):
    df_pairs = prepare_pairs(candidates_csv, jobs_csv, max_pairs_per_candidate=1)
    df_pairs = generate_labels(df_pairs)
    fb = FeatureBuilder()
    X, tfidf = fb.build(df_pairs, fit_tfidf=True)
    y = df_pairs['score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_default_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2:", r2_score(y_test, preds))

    bundle = {'model': model, 'tfidf': tfidf, 'feature_names': X.columns.tolist()}
    save_model(bundle, model_out_path)
    return bundle
