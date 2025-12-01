# src/features.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import Tuple, Optional
from .utils import clean_text, extract_skills, extract_years, extract_edu_level

class FeatureBuilder:
    def __init__(self, tfidf_max_features: int = 4000):
        self.tfidf_max = tfidf_max_features
        self.tfidf = None

    def fit_tfidf(self, candidate_texts, job_texts):
        corpus = [c + ' ' + j for c, j in zip(candidate_texts, job_texts)]
        self.tfidf = TfidfVectorizer(max_features=self.tfidf_max, ngram_range=(1,2))
        self.tfidf.fit(corpus)
        return self

    def transform_texts(self, candidate_texts, job_texts):
        cand = [clean_text(x) for x in candidate_texts]
        job = [clean_text(x) for x in job_texts]
        A = self.tfidf.transform(cand)
        B = self.tfidf.transform(job)
        cos = linear_kernel(A, B).diagonal()
        cand_norm = np.sqrt(A.multiply(A).sum(axis=1)).A1
        job_norm = np.sqrt(B.multiply(B).sum(axis=1)).A1
        return cos, cand_norm, job_norm

    def build(self, df, fit_tfidf: bool = True) -> Tuple[pd.DataFrame, Optional[TfidfVectorizer]]:
        candidate_texts = df['candidate_text'].fillna('').astype(str).tolist()
        job_texts = df['job_text'].fillna('').astype(str).tolist()

        if fit_tfidf or self.tfidf is None:
            self.fit_tfidf(candidate_texts, job_texts)

        cos, cand_norm, job_norm = self.transform_texts(candidate_texts, job_texts)

        cand_sk = [extract_skills(t) for t in candidate_texts]
        job_sk = [extract_skills(t) for t in job_texts]
        skill_overlap = [len(set(a).intersection(set(b))) for a, b in zip(cand_sk, job_sk)]
        job_skill_count = [max(1, len(b)) for b in job_sk]
        skill_ratio = [ov / jc for ov, jc in zip(skill_overlap, job_skill_count)]

        cand_exp = [extract_years(t) for t in candidate_texts]
        job_exp = [extract_years(t) for t in job_texts]
        exp_gap = [max(0.0, j - c) for c, j in zip(cand_exp, job_exp)]

        cand_edu = [extract_edu_level(t) for t in candidate_texts]
        job_edu = [extract_edu_level(t) for t in job_texts]
        edu_gap = [max(0, j - c) for c, j in zip(cand_edu, job_edu)]

        title_sim = []
        cand_titles = df.get('candidate_title', pd.Series([''] * len(df))).fillna('').astype(str).tolist()
        job_titles = df.get('job_title', pd.Series([''] * len(df))).fillna('').astype(str).tolist()
        from rapidfuzz import fuzz
        title_sim = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(cand_titles, job_titles)]

        features = pd.DataFrame({
            'cosine_sim': cos,
            'cand_norm': cand_norm,
            'job_norm': job_norm,
            'skill_overlap': skill_overlap,
            'skill_ratio': skill_ratio,
            'cand_exp': cand_exp,
            'job_exp': job_exp,
            'exp_gap': exp_gap,
            'cand_edu': cand_edu,
            'job_edu': job_edu,
            'edu_gap': edu_gap,
            'title_sim': title_sim
        })

        return features, self.tfidf
