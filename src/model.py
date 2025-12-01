# src/model.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from typing import Dict
import os

def save_model(bundle: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)

def load_model(path: str):
    return joblib.load(path)

def create_default_model():
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    return model
