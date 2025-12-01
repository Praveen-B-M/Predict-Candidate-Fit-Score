# src/utils.py
import re
from typing import List, Optional
from rapidfuzz import fuzz

EDU_LEVELS = {'phd':4,'doctor':4,'master':3,'ms':3,'m.s':3,'bachelor':2,'bsc':2,'bs':2,'b.tech':2,'diploma':1,'high school':0}
DEFAULT_SKILLS = [
    "python","java","c++","c#","javascript","react","angular","node","django","flask",
    "sql","mysql","postgresql","mongodb","aws","azure","gcp","docker","kubernetes",
    "tensorflow","pytorch","scikit-learn","nlp","spark","hadoop","git","linux","rest api"
]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_skills(text: str, skill_list: Optional[List[str]] = None) -> List[str]:
    txt = clean_text(text)
    skills = skill_list or DEFAULT_SKILLS
    found = []
    for s in skills:
        sl = s.lower()
        if re.search(r"\b" + re.escape(sl) + r"\b", txt):
            found.append(s)
        else:
            if len(txt) > 0 and fuzz.partial_ratio(sl, txt) >= 92:
                found.append(s)
    return list(set(found))

def extract_years(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    nums = re.findall(r"(\d+\.?\d*)\s*(?:years|yrs|y)", text.lower())
    if not nums:
        nums = re.findall(r"(\d+)(?=\s+experience|\s+exp)", text.lower())
    nums = [float(x) for x in nums] if nums else []
    return max(nums) if nums else 0.0

def extract_edu_level(text: str) -> int:
    txt = clean_text(text)
    best = 0
    for k, v in EDU_LEVELS.items():
        if k in txt:
            best = max(best, v)
    return best
