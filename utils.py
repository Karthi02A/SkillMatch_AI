import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

def extract_text_from_resume(uploaded_file):
    if uploaded_file is None:
        return ""
    return uploaded_file.read().decode("utf-8", errors="ignore")

def load_job_descriptions(file_path="job_descriptions.csv"):
    df = pd.read_csv(file_path)

    expected_cols = {"job_title", "job_description", "skills"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    # ✅ Ensure proper job title formatting like "Data Scientist"
    df["job_title"] = df["job_title"].astype(str).str.strip().str.title()
    df["job_description"] = df["job_description"].astype(str).apply(clean_text)
    df["skills"] = df["skills"].astype(str).apply(clean_text)
    return df

def get_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)


def extract_skills(text, skills_list):
    found_skills = []
    text = text.lower()
    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text):
            found_skills.append(skill)
    return found_skills
