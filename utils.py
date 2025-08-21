import re
import pandas as pd
import spacy

# ------------------- LOAD SPACY MODEL -------------------
def load_spacy_model():
    try:
        # Try normal load (when installed via pip/requirements.txt)
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            # Fallback: try to import as a module
            import en_core_web_sm
            return en_core_web_sm.load()
        except ImportError:
            raise OSError(
                "❌ spaCy model 'en_core_web_sm' not found. "
                "Make sure it's listed in requirements.txt:\n\n"
                "en_core_web_sm @ https://github.com/explosion/spacy-models/"
                "releases/download/en_core_web_sm-3.7.1/"
                "en_core_web_sm-3.7.1-py3-none-any.whl"
            )

nlp = load_spacy_model()
# ------------------- RESUME TEXT EXTRACTION -------------------
def extract_text_from_resume(uploaded_file):
    """Extract raw text from PDF, DOCX, or TXT resumes."""
    text = ""
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        from PyPDF2 import PdfReader
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text += page.extract_text() or ""

    elif file_type == "docx":
        import docx
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    return text.strip()


# ------------------- JOB DESCRIPTIONS LOADER -------------------
def load_job_descriptions(file_path):
    """
    Reads job_descriptions.csv and normalizes columns.
    Maps to: job_title, skills, job_description
    """
    df = pd.read_csv(file_path)

    # Normalize column names (strip + lowercase + replace spaces)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Map known columns
    rename_map = {}
    if "job_role" in df.columns:
        rename_map["job_role"] = "job_title"
    if "skill" in df.columns:
        rename_map["skill"] = "skills"
    if "job_descriptions" in df.columns:
        rename_map["job_descriptions"] = "job_description"

    df.rename(columns=rename_map, inplace=True)

    # Validate required columns
    expected_cols = {"job_title", "skills", "job_description"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"❌ Missing required columns. Found: {list(df.columns)}. "
            f"Expected at least: {expected_cols}"
        )

    return df[["job_title", "skills", "job_description"]]


# ------------------- MATCHING SCORE -------------------
def get_match_score(resume_text, job_description):
    """Compute similarity score (0–100) using spaCy embeddings."""
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    resume_doc = nlp(resume_text.lower())
    job_doc = nlp(job_description.lower())

    similarity = resume_doc.similarity(job_doc)
    return round(similarity * 100, 2)


# ------------------- SKILL EXTRACTION -------------------
def extract_skills(resume_text, jd_skills):
    """Extract skills from resume text by matching against JD skills."""
    resume_text_lower = resume_text.lower()
    matched = []
    for skill in str(jd_skills).split(","):
        skill = skill.strip()
        if skill and re.search(rf"\b{re.escape(skill.lower())}\b", resume_text_lower):
            matched.append(skill)
    return matched
