import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from utils import extract_text_from_resume, load_job_descriptions, get_match_score, extract_skills

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="🧠 SkillMatch AI", layout="wide", page_icon="🧠")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
        body { background-color: #f8f9fa; }
        .stMetric {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            padding: 15px;
            border-radius: 15px;
            color: white;
            font-size: 20px;
            text-align: center;
        }
        .skill-badge {
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 5px 12px;
            margin: 5px;
            border-radius: 15px;
            font-size: 14px;
        }
        .missing-badge {
            display: inline-block;
            background: #E74C3C;
            color: white;
            padding: 5px 12px;
            margin: 5px;
            border-radius: 15px;
            font-size: 14px;
        }
        .panel { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:14px 16px; }
    </style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center; color: #2575fc;'>🧠 SkillMatch AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smart Resume & Job Role Matcher</h4>", unsafe_allow_html=True)
st.write("---")

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    return load_job_descriptions("job_descriptions.csv")

job_data = load_data()

# ------------------- PROFESSIONAL TITLE FORMATTER -------------------
import re

_SPECIAL_CAPS = {
    "ai": "AI", "ml": "ML", "ui": "UI", "ux": "UX", "qa": "QA", "devops": "DevOps",
    "nlp": "NLP", "sql": "SQL", "aws": "AWS", "gcp": "GCP", "ios": "iOS", "hr": "HR",
    "seo": "SEO", "sdet": "SDET"
}

_MERGED_MAP = {
    "datascientist": "Data Scientist",
    "dataanalyst": "Data Analyst",
    "dataengineer": "Data Engineer",
    "machinelearningengineer": "Machine Learning Engineer",
    "javadeveloper": "Java Developer",
    "pythondeveloper": "Python Developer",
    "fullstackdeveloper": "Full Stack Developer",
    "frontenddeveloper": "Front End Developer",
    "backenddeveloper": "Back End Developer",
    "uiuxdesigner": "UI/UX Designer",
    "productmanager": "Product Manager",
    "projectmanager": "Project Manager",
    "businessanalyst": "Business Analyst",
    "cybersecurityanalyst": "Cybersecurity Analyst",
    "qaengineer": "QA Engineer",
    "devopsengineer": "DevOps Engineer",
    "cloudengineer": "Cloud Engineer",
    "androiddeveloper": "Android Developer",
    "iosdeveloper": "iOS Developer",
    "webdeveloper": "Web Developer",
    "networkengineer": "Network Engineer",
    "softwaretester": "Software Tester",
    "blockchaindeveloper": "Blockchain Developer",
    "bigdataengineer": "Big Data Engineer",
    "nlpengineer": "NLP Engineer",
    "databaseadministrator": "Database Administrator",
    "uiux": "UI/UX Designer",
}

def _cap_token(tok: str) -> str:
    if not tok:
        return tok
    if "/" in tok:
        return "/".join(_cap_token(p) for p in tok.split("/"))
    if tok in _SPECIAL_CAPS:
        return _SPECIAL_CAPS[tok]
    return tok.capitalize()

def prettify_role(role: str) -> str:
    if not isinstance(role, str):
        return ""
    raw = role.strip()
    camel_split = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
    s = camel_split.replace("-", " ").replace("_", " ").strip()
    key = re.sub(r"[^A-Za-z0-9]", "", s).lower()
    if key in _MERGED_MAP:
        return _MERGED_MAP[key]
    tokens = re.sub(r"\s+", " ", s).strip().lower().split(" ")
    pretty = " ".join(_cap_token(t) for t in tokens if t)
    pretty = pretty.replace("Fullstack", "Full Stack").replace("Frontend", "Front End").replace("Backend", "Back End")
    return pretty

if "job_title" in job_data.columns:
    job_data["display_title"] = job_data["job_title"].apply(prettify_role)
else:
    st.error("CSV must include a 'job_title' column.")
    st.stop()

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📂 Upload your Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    resume_text = extract_text_from_resume(uploaded_file)
    job_role_display = st.selectbox("🎯 Select a Job Role", sorted(job_data["display_title"].unique()))

    if st.button("🚀 Match Now"):
        if job_role_display not in job_data["display_title"].values:
            st.error("❌ Invalid job role selected. Please choose a valid role.")
        else:
            selected_job = job_data[job_data["display_title"] == job_role_display].iloc[0]

            # Match score
            score = get_match_score(resume_text, selected_job["job_description"])
            score_capped = max(0, min(int(round(score)), 100))

            st.markdown(f"<h3 style='color:#6a11cb;'>🔎 Match Score</h3>", unsafe_allow_html=True)
            st.progress(score_capped)
            st.metric(label="📊 Matching Percentage", value=f"{score:.2f}%")

            # Skills comparison
            jd_skills = [s.strip() for s in str(selected_job["skills"]).split(",") if s.strip()]
            resume_skills = extract_skills(resume_text, jd_skills)
            resume_norm = {s.strip().lower() for s in resume_skills}
            missing_skills = [s for s in jd_skills if s.strip().lower() not in resume_norm]

            # ✅ Matched Skills
            st.write("### ✅ Matched Skills")
            if resume_skills:
                st.markdown("<div class='panel'>" + "".join(
                    f"<span class='skill-badge'>{s}</span>" for s in resume_skills
                ) + "</div>", unsafe_allow_html=True)
            else:
                st.info("No matching skills found.")

            # ❌ Missing Skills
            st.write("### ❌ Missing Skills")
            if missing_skills:
                st.markdown("<div class='panel'>" + "".join(
                    f"<span class='missing-badge'>{s}</span>" for s in missing_skills
                ) + "</div>", unsafe_allow_html=True)
            else:
                st.success("You have all the required skills! 🎉")

            # 📊 Bar chart
            st.write("### 📊 Skills Overview")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["Matched Skills", "Missing Skills"], [len(resume_skills), len(missing_skills)])
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Job description
            st.write("### 📄 Job Description")
            st.code(selected_job['job_description'], language="text")


            # ------------------- DOWNLOAD REPORT -------------------
            st.write("### 📥 Download Your Report")

            # CSV
            csv_df = pd.DataFrame({
                "Job Role": [job_role_display],
                "Match Score (%)": [score],
                "Matched Skills": [", ".join(resume_skills)],
                "Missing Skills": [", ".join(missing_skills)]
            })
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV Report", data=csv_bytes, file_name="skillmatch_report.csv", mime="text/csv")

            # PDF Export
            pdf_bytes = None
            try:
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.pagesizes import letter

                buf = BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = [
                    Paragraph("🧠 SkillMatch AI Report", styles["Title"]),
                    Spacer(1, 12),
                    Paragraph(f"Job Role: {job_role_display}", styles["Normal"]),
                    Paragraph(f"Match Score: {score:.2f}%", styles["Normal"]),
                    Paragraph(f"Matched Skills: {', '.join(resume_skills) if resume_skills else '—'}", styles["Normal"]),
                    Paragraph(f"Missing Skills: {', '.join(missing_skills) if missing_skills else '—'}", styles["Normal"]),
                ]
                doc.build(elements)
                pdf_bytes = buf.getvalue()
            except Exception:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(0, 10, txt="🧠 SkillMatch AI Report", ln=True, align="C")
                pdf.set_font("Arial", size=12)
                pdf.ln(4)
                pdf.multi_cell(0, 8, txt=f"Job Role: {job_role_display}")
                pdf.multi_cell(0, 8, txt=f"Match Score: {score:.2f}%")
                pdf.multi_cell(0, 8, txt=f"Matched Skills: {', '.join(resume_skills) if resume_skills else '—'}")
                pdf.multi_cell(0, 8, txt=f"Missing Skills: {', '.join(missing_skills) if missing_skills else '—'}")
                pdf_bytes = pdf.output(dest="S").encode("latin-1")

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="skillmatch_report.pdf",
                mime="application/pdf"
            )


