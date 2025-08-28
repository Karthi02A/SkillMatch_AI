import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import logging
from typing import Dict, List

# Import our enhanced utilities
from utils import (
    extract_text_from_resume, 
    load_job_descriptions, 
    get_match_score, 
    extract_skills,
    extract_skills_advanced,
    calculate_comprehensive_score,
    generate_skill_recommendations
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- PAGE SETUP -------------------
st.set_page_config(
    page_title="🧠 SkillMatch AI", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ------------------- ENHANCED CUSTOM CSS -------------------
st.markdown("""
    <style>
        /* Global Styles */
        .main { padding-top: 2rem; }
        
        /* Header Styles */
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            text-align: center;
            color: #6c757d;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        /* Skill Badges */
        .skill-badge {
            display: inline-block;
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        
        .skill-badge:hover {
            transform: translateY(-2px);
        }
        
        .missing-badge {
            display: inline-block;
            background: linear-gradient(135deg, #dc3545, #fd7e14);
            color: white;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        
        .missing-badge:hover {
            transform: translateY(-2px);
        }
        
        /* Panel Styles */
        .info-panel {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .success-panel {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 1px solid #c3e6cb;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .warning-panel {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffeaa7;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Progress Bar Custom */
        .stProgress .st-bo {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        /* File Uploader */
        .uploadedFile {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: #f8f9fa;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Custom buttons */
        .stButton button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown('<h1 class="main-header">🧠 SkillMatch AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Resume & Job Role Matching Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("### 📋 Application Info")
    st.info("""
    **SkillMatch AI** helps you:
    - 📊 Calculate resume-job compatibility
    - 🎯 Identify skill gaps
    - 📈 Get improvement recommendations
    - 📄 Generate detailed reports
    """)
    
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. **Upload** your resume (PDF/DOCX/TXT)
    2. **Select** a target job role
    3. **Click** 'Analyze Match' button
    4. **Review** your results and recommendations
    5. **Download** your detailed report
    """)
    
    st.markdown("### 📞 Support")
    st.markdown("Having issues? Check our [FAQ](#) or [Contact Support](#)")

# ------------------- DATA LOADING -------------------
@st.cache_data(ttl=3600)
def load_data():
    """Load job descriptions with error handling"""
    try:
        return load_job_descriptions("job_descriptions.csv")
    except Exception as e:
        st.error(f"❌ Error loading job data: {str(e)}")
        st.info("📝 Make sure 'job_descriptions.csv' exists with columns: job_title, skills, job_description")
        st.stop()

# Load data
with st.spinner("🔄 Loading job descriptions..."):
    job_data = load_data()

st.success(f"✅ Loaded {len(job_data)} job descriptions successfully!")

# ------------------- PROFESSIONAL TITLE FORMATTER -------------------
import re

_SPECIAL_CAPS = {
    "ai": "AI", "ml": "ML", "ui": "UI", "ux": "UX", "qa": "QA", "devops": "DevOps",
    "nlp": "NLP", "sql": "SQL", "aws": "AWS", "gcp": "GCP", "ios": "iOS", "hr": "HR",
    "seo": "SEO", "sdet": "SDET", "api": "API", "rest": "REST", "json": "JSON",
    "html": "HTML", "css": "CSS", "javascript": "JavaScript", "typescript": "TypeScript"
}

_MERGED_MAP = {
    "datascientist": "Data Scientist",
    "dataanalyst": "Data Analyst", 
    "dataengineer": "Data Engineer",
    "machinelearningengineer": "Machine Learning Engineer",
    "javadeveloper": "Java Developer",
    "pythondeveloper": "Python Developer",
    "fullstackdeveloper": "Full Stack Developer",
    "frontenddeveloper": "Frontend Developer",
    "backenddeveloper": "Backend Developer",
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
}

def _cap_token(tok: str) -> str:
    """Capitalize token with special cases"""
    if not tok:
        return tok
    if "/" in tok:
        return "/".join(_cap_token(p) for p in tok.split("/"))
    if tok.lower() in _SPECIAL_CAPS:
        return _SPECIAL_CAPS[tok.lower()]
    return tok.capitalize()

def prettify_role(role: str) -> str:
    """Convert job role to professional display format"""
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
    
    # Additional formatting
    replacements = {
        "Fullstack": "Full Stack",
        "Frontend": "Frontend", 
        "Backend": "Backend"
    }
    
    for old, new in replacements.items():
        pretty = pretty.replace(old, new)
    
    return pretty

# Apply title formatting
if "job_title" in job_data.columns:
    job_data["display_title"] = job_data["job_title"].apply(prettify_role)
else:
    st.error("❌ CSV must include a 'job_title' column.")
    st.stop()

# ------------------- MAIN APPLICATION -------------------
st.markdown("## 📁 Upload Your Resume")

# File upload with enhanced styling
uploaded_file = st.file_uploader(
    "Choose your resume file",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT (Max size: 200MB)"
)

if uploaded_file:
    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "File Size": f"{uploaded_file.size / 1024:.2f} KB",
        "File Type": uploaded_file.type
    }
    
    with st.expander("📄 File Details", expanded=False):
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
    
    # Extract resume text
    with st.spinner("📖 Processing your resume..."):
        resume_text = extract_text_from_resume(uploaded_file)
    
    if resume_text:
        st.success("✅ Resume processed successfully!")
        
        # Show text preview
        with st.expander("👀 Resume Text Preview", expanded=False):
            st.text_area(
                "Extracted Text (First 500 characters):",
                value=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                height=150,
                disabled=True
            )
        
        # Job role selection
        st.markdown("## 🎯 Select Target Job Role")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create categories for better UX
            job_roles = sorted(job_data["display_title"].unique())
            job_role_display = st.selectbox(
                "Choose the job role you're targeting:",
                options=job_roles,
                help="Select the job role that best matches your career goals"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            analyze_button = st.button("🚀 Analyze Match", use_container_width=True)
        
        # Analysis section
        if analyze_button:
            if job_role_display not in job_data["display_title"].values:
                st.error("❌ Invalid job role selected. Please choose a valid role.")
            else:
                with st.spinner("🔍 Analyzing your match..."):
                    # Get selected job data
                    selected_job = job_data[job_data["display_title"] == job_role_display].iloc[0]
                    
                    # Calculate comprehensive match score
                    try:
                        # Option 1: If calculate_comprehensive_score takes only 2 arguments
                        # Check what your utils.py function signature actually is
                        
                        # First, let's try the basic match score
                        basic_match_score = get_match_score(resume_text, selected_job["skills"])
                        
                        # Extract skills for detailed analysis
                        resume_skills = extract_skills_advanced(resume_text)
                        job_skills_text = selected_job["skills"]
                        job_skills = extract_skills_advanced(job_skills_text)
                        
                        # Calculate matched and missing skills
                        matched_skills = []
                        missing_skills = []
                        
                        for skill in job_skills:
                            skill_lower = skill.lower()
                            if any(skill_lower in resume_skill.lower() or resume_skill.lower() in skill_lower 
                                  for resume_skill in resume_skills):
                                matched_skills.append(skill)
                            else:
                                missing_skills.append(skill)
                        
                        # Calculate scores
                        if len(job_skills) > 0:
                            skill_match_percentage = (len(matched_skills) / len(job_skills)) * 100
                        else:
                            skill_match_percentage = 0
                        
                        # Try different function calls based on your utils.py implementation
                        try:
                            # Try with 3 parameters first
                            match_result = calculate_comprehensive_score(
                                resume_text, 
                                selected_job["skills"], 
                                selected_job["job_description"]
                            )
                            overall_score = match_result["overall_score"]
                            skill_match = match_result["skill_match_score"]
                            context_match = match_result["context_match_score"]
                            matched_skills = match_result["matched_skills"]
                            missing_skills = match_result["missing_skills"]
                        
                        except TypeError as e:
                            if "takes 2 positional arguments but 3 were given" in str(e):
                                # Try with 2 parameters
                                match_result = calculate_comprehensive_score(resume_text, selected_job["skills"])
                                
                                if isinstance(match_result, dict):
                                    overall_score = match_result.get("overall_score", skill_match_percentage)
                                    skill_match = match_result.get("skill_match_score", skill_match_percentage)
                                    context_match = match_result.get("context_match_score", basic_match_score)
                                    matched_skills = match_result.get("matched_skills", matched_skills)
                                    missing_skills = match_result.get("missing_skills", missing_skills)
                                else:
                                    # If it returns a single score
                                    overall_score = match_result
                                    skill_match = skill_match_percentage
                                    context_match = basic_match_score
                            else:
                                raise e
                        
                        except Exception as e:
                            # Fallback to basic calculations
                            st.warning(f"⚠️ Advanced scoring unavailable, using basic analysis: {str(e)}")
                            overall_score = skill_match_percentage
                            skill_match = skill_match_percentage
                            context_match = basic_match_score
                        
                        # Generate recommendations
                        try:
                            recommendations = generate_skill_recommendations(
                                missing_skills, 
                                selected_job["job_title"]
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Recommendations unavailable: {str(e)}")
                            recommendations = [
                                f"Consider learning {skill}" for skill in missing_skills[:5]
                            ]
                        
                        # ------------------- RESULTS DISPLAY -------------------
                        st.markdown("## 📊 Analysis Results")
                        st.markdown("---")
                        
                        # Overall Score Display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            score_color = "🟢" if overall_score >= 70 else "🟡" if overall_score >= 50 else "🔴"
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{score_color} {overall_score:.1f}%</div>
                                    <div class="metric-label">Overall Match Score</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">🎯 {skill_match:.1f}%</div>
                                    <div class="metric-label">Skill Match</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">📝 {context_match:.1f}%</div>
                                    <div class="metric-label">Context Match</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Progress bars for detailed breakdown
                        st.markdown("### 📈 Detailed Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Skills Alignment:**")
                            st.progress(skill_match / 100)
                            st.write(f"{len(matched_skills)} skills matched out of {len(job_skills)} required")
                        
                        with col2:
                            st.write("**Experience Context:**")
                            st.progress(context_match / 100)
                            st.write("Based on resume content analysis")
                        
                        # Skills Analysis
                        st.markdown("### 🎯 Skills Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ✅ Matched Skills")
                            if matched_skills:
                                skills_html = "".join([f'<span class="skill-badge">{skill}</span>' for skill in matched_skills[:15]])
                                st.markdown(f'<div class="success-panel">{skills_html}</div>', unsafe_allow_html=True)
                                
                                if len(matched_skills) > 15:
                                    with st.expander(f"Show all {len(matched_skills)} matched skills"):
                                        remaining_skills = "".join([f'<span class="skill-badge">{skill}</span>' for skill in matched_skills[15:]])
                                        st.markdown(remaining_skills, unsafe_allow_html=True)
                            else:
                                st.warning("No skills matched. Consider updating your resume with relevant keywords.")
                        
                        with col2:
                            st.markdown("#### ❌ Missing Skills")
                            if missing_skills:
                                missing_html = "".join([f'<span class="missing-badge">{skill}</span>' for skill in missing_skills[:15]])
                                st.markdown(f'<div class="warning-panel">{missing_html}</div>', unsafe_allow_html=True)
                                
                                if len(missing_skills) > 15:
                                    with st.expander(f"Show all {len(missing_skills)} missing skills"):
                                        remaining_missing = "".join([f'<span class="missing-badge">{skill}</span>' for skill in missing_skills[15:]])
                                        st.markdown(remaining_missing, unsafe_allow_html=True)
                            else:
                                st.success("🎉 Great! You have all the required skills!")
                        
                        # Recommendations Section
                        if recommendations:
                            st.markdown("### 💡 Improvement Recommendations")
                            st.markdown('<div class="info-panel">', unsafe_allow_html=True)
                            
                            for i, rec in enumerate(recommendations[:5], 1):
                                st.markdown(f"**{i}.** {rec}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualization Section
                        st.markdown("### 📊 Visual Analysis")
                        
                        # Create visualizations
                        fig_col1, fig_col2 = st.columns(2)
                        
                        with fig_col1:
                            # Score breakdown pie chart
                            if matched_skills or missing_skills:
                                labels = ['Matched Skills', 'Missing Skills']
                                values = [len(matched_skills), len(missing_skills)]
                                colors = ['#28a745', '#dc3545']
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=labels, 
                                    values=values,
                                    hole=.3,
                                    marker_colors=colors
                                )])
                                fig.update_layout(
                                    title="Skills Distribution",
                                    font=dict(size=14),
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No skills data available for visualization")
                        
                        with fig_col2:
                            # Score comparison bar chart
                            categories = ['Skill Match', 'Context Match', 'Overall Score']
                            scores = [skill_match, context_match, overall_score]
                            
                            fig = go.Figure([go.Bar(
                                x=categories,
                                y=scores,
                                marker_color=['#667eea', '#764ba2', '#28a745']
                            )])
                            fig.update_layout(
                                title="Score Breakdown",
                                yaxis=dict(range=[0, 100]),
                                font=dict(size=14),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed Job Requirements
                        with st.expander("📋 View Full Job Requirements", expanded=False):
                            st.markdown("**Job Title:** " + prettify_role(selected_job["job_title"]))
                            st.markdown("**Required Skills:**")
                            st.write(selected_job["skills"])
                            if "job_description" in selected_job.index:
                                st.markdown("**Job Description:**")
                                st.write(selected_job["job_description"])
                        
                        # Export functionality
                        st.markdown("### 💾 Export Results")
                        
                        # Prepare report data
                        report_data = {
                            "Analysis Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Target Job Role": job_role_display,
                            "Overall Match Score": f"{overall_score:.1f}%",
                            "Skill Match Score": f"{skill_match:.1f}%",
                            "Context Match Score": f"{context_match:.1f}%",
                            "Matched Skills": ", ".join(matched_skills),
                            "Missing Skills": ", ".join(missing_skills),
                            "Recommendations": " | ".join(recommendations[:5])
                        }
                        
                        # Convert to DataFrame for export
                        report_df = pd.DataFrame([report_data])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_buffer = BytesIO()
                            report_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="📄 Download CSV Report",
                                data=csv_buffer.getvalue(),
                                file_name=f"skillmatch_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Create a summary text report
                            text_report = f"""
SKILLMATCH AI ANALYSIS REPORT
================================
Date: {report_data['Analysis Date']}
Target Role: {report_data['Target Job Role']}

SCORES:
- Overall Match: {report_data['Overall Match Score']}
- Skill Match: {report_data['Skill Match Score']}  
- Context Match: {report_data['Context Match Score']}

MATCHED SKILLS:
{report_data['Matched Skills']}

MISSING SKILLS:
{report_data['Missing Skills']}

RECOMMENDATIONS:
{chr(10).join([f"- {rec}" for rec in recommendations[:5]])}

Generated by SkillMatch AI
                            """
                            
                            st.download_button(
                                label="📝 Download Text Report",
                                data=text_report,
                                file_name=f"skillmatch_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")
                        
                        # Debug information
                        st.info("🔧 Debug Information:")
                        st.write(f"Function signature error: {type(e).__name__}")
                        st.write("Please check your utils.py file functions.")
                        
                        # Show available functions for debugging
                        try:
                            from utils import calculate_comprehensive_score
                            import inspect
                            sig = inspect.signature(calculate_comprehensive_score)
                            st.write(f"calculate_comprehensive_score signature: {sig}")
                        except Exception as debug_e:
                            st.write(f"Could not inspect function: {debug_e}")
