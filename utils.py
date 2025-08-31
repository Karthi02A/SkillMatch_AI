import re
import pandas as pd
import spacy
import logging
import functools
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- CONFIGURATION -------------------
class Config:
    MAX_TEXT_LENGTH = 5000
    FUZZY_MATCH_THRESHOLD = 0.8
    CACHE_SIZE = 100
    JOB_ROLE_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity for valid job role

# ------------------- RESUME PROCESSOR CLASS -------------------
class ResumeProcessor:
    def __init__(self):
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load spaCy model with better error handling"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            try:
                import en_core_web_sm
                self.nlp = en_core_web_sm.load()
                logger.info("spaCy model loaded via module import")
            except ImportError:
                error_msg = (
                    "❌ spaCy model 'en_core_web_sm' not found. "
                    "Make sure it's listed in requirements.txt:\n\n"
                    "en_core_web_sm @ https://github.com/explosion/spacy-models/"
                    "releases/download/en_core_web_sm-3.7.1/"
                    "en_core_web_sm-3.7.1-py3-none-any.whl"
                )
                logger.error(error_msg)
                raise OSError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy model: {e}")
            raise
    
    @functools.lru_cache(maxsize=Config.CACHE_SIZE)
    def get_similarity_score(self, resume_text: str, job_desc: str) -> float:
        """Cached similarity calculation with performance optimization"""
        if not self.nlp or not resume_text.strip() or not job_desc.strip():
            return 0.0
            
        try:
            # Truncate long texts for better performance
            resume_doc = self.nlp(resume_text.lower()[:Config.MAX_TEXT_LENGTH])
            job_doc = self.nlp(job_desc.lower()[:Config.MAX_TEXT_LENGTH])
            
            similarity = resume_doc.similarity(job_doc)
            return round(similarity * 100, 2)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

# Global processor instance
processor = ResumeProcessor()

# ------------------- JOB ROLE VALIDATION -------------------
def validate_job_role(selected_job_role: str, job_descriptions_df: pd.DataFrame) -> Dict:
    """
    Validate if the selected job role exists in the job descriptions dataset
    
    Args:
        selected_job_role (str): The job role selected by user
        job_descriptions_df (pd.DataFrame): DataFrame containing job descriptions
        
    Returns:
        dict: Validation result with status and message
    """
    try:
        if not selected_job_role or not isinstance(selected_job_role, str):
            return {
                "is_valid": False,
                "message": "❌ Please select a valid job role",
                "suggested_roles": []
            }
        
        if job_descriptions_df.empty:
            return {
                "is_valid": False,
                "message": "❌ No job descriptions data available",
                "suggested_roles": []
            }
        
        # Get available job titles
        available_jobs = job_descriptions_df['job_title'].str.strip().str.lower().tolist()
        selected_job_lower = selected_job_role.strip().lower()
        
        # Check for exact match
        if selected_job_lower in available_jobs:
            return {
                "is_valid": True,
                "message": "✅ Valid job role selected",
                "suggested_roles": []
            }
        
        # Check for fuzzy match with available job titles
        best_matches = []
        for job in available_jobs:
            if job:  # Skip empty values
                similarity = SequenceMatcher(None, selected_job_lower, job).ratio()
                if similarity > Config.JOB_ROLE_SIMILARITY_THRESHOLD:
                    best_matches.append((job, similarity))
        
        # Sort by similarity and get top 3 suggestions
        best_matches.sort(key=lambda x: x[1], reverse=True)
        suggested_roles = [match[0].title() for match in best_matches[:3]]
        
        if suggested_roles:
            return {
                "is_valid": False,
                "message": f"❌ Invalid job role: '{selected_job_role}'. Did you mean one of these?",
                "suggested_roles": suggested_roles
            }
        else:
            # Get random sample of available roles for suggestion
            available_roles_sample = job_descriptions_df['job_title'].dropna().sample(
                min(5, len(job_descriptions_df))
            ).tolist()
            
            return {
                "is_valid": False,
                "message": f"❌ Invalid job role: '{selected_job_role}'. This role is not available in our database.",
                "suggested_roles": available_roles_sample
            }
            
    except Exception as e:
        logger.error(f"Error validating job role: {e}")
        return {
            "is_valid": False,
            "message": f"❌ Error validating job role: {str(e)}",
            "suggested_roles": []
        }

def get_job_details(job_title: str, job_descriptions_df: pd.DataFrame) -> Optional[Dict]:
    """
    Get job details for a specific job title
    
    Args:
        job_title (str): The job title to search for
        job_descriptions_df (pd.DataFrame): DataFrame containing job descriptions
        
    Returns:
        dict or None: Job details if found, None otherwise
    """
    try:
        if not job_title or job_descriptions_df.empty:
            return None
        
        # Find matching job (case-insensitive)
        job_title_lower = job_title.strip().lower()
        matching_jobs = job_descriptions_df[
            job_descriptions_df['job_title'].str.lower().str.strip() == job_title_lower
        ]
        
        if matching_jobs.empty:
            return None
        
        # Return first match
        job_data = matching_jobs.iloc[0]
        return {
            "job_title": job_data['job_title'],
            "skills": job_data['skills'],
            "job_description": job_data['job_description']
        }
        
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        return None

# ------------------- RESUME TEXT EXTRACTION -------------------
def extract_text_from_resume(uploaded_file) -> str:
    """Extract raw text from PDF, DOCX, or TXT resumes with robust error handling."""
    if not uploaded_file:
        return ""
    
    try:
        text = ""
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        if file_type == "pdf":
            try:
                from PyPDF2 import PdfReader
                pdf = PdfReader(uploaded_file)
                
                if len(pdf.pages) == 0:
                    st.error("❌ PDF file appears to be empty or corrupted")
                    return ""
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error reading page {page_num + 1}: {e}")
                        continue
                        
            except Exception as e:
                st.error(f"❌ Error reading PDF file: {str(e)}")
                return ""

        elif file_type == "docx":
            try:
                import docx
                doc = docx.Document(uploaded_file)
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                text = "\n".join(paragraphs)
                
                if not text.strip():
                    st.warning("⚠️ DOCX file appears to be empty")
                    
            except Exception as e:
                st.error(f"❌ Error reading DOCX file: {str(e)}")
                return ""

        elif file_type == "txt":
            try:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    text = uploaded_file.read().decode("latin-1", errors="ignore")
                except Exception as e:
                    st.error(f"❌ Error reading TXT file: {str(e)}")
                    return ""
            except Exception as e:
                st.error(f"❌ Error processing TXT file: {str(e)}")
                return ""
        else:
            st.error(f"❌ Unsupported file type: {file_type}")
            return ""

        if not text.strip():
            st.warning("⚠️ No text content found in the uploaded file")
            return ""
            
        logger.info(f"Successfully extracted {len(text)} characters from {file_type.upper()} file")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Unexpected error in text extraction: {e}")
        st.error(f"❌ Unexpected error processing file: {str(e)}")
        return ""

# ------------------- JOB DESCRIPTIONS LOADER -------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_job_descriptions(file_path: str) -> pd.DataFrame:
    """
    Enhanced job descriptions loader with validation and preprocessing.
    Maps to: job_title, skills, job_description
    """
    try:
        df = pd.read_csv(file_path)
        
        # Data validation
        if df.empty:
            raise ValueError("❌ CSV file is empty")
            
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Normalize column names (strip + lowercase + replace spaces)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

        # Handle missing values
        df = df.fillna("")
        
        # Map known columns to standard names
        column_mapping = {
            "job_role": "job_title",
            "skill": "skills", 
            "job_descriptions": "job_description"
        }
        df.rename(columns=column_mapping, inplace=True)

        # Validate required columns
        required_cols = {"job_title", "skills", "job_description"}
        available_cols = set(df.columns)
        missing_cols = required_cols - available_cols
        
        if missing_cols:
            raise ValueError(
                f"❌ Missing required columns: {missing_cols}. "
                f"Available columns: {list(available_cols)}"
            )

        # Clean and validate data
        result_df = df[list(required_cols)].copy()
        
        # Remove empty rows
        initial_rows = len(result_df)
        result_df = result_df[
            (result_df["job_title"].str.strip() != "") & 
            (result_df["job_description"].str.strip() != "")
        ]
        
        if len(result_df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(result_df)} empty rows")
        
        if result_df.empty:
            raise ValueError("❌ No valid job descriptions found after cleaning")
            
        logger.info(f"Successfully processed {len(result_df)} valid job descriptions")
        return result_df

    except FileNotFoundError:
        error_msg = f"❌ Job descriptions file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except pd.errors.EmptyDataError:
        error_msg = "❌ CSV file is empty or corrupted"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"Error loading job descriptions: {e}")
        raise ValueError(f"❌ Error loading job descriptions: {str(e)}")

# ------------------- ENHANCED MATCHING SCORE WITH VALIDATION -------------------
def get_match_score(resume_text: str, job_desc: str) -> float:
    """Compute similarity score (0–100) using spaCy embeddings with caching."""
    if not resume_text or not job_desc:
        return 0.0
    
    try:
        return processor.get_similarity_score(resume_text, job_desc)
    except Exception as e:
        logger.error(f"Error calculating match score: {e}")
        return 0.0

def calculate_resume_job_match(resume_text: str, selected_job_role: str, job_descriptions_df: pd.DataFrame) -> Dict:
    """
    Main function to calculate resume-job matching with job role validation
    
    Args:
        resume_text (str): Extracted text from resume
        selected_job_role (str): Job role selected by user
        job_descriptions_df (pd.DataFrame): DataFrame containing job descriptions
        
    Returns:
        dict: Complete analysis result with validation status
    """
    try:
        # Step 1: Validate job role
        validation_result = validate_job_role(selected_job_role, job_descriptions_df)
        
        if not validation_result["is_valid"]:
            return {
                "is_valid_job": False,
                "validation_message": validation_result["message"],
                "suggested_roles": validation_result["suggested_roles"],
                "overall_score": None,
                "skill_match_score": None,
                "context_match_score": None,
                "matched_skills": [],
                "missing_skills": [],
                "recommendations": []
            }
        
        # Step 2: Get job details
        job_details = get_job_details(selected_job_role, job_descriptions_df)
        
        if not job_details:
            return {
                "is_valid_job": False,
                "validation_message": f"❌ Could not find details for job role: '{selected_job_role}'",
                "suggested_roles": job_descriptions_df['job_title'].head(5).tolist(),
                "overall_score": None,
                "skill_match_score": None,
                "context_match_score": None,
                "matched_skills": [],
                "missing_skills": [],
                "recommendations": []
            }
        
        # Step 3: Calculate comprehensive score
        score_result = calculate_comprehensive_score(
            resume_text, 
            job_details["skills"], 
            job_details["job_description"]
        )
        
        # Step 4: Generate recommendations
        recommendations = generate_skill_recommendations(
            score_result["missing_skills"], 
            selected_job_role
        )
        
        return {
            "is_valid_job": True,
            "validation_message": "✅ Valid job role",
            "job_details": job_details,
            "overall_score": score_result["overall_score"],
            "skill_match_score": score_result["skill_match_score"],
            "context_match_score": score_result["context_match_score"],
            "matched_skills": score_result["matched_skills"],
            "missing_skills": score_result["missing_skills"],
            "total_skills": score_result["total_skills"],
            "matched_count": score_result["matched_count"],
            "recommendations": recommendations,
            "suggested_roles": []
        }
        
    except Exception as e:
        logger.error(f"Error in resume-job matching: {e}")
        return {
            "is_valid_job": False,
            "validation_message": f"❌ Error processing job match: {str(e)}",
            "suggested_roles": [],
            "overall_score": None,
            "skill_match_score": None,
            "context_match_score": None,
            "matched_skills": [],
            "missing_skills": [],
            "recommendations": []
        }

# ------------------- ADVANCED SKILL EXTRACTION -------------------
def extract_skills_advanced(text: str) -> List[str]:
    """Enhanced skill extraction using NLP and pattern matching"""
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    skills_found = []
    
    # Common technical skills patterns
    skill_patterns = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
        'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind',
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sql',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'machine learning', 'deep learning', 'artificial intelligence', 'nlp',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api', 'rest', 'graphql',
        'ui/ux', 'figma', 'sketch', 'adobe', 'photoshop'
    }
    
    # Extract skills using word boundaries
    for skill in skill_patterns:
        if re.search(rf'\b{re.escape(skill.lower())}\b', text_lower):
            skills_found.append(skill.title())
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(skills_found))

def extract_skills(resume_text: str, jd_skills: str) -> List[str]:
    """Extract matching skills between resume and job description"""
    if not resume_text or not jd_skills:
        return []
    
    # Parse job skills
    job_skills_list = [s.strip() for s in str(jd_skills).split(",") if s.strip()]
    resume_text_lower = resume_text.lower()
    
    matched_skills = []
    
    for skill in job_skills_list:
        skill_lower = skill.strip().lower()
        if not skill_lower:
            continue
            
        # Check for exact match or partial match
        if (re.search(rf'\b{re.escape(skill_lower)}\b', resume_text_lower) or
            skill_lower in resume_text_lower):
            matched_skills.append(skill)
            continue
            
        # Fuzzy matching for similar skills
        words = resume_text_lower.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                similarity = SequenceMatcher(None, skill_lower, word).ratio()
                if similarity > Config.FUZZY_MATCH_THRESHOLD:
                    matched_skills.append(skill)
                    break
    
    return matched_skills

# ------------------- COMPREHENSIVE SCORING - FIXED VERSION -------------------
def calculate_comprehensive_score(resume_text: str, job_skills: str, job_description: str = None) -> Dict:
    """
    Calculate comprehensive matching score between resume and job requirements
    
    Args:
        resume_text (str): The extracted text from resume
        job_skills (str): Required skills for the job (comma-separated)
        job_description (str, optional): Job description text
        
    Returns:
        dict: Dictionary containing various scores and matched/missing skills
    """
    try:
        if not resume_text or not job_skills:
            return {
                "overall_score": 0.0,
                "skill_match_score": 0.0,
                "context_match_score": 0.0,
                "matched_skills": [],
                "missing_skills": []
            }
        
        # Extract and match skills
        job_skills_list = [s.strip() for s in str(job_skills).split(",") if s.strip()]
        matched_skills = extract_skills(resume_text, job_skills)
        missing_skills = [skill for skill in job_skills_list if skill not in matched_skills]
        
        # Calculate skill match percentage
        if job_skills_list:
            skill_match_percentage = (len(matched_skills) / len(job_skills_list)) * 100
        else:
            skill_match_percentage = 0.0
        
        # Calculate context match using job description if available
        if job_description:
            context_match_percentage = get_match_score(resume_text, job_description)
        else:
            # Fallback to skills-based context matching
            skills_text = " ".join(job_skills_list)
            context_match_percentage = get_match_score(resume_text, skills_text)
        
        # Calculate overall score (weighted average)
        # 60% skill match + 40% context match
        overall_score = (skill_match_percentage * 0.6) + (context_match_percentage * 0.4)
        
        return {
            "overall_score": round(overall_score, 2),
            "skill_match_score": round(skill_match_percentage, 2),
            "context_match_score": round(context_match_percentage, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "total_skills": len(job_skills_list),
            "matched_count": len(matched_skills)
        }
        
    except Exception as e:
        logger.error(f"Error calculating comprehensive score: {e}")
        return {
            "overall_score": 0.0,
            "skill_match_score": 0.0,
            "context_match_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "total_skills": 0,
            "matched_count": 0
        }

def calculate_keyword_match(resume_text: str, keywords: str) -> float:
    """Calculate keyword matching score"""
    if not keywords or not resume_text:
        return 0.0
    
    try:
        keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        if not keyword_list:
            return 0.0
            
        resume_lower = resume_text.lower()
        matched_keywords = 0
        
        for keyword in keyword_list:
            if keyword in resume_lower:
                matched_keywords += 1
        
        return (matched_keywords / len(keyword_list)) * 100
        
    except Exception as e:
        logger.error(f"Error calculating keyword match: {e}")
        return 0.0

# ------------------- SKILL RECOMMENDATIONS -------------------
def generate_skill_recommendations(missing_skills: List[str], job_title: str = "") -> List[str]:
    """Generate learning recommendations for missing skills"""
    if not missing_skills:
        return []
    
    recommendations = []
    
    skill_resources = {
        'python': 'Learn Python through Python.org tutorials, Codecademy, or Real Python',
        'machine learning': 'Start with Coursera\'s ML course, Kaggle Learn, or Fast.ai',
        'javascript': 'Master JavaScript with MDN Web Docs, FreeCodeCamp, or JavaScript.info',
        'react': 'Build React skills using official React docs, Scrimba, or React tutorials',
        'sql': 'Practice SQL with W3Schools, SQLBolt, or HackerRank SQL challenges',
        'aws': 'Get AWS certified through AWS Training, A Cloud Guru, or official AWS docs',
        'docker': 'Learn containerization with Docker documentation and Docker Hub tutorials',
        'kubernetes': 'Master orchestration with Kubernetes official tutorials and hands-on labs',
        'node.js': 'Build backend skills with Node.js docs and Express.js tutorials',
        'git': 'Version control mastery through Git documentation and GitHub Learning Lab'
    }
    
    for skill in missing_skills[:5]:  # Limit to top 5 missing skills
        skill_lower = skill.lower()
        
        # Check for exact or partial matches
        recommendation = None
        for key, value in skill_resources.items():
            if key in skill_lower or skill_lower in key:
                recommendation = value
                break
        
        if recommendation:
            recommendations.append(f"📚 **{skill}**: {recommendation}")
        else:
            recommendations.append(f"📚 **{skill}**: Search for online courses on Udemy, Coursera, or YouTube")
    
    # Add general advice based on job title
    if job_title and len(recommendations) < 5:
        if any(term in job_title.lower() for term in ['data', 'analyst', 'scientist']):
            recommendations.append("📊 Focus on data analysis tools and statistical knowledge")
        elif any(term in job_title.lower() for term in ['developer', 'engineer', 'programmer']):
            recommendations.append("💻 Practice coding challenges on LeetCode or HackerRank")
        elif any(term in job_title.lower() for term in ['manager', 'lead']):
            recommendations.append("👥 Develop leadership and project management skills")
    
    return recommendations

# ------------------- UTILITY FUNCTIONS FOR UI -------------------
def display_validation_message(validation_result: Dict):
    """Display validation message in Streamlit UI"""
    if validation_result["is_valid_job"]:
        st.success(validation_result["validation_message"])
    else:
        st.error(validation_result["validation_message"])
        
        # Show suggested roles if available
        if validation_result["suggested_roles"]:
            st.info("**Available job roles you might be looking for:**")
            for role in validation_result["suggested_roles"]:
                st.write(f"• {role}")

def get_score_color(score: float) -> str:
    """Get color based on score for better UI visualization"""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "orange"
    elif score >= 40:
        return "red"
    else:
        return "darkred"

def format_score_display(score: float, label: str) -> str:
    """Format score for display with color coding"""
    color = get_score_color(score)
    return f":{color}[**{label}**: {score}%]"
