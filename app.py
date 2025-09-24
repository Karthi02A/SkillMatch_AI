import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import logging
from typing import Dict, List, Optional
import os
import sys
import time
import subprocess
import re

# Fix spaCy model installation for Streamlit Cloud
def ensure_spacy_model():
    """Ensure spaCy model is available with fallback handling"""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except (OSError, IOError):
        try:
            st.info("Installing language model... This may take a moment.")
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                st.success("Language model installed successfully!")
                return True
            else:
                st.warning("Could not install spaCy model. Using fallback methods.")
                return False
        except Exception as e:
            st.warning(f"spaCy installation failed: {e}. Using basic analysis.")
            return False
    except Exception as e:
        st.error(f"Unexpected error with spaCy: {e}")
        return False

# Run this before your main app
if 'spacy_loaded' not in st.session_state:
    st.session_state.spacy_loaded = ensure_spacy_model()

# Import our enhanced utilities with error handling
try:
    from utils import (
        extract_text_from_resume, 
        load_job_descriptions, 
        get_match_score, 
        extract_skills,
        extract_skills_advanced,
        calculate_comprehensive_score,
        generate_skill_recommendations
    )
except ImportError as e:
    st.error(f"Error importing utilities: {e}")
    st.info("Please ensure all required utility functions are available in utils.py")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- PAGE SETUP -------------------
st.set_page_config(
    page_title="SkillMatch AI", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ------------------- ENHANCED JOB TITLE FORMATTER -------------------
# Comprehensive mapping for professional job titles
PROFESSIONAL_TITLE_MAP = {
    # Data & Analytics Roles
    "data scientist": "Data Scientist",
    "datascientist": "Data Scientist",
    "data_scientist": "Data Scientist",
    "senior data scientist": "Senior Data Scientist",
    "principal data scientist": "Principal Data Scientist",
    "lead data scientist": "Lead Data Scientist",
    "data analyst": "Data Analyst",
    "dataanalyst": "Data Analyst",
    "data_analyst": "Data Analyst",
    "senior data analyst": "Senior Data Analyst",
    "business analyst": "Business Analyst",
    "businessanalyst": "Business Analyst",
    "business_analyst": "Business Analyst",
    "senior business analyst": "Senior Business Analyst",
    "data engineer": "Data Engineer",
    "dataengineer": "Data Engineer",
    "data_engineer": "Data Engineer",
    "senior data engineer": "Senior Data Engineer",
    "big data engineer": "Big Data Engineer",
    "bigdataengineer": "Big Data Engineer",
    "analytics manager": "Analytics Manager",
    "data science manager": "Data Science Manager",
    "quantitative analyst": "Quantitative Analyst",
    "research scientist": "Research Scientist",
    
    # Machine Learning & AI
    "machine learning engineer": "Machine Learning Engineer",
    "machinelearningengineer": "Machine Learning Engineer",
    "ml engineer": "ML Engineer",
    "mlengineer": "ML Engineer",
    "ai engineer": "AI Engineer",
    "artificial intelligence engineer": "Artificial Intelligence Engineer",
    "deep learning engineer": "Deep Learning Engineer",
    "computer vision engineer": "Computer Vision Engineer",
    "nlp engineer": "NLP Engineer",
    "natural language processing engineer": "Natural Language Processing Engineer",
    "ai researcher": "AI Researcher",
    "machine learning scientist": "Machine Learning Scientist",
    "ai specialist": "AI Specialist",
    
    # Software Development
    "software engineer": "Software Engineer",
    "software developer": "Software Developer",
    "senior software engineer": "Senior Software Engineer",
    "principal software engineer": "Principal Software Engineer",
    "lead software engineer": "Lead Software Engineer",
    "full stack developer": "Full Stack Developer",
    "fullstack developer": "Full Stack Developer",
    "fullstackdeveloper": "Full Stack Developer",
    "frontend developer": "Frontend Developer",
    "front end developer": "Frontend Developer",
    "frontenddeveloper": "Frontend Developer",
    "backend developer": "Backend Developer",
    "back end developer": "Backend Developer",
    "backenddeveloper": "Backend Developer",
    "web developer": "Web Developer",
    "webdeveloper": "Web Developer",
    "mobile developer": "Mobile Developer",
    "android developer": "Android Developer",
    "androiddeveloper": "Android Developer",
    "ios developer": "iOS Developer",
    "iosdeveloper": "iOS Developer",
    "react developer": "React Developer",
    "angular developer": "Angular Developer",
    "vue developer": "Vue Developer",
    "node js developer": "Node.js Developer",
    "nodejs developer": "Node.js Developer",
    "python developer": "Python Developer",
    "pythondeveloper": "Python Developer",
    "java developer": "Java Developer",
    "javadeveloper": "Java Developer",
    "javascript developer": "JavaScript Developer",
    "c# developer": "C# Developer",
    "c sharp developer": "C# Developer",
    ".net developer": ".NET Developer",
    "dotnet developer": ".NET Developer",
    "php developer": "PHP Developer",
    "ruby developer": "Ruby Developer",
    "go developer": "Go Developer",
    "golang developer": "Go Developer",
    "rust developer": "Rust Developer",
    "scala developer": "Scala Developer",
    "kotlin developer": "Kotlin Developer",
    
    # DevOps & Infrastructure
    "devops engineer": "DevOps Engineer",
    "devopsengineer": "DevOps Engineer",
    "site reliability engineer": "Site Reliability Engineer",
    "sre": "Site Reliability Engineer",
    "platform engineer": "Platform Engineer",
    "infrastructure engineer": "Infrastructure Engineer",
    "cloud engineer": "Cloud Engineer",
    "cloudengineer": "Cloud Engineer",
    "aws engineer": "AWS Engineer",
    "azure engineer": "Azure Engineer",
    "gcp engineer": "GCP Engineer",
    "kubernetes engineer": "Kubernetes Engineer",
    "docker engineer": "Docker Engineer",
    "systems engineer": "Systems Engineer",
    "network engineer": "Network Engineer",
    "networkengineer": "Network Engineer",
    "security engineer": "Security Engineer",
    "cybersecurity engineer": "Cybersecurity Engineer",
    "information security analyst": "Information Security Analyst",
    
    # Quality Assurance & Testing
    "qa engineer": "QA Engineer",
    "qaengineer": "QA Engineer",
    "quality assurance engineer": "Quality Assurance Engineer",
    "test engineer": "Test Engineer",
    "software tester": "Software Tester",
    "softwaretester": "Software Tester",
    "automation engineer": "Automation Engineer",
    "sdet": "Software Development Engineer in Test",
    "qa analyst": "QA Analyst",
    "test analyst": "Test Analyst",
    "performance tester": "Performance Tester",
    "manual tester": "Manual Tester",
    "automation tester": "Automation Tester",
    
    # Product & Management
    "product manager": "Product Manager",
    "productmanager": "Product Manager",
    "senior product manager": "Senior Product Manager",
    "principal product manager": "Principal Product Manager",
    "product owner": "Product Owner",
    "project manager": "Project Manager",
    "projectmanager": "Project Manager",
    "program manager": "Program Manager",
    "scrum master": "Scrum Master",
    "agile coach": "Agile Coach",
    "delivery manager": "Delivery Manager",
    "technical product manager": "Technical Product Manager",
    "product marketing manager": "Product Marketing Manager",
    
    # Design & UX
    "ui ux designer": "UI/UX Designer",
    "ui/ux designer": "UI/UX Designer",
    "uiux designer": "UI/UX Designer",
    "uiuxdesigner": "UI/UX Designer",
    "ux designer": "UX Designer",
    "ui designer": "UI Designer",
    "product designer": "Product Designer",
    "visual designer": "Visual Designer",
    "graphic designer": "Graphic Designer",
    "web designer": "Web Designer",
    "interaction designer": "Interaction Designer",
    "user researcher": "User Researcher",
    "ux researcher": "UX Researcher",
    "design system designer": "Design System Designer",
    
    # Database & Systems
    "database administrator": "Database Administrator",
    "databaseadministrator": "Database Administrator",
    "dba": "Database Administrator",
    "database engineer": "Database Engineer",
    "sql developer": "SQL Developer",
    "mongodb developer": "MongoDB Developer",
    "postgresql developer": "PostgreSQL Developer",
    "mysql developer": "MySQL Developer",
    "oracle developer": "Oracle Developer",
    "systems administrator": "Systems Administrator",
    "system administrator": "Systems Administrator",
    "linux administrator": "Linux Administrator",
    "windows administrator": "Windows Administrator",
    
    # Specialized Technical Roles
    "blockchain developer": "Blockchain Developer",
    "blockchaindeveloper": "Blockchain Developer",
    "smart contract developer": "Smart Contract Developer",
    "ethereum developer": "Ethereum Developer",
    "solidity developer": "Solidity Developer",
    "game developer": "Game Developer",
    "unity developer": "Unity Developer",
    "unreal engine developer": "Unreal Engine Developer",
    "vr developer": "VR Developer",
    "ar developer": "AR Developer",
    "embedded systems engineer": "Embedded Systems Engineer",
    "firmware engineer": "Firmware Engineer",
    "hardware engineer": "Hardware Engineer",
    "robotics engineer": "Robotics Engineer",
    "iot engineer": "IoT Engineer",
    "edge computing engineer": "Edge Computing Engineer",
    
    # Architecture & Leadership
    "software architect": "Software Architect",
    "solution architect": "Solutions Architect",
    "enterprise architect": "Enterprise Architect",
    "cloud architect": "Cloud Architect",
    "data architect": "Data Architect",
    "security architect": "Security Architect",
    "technical lead": "Technical Lead",
    "tech lead": "Technical Lead",
    "engineering manager": "Engineering Manager",
    "development manager": "Development Manager",
    "technical manager": "Technical Manager",
    "cto": "Chief Technology Officer",
    "chief technology officer": "Chief Technology Officer",
    "vp engineering": "VP of Engineering",
    "vice president engineering": "VP of Engineering",
    "director of engineering": "Director of Engineering",
    
    # Sales & Marketing (Tech)
    "technical sales engineer": "Technical Sales Engineer",
    "sales engineer": "Sales Engineer",
    "solution consultant": "Solutions Consultant",
    "technical consultant": "Technical Consultant",
    "implementation specialist": "Implementation Specialist",
    "customer success engineer": "Customer Success Engineer",
    "developer advocate": "Developer Advocate",
    "developer relations": "Developer Relations",
    "technical writer": "Technical Writer",
    "documentation specialist": "Documentation Specialist",
    
    # Support & Operations
    "technical support engineer": "Technical Support Engineer",
    "support engineer": "Support Engineer",
    "customer support specialist": "Customer Support Specialist",
    "it support specialist": "IT Support Specialist",
    "help desk technician": "Help Desk Technician",
    "it technician": "IT Technician",
    "field service engineer": "Field Service Engineer",
    "operations engineer": "Operations Engineer",
    "production support engineer": "Production Support Engineer",
}

# Special capitalization rules
SPECIAL_CAPS = {
    "ai": "AI", "ml": "ML", "ui": "UI", "ux": "UX", "qa": "QA", "devops": "DevOps",
    "nlp": "NLP", "sql": "SQL", "aws": "AWS", "gcp": "GCP", "ios": "iOS", "hr": "HR",
    "seo": "SEO", "sdet": "SDET", "api": "API", "rest": "REST", "json": "JSON",
    "html": "HTML", "css": "CSS", "javascript": "JavaScript", "typescript": "TypeScript",
    "mongodb": "MongoDB", "postgresql": "PostgreSQL", "mysql": "MySQL", "nosql": "NoSQL",
    "cicd": "CI/CD", "ci/cd": "CI/CD", "oauth": "OAuth", "jwt": "JWT", "xml": "XML",
    "saas": "SaaS", "paas": "PaaS", "iaas": "IaaS", "crm": "CRM", "erp": "ERP",
    "bi": "BI", "etl": "ETL", "olap": "OLAP", "oltp": "OLTP", "crud": "CRUD",
    "sdk": "SDK", "ide": "IDE", "gui": "GUI", "cli": "CLI", "ssh": "SSH",
    "ssl": "SSL", "tls": "TLS", "https": "HTTPS", "http": "HTTP", "ftp": "FTP",
    "tcp": "TCP", "udp": "UDP", "ip": "IP", "dns": "DNS", "cdn": "CDN",
    "vpn": "VPN", "lan": "LAN", "wan": "WAN", "wifi": "WiFi", "iot": "IoT",
    "ar": "AR", "vr": "VR", "xr": "XR", "3d": "3D", "2d": "2D",
    "gpu": "GPU", "cpu": "CPU", "ram": "RAM", "ssd": "SSD", "hdd": "HDD",
    "os": "OS", "linux": "Linux", "unix": "Unix", "windows": "Windows",
    "macos": "macOS", "android": "Android", "kubernetes": "Kubernetes",
    "docker": "Docker", "git": "Git", "github": "GitHub", "gitlab": "GitLab",
    "jenkins": "Jenkins", "terraform": "Terraform", "ansible": "Ansible",
    "puppet": "Puppet", "chef": "Chef", "nagios": "Nagios", "splunk": "Splunk",
    "elasticsearch": "Elasticsearch", "kibana": "Kibana", "logstash": "Logstash",
    "redis": "Redis", "memcached": "Memcached", "nginx": "Nginx", "apache": "Apache",
    "tomcat": "Tomcat", "jboss": "JBoss", "websphere": "WebSphere",
    "spring": "Spring", "hibernate": "Hibernate", "struts": "Struts",
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "express": "Express",
    "react": "React", "angular": "Angular", "vue": "Vue", "jquery": "jQuery",
    "bootstrap": "Bootstrap", "sass": "SASS", "less": "LESS", "webpack": "Webpack",
    "babel": "Babel", "eslint": "ESLint", "prettier": "Prettier", "jest": "Jest",
    "cypress": "Cypress", "selenium": "Selenium", "puppeteer": "Puppeteer",
    "postman": "Postman", "swagger": "Swagger", "graphql": "GraphQL",
    "grpc": "gRPC", "soap": "SOAP", "rpc": "RPC", "mqtt": "MQTT",
    "kafka": "Kafka", "rabbitmq": "RabbitMQ", "activemq": "ActiveMQ",
    "spark": "Spark", "hadoop": "Hadoop", "hive": "Hive", "pig": "Pig",
    "tableau": "Tableau", "powerbi": "Power BI", "looker": "Looker",
    "snowflake": "Snowflake", "databricks": "Databricks", "airflow": "Airflow",
    "dbt": "dbt", "great expectations": "Great Expectations",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "keras": "Keras",
    "scikit-learn": "Scikit-Learn", "pandas": "Pandas", "numpy": "NumPy",
    "matplotlib": "Matplotlib", "seaborn": "Seaborn", "plotly": "Plotly",
    "jupyter": "Jupyter", "anaconda": "Anaconda", "conda": "Conda",
    "r": "R", "sas": "SAS", "spss": "SPSS", "stata": "Stata",
    "matlab": "MATLAB", "octave": "Octave", "mathematica": "Mathematica"
}

def normalize_job_title(title: str) -> str:
    """Normalize job title for lookup"""
    if not isinstance(title, str):
        return ""
    
    # Convert to lowercase and remove extra spaces
    normalized = re.sub(r'\s+', ' ', title.strip().lower())
    
    # Remove common prefixes/suffixes that don't affect core role
    prefixes_to_remove = [
        'junior ', 'senior ', 'lead ', 'principal ', 'staff ',
        'associate ', 'assistant ', 'intern ', 'trainee ', 'entry level ',
        'mid level ', 'experienced ', 'expert ', 'specialist ',
        'consultant ', 'freelance ', 'contract ', 'temporary ', 'part time ',
        'full time ', 'remote ', 'onsite ', 'hybrid '
    ]
    
    suffixes_to_remove = [
        ' i', ' ii', ' iii', ' iv', ' v',
        ' 1', ' 2', ' 3', ' 4', ' 5',
        ' intern', ' internship', ' trainee', ' graduate',
        ' contractor', ' consultant', ' freelancer',
        ' remote', ' onsite', ' hybrid'
    ]
    
    # Remove prefixes
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    # Remove suffixes
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    
    return normalized.strip()

def capitalize_word(word: str) -> str:
    """Capitalize word with special cases"""
    if not word:
        return word
    
    word_lower = word.lower()
    
    # Handle special cases
    if word_lower in SPECIAL_CAPS:
        return SPECIAL_CAPS[word_lower]
    
    # Handle compound words with slashes
    if "/" in word:
        return "/".join(capitalize_word(part) for part in word.split("/"))
    
    # Handle compound words with dots
    if "." in word and word_lower in SPECIAL_CAPS:
        return SPECIAL_CAPS[word_lower]
    
    # Handle hyphenated words
    if "-" in word:
        return "-".join(capitalize_word(part) for part in word.split("-"))
    
    # Regular capitalization
    return word.capitalize()

def prettify_role(role: str) -> str:
    """Convert job role to professional display format"""
    if not isinstance(role, str) or not role.strip():
        return ""
    
    original_role = role.strip()
    
    # First, try exact match in professional title map
    role_lower = original_role.lower()
    if role_lower in PROFESSIONAL_TITLE_MAP:
        return PROFESSIONAL_TITLE_MAP[role_lower]
    
    # Try normalized lookup (without seniority levels)
    normalized = normalize_job_title(original_role)
    if normalized in PROFESSIONAL_TITLE_MAP:
        # Preserve seniority level from original
        seniority_prefixes = [
            'Junior', 'Senior', 'Lead', 'Principal', 'Staff',
            'Associate', 'Assistant', 'Entry Level', 'Mid Level',
            'Experienced', 'Expert'
        ]
        
        for prefix in seniority_prefixes:
            if original_role.lower().startswith(prefix.lower() + ' '):
                return f"{prefix} {PROFESSIONAL_TITLE_MAP[normalized]}"
        
        return PROFESSIONAL_TITLE_MAP[normalized]
    
    # Fallback: Clean up the original title
    # Handle camelCase splitting
    camel_split = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', original_role)
    
    # Replace separators with spaces
    cleaned = re.sub(r'[-_/]+', ' ', camel_split)
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Split into words and capitalize each
    words = cleaned.split()
    capitalized_words = [capitalize_word(word) for word in words]
    
    result = ' '.join(capitalized_words)
    
    # Apply common corrections
    corrections = {
        'Fullstack': 'Full Stack',
        'Frontend': 'Frontend',
        'Backend': 'Backend',
        'Devops': 'DevOps',
        'Ai ': 'AI ',
        'Ml ': 'ML ',
        'Api ': 'API ',
        'Ui ': 'UI ',
        'Ux ': 'UX ',
        'Qa ': 'QA ',
        'Seo ': 'SEO ',
        'Cms ': 'CMS ',
        'Crm ': 'CRM ',
        'Erp ': 'ERP ',
        'Bi ': 'BI ',
        'It ': 'IT ',
        'Hr ': 'HR '
    }
    
    for old, new in corrections.items():
        result = result.replace(old, new)
    
    return result

# ------------------- CSS STYLES -------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary-color: #7c3aed;
            --primary-dark: #5b21b6;
            --secondary-color: #06b6d4;
            --accent-color: #f59e0b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --dark-bg: #0a0a0f;
            --darker-bg: #050507;
            --card-bg: #1a1b23;
            --card-hover: #22232b;
            --text-primary: #ffffff;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --border-color: #27272a;
            --border-glow: #3f3f46;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.6);
            --shadow-glow: 0 0 15px rgba(124, 58, 237, 0.25);
            --shadow-glow-cyan: 0 0 15px rgba(6, 182, 212, 0.25);
            --shadow-glow-success: 0 0 15px rgba(16, 185, 129, 0.25);
            --gradient-primary: linear-gradient(135deg, #7c3aed 0%, #a855f7 25%, #06b6d4 75%, #3b82f6 100%);
            --gradient-secondary: linear-gradient(135deg, #06b6d4 0%, #0891b2 25%, #7c3aed 75%, #a855f7 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 25%, #06b6d4 75%, #0891b2 100%);
            --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #d97706 25%, #ea580c 75%, #dc2626 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 25%, #b91c1c 75%, #991b1b 100%);
            --gradient-bg: linear-gradient(135deg, #0a0a0f 0%, #1a1b23 25%, #0f0f14 75%, #050507 100%);
            --neon-purple: #8b5cf6;
            --neon-cyan: #06b6d4;
            --neon-green: #10b981;
        }
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 500;
        }
        
        .stApp {
            background: var(--gradient-bg) !important;
        }
        
        .main {
            padding: 0.75rem 1.5rem 2rem 1.5rem;
            background: var(--gradient-bg);
            min-height: 100vh;
            color: var(--text-primary);
        }
        
        .main::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
            z-index: -1;
            animation: subtleFloat 20s ease-in-out infinite alternate;
        }
        
        @keyframes subtleFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
        }
        
        .main::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(124, 58, 237, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(124, 58, 237, 0.02) 1px, transparent 1px);
            background-size: 40px 40px;
            z-index: -1;
            opacity: 0.6;
        }
        
        .hero-section {
            text-align: center;
            padding: 2.5rem 0 3rem 0;
            background: var(--card-bg);
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-glow);
            position: relative;
            overflow: hidden;
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-primary);
            animation: subtlePulse 3s ease-in-out infinite alternate;
        }
        
        @keyframes subtlePulse {
            0% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        
        .main-header {
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.75rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
            letter-spacing: -0.01em;
            animation: slideInDown 0.8s ease-out;
        }
        
        .sub-header {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            animation: slideInUp 0.8s ease-out 0.1s both;
        }
        
        .feature-badges {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 2rem;
            animation: slideInUp 0.8s ease-out 0.2s both;
        }
        
        .feature-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--card-bg);
            color: var(--text-primary);
            padding: 0.6rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid var(--border-glow);
            transition: all 0.25s ease;
            box-shadow: var(--shadow-sm);
        }
        
        .feature-badge:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow);
            border-color: var(--neon-purple);
        }
        
        @keyframes slideInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .modern-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-glow);
            transition: all 0.25s ease;
            position: relative;
        }
        
        .modern-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--gradient-primary);
            opacity: 0.8;
        }
        
        .modern-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-glow);
            border-color: var(--neon-purple);
        }
        
        .upload-card {
            background: var(--card-bg);
            border: 2px dashed var(--neon-purple);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            text-align: center;
            margin: 2rem 0;
            transition: all 0.25s ease;
            position: relative;
            box-shadow: var(--shadow-md);
        }
        
        .upload-card:hover {
            border-color: var(--neon-cyan);
            background: var(--card-hover);
            box-shadow: var(--shadow-glow-cyan);
        }
        
        .upload-card::after {
            content: '📄';
            position: absolute;
            top: 1rem;
            right: 1.5rem;
            font-size: 1.5rem;
            opacity: 0.2;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-primary);
            margin: 2rem 0 1.5rem 0;
            position: relative;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .section-header::after {
            content: '';
            flex: 1;
            height: 2px;
            background: var(--gradient-primary);
            margin-left: 1rem;
            border-radius: 1px;
            opacity: 0.8;
        }
        
        .metric-card {
            background: var(--gradient-primary);
            padding: 1.5rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            box-shadow: var(--shadow-md);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .metric-card:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: var(--shadow-glow);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: -25%;
            right: -25%;
            width: 150%;
            height: 150%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: subtleShimmer 6s ease-in-out infinite;
        }
        
        @keyframes subtleShimmer {
            0%, 100% { transform: rotate(0deg); opacity: 0.2; }
            50% { transform: rotate(180deg); opacity: 0.4; }
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 2;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
            z-index: 2;
        }
        
        .skill-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--gradient-success);
            color: white;
            padding: 0.5rem 0.8rem;
            margin: 0.3rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 700;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
            border: 1px solid transparent;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .skill-badge:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: var(--shadow-glow-success);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .skill-badge::before {
            content: '✓';
            font-size: 0.7rem;
            font-weight: 800;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            width: 1rem;
            height: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .missing-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--gradient-danger);
            color: white;
            padding: 0.5rem 0.8rem;
            margin: 0.3rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 700;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
            border: 1px solid transparent;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .missing-badge:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .missing-badge::before {
            content: '!';
            font-size: 0.7rem;
            font-weight: 800;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            width: 1rem;
            height: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .info-panel {
            background: linear-gradient(135deg, var(--card-bg) 0%, var(--card-hover) 100%);
            border: 1px solid var(--border-glow);
            border-left: 3px solid var(--neon-purple);
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-sm);
            position: relative;
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .success-panel {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, var(--card-bg) 100%);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-left: 3px solid var(--neon-green);
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-sm);
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .warning-panel {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, var(--card-bg) 100%);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-left: 3px solid var(--accent-color);
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-sm);
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .stButton button {
            background: var(--gradient-primary) !important;
            border: 1px solid var(--neon-purple) !important;
            border-radius: 20px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            transition: all 0.25s ease !important;
            box-shadow: var(--shadow-sm) !important;
            position: relative !important;
            overflow: hidden !important;
            color: white !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: var(--shadow-glow) !important;
            border-color: var(--neon-cyan) !important;
        }
        
        .stButton button:active {
            transform: translateY(-1px) !important;
        }
        
        .stFileUploader {
            border: none !important;
        }
        
        .stFileUploader > div > div {
            border: 2px dashed var(--neon-purple) !important;
            border-radius: 16px !important;
            background: var(--card-bg) !important;
            transition: all 0.25s ease !important;
            color: var(--text-primary) !important;
            padding: 1.5rem !important;
        }
        
        .stFileUploader > div > div:hover {
            border-color: var(--neon-cyan) !important;
            background: var(--card-hover) !important;
            box-shadow: var(--shadow-glow-cyan) !important;
        }
        
        .css-1d391kg {
            background: var(--gradient-primary) !important;
        }
        
        .sidebar .sidebar-content {
            background: var(--gradient-primary) !important;
            color: white !important;
        }
        
        .stProgress > div > div > div {
            background: var(--gradient-primary) !important;
            box-shadow: 0 0 8px var(--neon-purple) !important;
        }
        
        .stProgress > div > div {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-glow) !important;
        }
        
        .stSelectbox > div > div {
            border-radius: 12px !important;
            border: 1px solid var(--border-glow) !important;
            box-shadow: var(--shadow-sm) !important;
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: var(--neon-purple) !important;
            box-shadow: var(--shadow-glow) !important;
        }
        
        .stTextArea > div > div {
            border-radius: 12px !important;
            border: 1px solid var(--border-glow) !important;
            box-shadow: var(--shadow-sm) !important;
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }
        
        .stTextArea > div > div:focus-within {
            border-color: var(--neon-purple) !important;
            box-shadow: var(--shadow-glow) !important;
        }
        
        .streamlit-expanderHeader {
            background: var(--card-bg) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-glow) !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.3px !important;
            font-size: 0.9rem !important;
        }
        
        .streamlit-expanderContent {
            background: var(--card-hover) !important;
            border-radius: 0 0 12px 12px !important;
            border: 1px solid var(--border-glow) !important;
            border-top: none !important;
            color: var(--text-primary) !important;
        }
        
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .mini-metric-card {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border-glow);
            box-shadow: var(--shadow-sm);
            transition: all 0.25s ease;
        }
        
        .mini-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            border-color: var(--neon-purple);
        }
        
        .mini-metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .mini-metric-label {
            font-size: 0.75rem;
            opacity: 0.8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--darker-bg);
            border-radius: 8px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gradient-primary);
            border-radius: 8px;
            border: 1px solid var(--darker-bg);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--gradient-secondary);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }
        
        p, span, div {
            color: var(--text-primary) !important;
        }
        
        .stMarkdown {
            color: var(--text-primary) !important;
        }
        
        .sidebar-feature {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(124, 58, 237, 0.15);
            border-radius: 12px;
            border: 1px solid rgba(124, 58, 237, 0.25);
        }
        
        .sidebar-step {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.5rem 0;
            padding: 0.75rem;
            background: rgba(124, 58, 237, 0.2);
            border-radius: 10px;
            border: 1px solid rgba(124, 58, 237, 0.3);
        }
        
        .step-number {
            background: var(--gradient-primary);
            color: white;
            border-radius: 50%;
            width: 1.75rem;
            height: 1.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.8rem;
            box-shadow: 0 0 10px rgba(124, 58, 237, 0.4);
        }
        
        .step-text {
            color: white;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .main {
                padding: 0.5rem;
            }
            
            .modern-card {
                padding: 1.25rem;
            }
            
            .section-header {
                font-size: 1.25rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
            
            .feature-badges {
                flex-direction: column;
                align-items: center;
            }
        }
        
        .file-detail-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: var(--card-hover);
            border-radius: 8px;
            border-left: 2px solid var(--primary-color);
        }
        
        .file-detail-key {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.85rem;
        }
        
        .file-detail-value {
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
        
        .code-preview {
            background: #000;
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.75rem;
            line-height: 1.4;
            color: var(--text-primary);
            max-height: 150px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- HERO SECTION -------------------
st.markdown("""
    <div class="hero-section">
        <h1 class="main-header">SkillMatch AI</h1>
        <p class="sub-header">AI-Powered Resume Analysis & Career Optimization</p>
        <div class="feature-badges">
            <div class="feature-badge">
                <span>🎯</span> Precision Matching
            </div>
            <div class="feature-badge">
                <span>🚀</span> AI Insights
            </div>
            <div class="feature-badge">
                <span>⚡</span> Instant Analysis
            </div>
            <div class="feature-badge">
                <span>🔥</span> Career Boost
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("### Advanced Features")
    st.markdown("""
    <div class="info-panel">
        <strong>SkillMatch AI</strong> delivers intelligent career analysis:
        <br><br>
        <div class="sidebar-feature">
            <div>🎯 <strong>Smart Analysis:</strong> Advanced matching algorithms</div>
            <div>⚡ <strong>Quick Insights:</strong> Real-time gap identification</div>
            <div>🚀 <strong>AI Recommendations:</strong> Personalized career advice</div>
            <div>💎 <strong>Detailed Reports:</strong> Professional analysis documents</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Start")
    steps = [
        "📤 Upload Resume",
        "🎯 Select Role", 
        "⚡ Run Analysis",
        "📊 View Results",
        "🔥 Download Report"
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="sidebar-step">
            <div class="step-number">{i}</div>
            <span class="step-text">{step}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Pro Tips")
    st.markdown("""
    <div class="sidebar-feature">
        <div style="margin-bottom: 0.5rem; font-weight: 700;">💼 Keep Resume Updated</div>
        <div style="margin-bottom: 0.5rem; font-weight: 700;">🔍 Use Power Keywords</div>
        <div style="margin-bottom: 0.5rem; font-weight: 700;">📈 Track Progress</div>
        <div style="font-weight: 700;">🎯 Target Smart Roles</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Support")
    st.markdown("""
    <div style="text-align: center; color: white;">
        <div style="margin: 0.5rem 0; font-weight: 700; font-size: 0.8rem;">📚 Guide</div>
        <div style="margin: 0.5rem 0; font-weight: 700; font-size: 0.8rem;">💬 Help</div>
        <div style="margin: 0.5rem 0; font-weight: 700; font-size: 0.8rem;">📧 Contact</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------- DATA LOADING -------------------
@st.cache_data(ttl=3600)
def load_data():
    """Load job descriptions with comprehensive error handling"""
    try:
        if not os.path.exists("job_descriptions.csv"):
            st.error("❌ job_descriptions.csv file not found")
            st.info("🔍 Please ensure 'job_descriptions.csv' exists in the same directory as app.py")
            st.stop()
        
        df = load_job_descriptions("job_descriptions.csv")
        
        # Validate required columns
        required_columns = ['job_title', 'skills']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns in CSV: {', '.join(missing_columns)}")
            st.info("🔍 Required columns: job_title, skills, job_description (optional)")
            st.stop()
        
        # Clean and validate data
        df = df.dropna(subset=['job_title', 'skills'])  # Remove rows with missing critical data
        df['job_title'] = df['job_title'].astype(str).str.strip()  # Clean job titles
        df['skills'] = df['skills'].astype(str).str.strip()  # Clean skills
        
        if len(df) == 0:
            st.error("❌ No valid job data found in CSV file")
            st.stop()
        
        return df
        
    except pd.errors.EmptyDataError:
        st.error("❌ The CSV file is empty")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        st.info("🔍 Please check your CSV file format")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading job data: {str(e)}")
        st.info("🔍 Make sure 'job_descriptions.csv' exists with columns: job_title, skills, job_description")
        st.stop()

# Load data with comprehensive error handling
try:
    with st.spinner("🔄 Loading job descriptions..."):
        job_data = load_data()
    
    # Apply professional title formatting
    job_data["display_title"] = job_data["job_title"].apply(prettify_role)
    
    # Remove any empty display titles
    job_data = job_data[job_data["display_title"] != ""]
    
    # Sort by display title for better UX
    job_data = job_data.sort_values("display_title").reset_index(drop=True)
    
    st.markdown(f"""
    <div class="success-panel">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="font-size: 1.25rem;">✅</div>
            <div>
                <strong style="font-size: 1rem; font-weight: 800;">System Ready!</strong><br>
                <span style="color: #10b981; font-weight: 600; font-size: 0.9rem;">Loaded {len(job_data)} job profiles with professional titles</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"❌ Failed to initialize application: {str(e)}")
    st.stop()

# ------------------- MAIN APPLICATION -------------------
st.markdown('<div class="section-header">🔍 Upload Resume</div>', unsafe_allow_html=True)

st.markdown("""
<div class="modern-card">
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">📄</div>
        <h3 style="color: var(--text-primary); margin-bottom: 0.5rem; font-weight: 800; font-size: 1.25rem;">Upload Your Resume</h3>
        <p style="color: var(--text-secondary); font-weight: 600; font-size: 0.9rem;">Supported: PDF • DOCX • TXT • Max: 200MB</p>
    </div>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Choose your resume file",
    type=["pdf", "docx", "txt"],
    help="Drag and drop your file here or click to browse",
    label_visibility="collapsed"
)

if uploaded_file:
    st.markdown("""
    <div class="modern-card">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
            <div style="font-size: 1.5rem;">🔎</div>
            <div style="flex: 1;">
                <h4 style="margin: 0; color: var(--text-primary); font-size: 1rem;">File Uploaded Successfully</h4>
                <p style="margin: 0.25rem 0 0 0; color: var(--text-secondary); font-size: 0.8rem;">Ready for analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File details
    with st.expander("📄 File Details", expanded=False):
        file_details = {
            "📎 Filename": uploaded_file.name,
            "📊 File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "🗂️ File Type": uploaded_file.type or "Unknown"
        }
        
        for key, value in file_details.items():
            st.markdown(f"""
            <div class="file-detail-item">
                <span class="file-detail-key">{key}</span>
                <span class="file-detail-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Extract resume text with error handling
    try:
        with st.spinner("📖 Processing resume..."):
            resume_text = extract_text_from_resume(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        resume_text = None
    
    if resume_text and resume_text.strip():
        st.markdown("""
        <div class="success-panel">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="font-size: 1.25rem;">✅</div>
                <div>
                    <strong style="font-size: 0.95rem;">Resume Processed!</strong><br>
                    <span style="color: #059669; font-size: 0.8rem;">Text extracted successfully</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Text preview
        with st.expander("👀 Resume Preview", expanded=False):
            preview_text = resume_text[:400] + "..." if len(resume_text) > 400 else resume_text
            st.markdown(f"""
            <div style="background: var(--card-bg); border-radius: 12px; padding: 1rem; border: 1px solid var(--border-color);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1rem;">📄</span>
                    <strong style="color: var(--text-primary); font-size: 0.9rem;">Extracted Text</strong>
                    <span style="background: var(--primary-color); color: white; padding: 0.2rem 0.4rem; border-radius: 8px; font-size: 0.7rem; font-weight: 600;">{len(resume_text)} chars</span>
                </div>
                <div class="code-preview">
                    {preview_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Job role selection section
        st.markdown('<div class="section-header">🎯 Select Target Role</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="modern-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.75rem;">🎯</div>
                <h3 style="color: var(--text-primary); margin-bottom: 0.5rem; font-size: 1.1rem;">Choose Your Target Role</h3>
                <p style="color: var(--text-secondary); font-size: 0.85rem;">Select the job position that matches your goals</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            try:
                job_roles = sorted(job_data["display_title"].unique())
                job_role_display = st.selectbox(
                    "Available Job Roles:",
                    options=job_roles,
                    help="Select from our database of professional job roles",
                    key="job_role_selector"
                )
                
                if job_role_display:
                    role_count = len(job_data[job_data["display_title"] == job_role_display])
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%); border-radius: 8px; padding: 0.75rem; margin: 0.75rem 0; border: 1px solid rgba(99, 102, 241, 0.2);">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1rem;">📊</span>
                            <span style="font-weight: 600; color: var(--text-primary); font-size: 0.85rem;">Role Stats:</span>
                            <span style="background: var(--primary-color); color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600;">{role_count} variations</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error loading job roles: {str(e)}")
                st.stop()
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button(
                "🚀 Analyze", 
                use_container_width=True,
                help="Start AI analysis"
            )
        
        # Analysis logic
        if analyze_button:
            if not job_role_display or job_role_display not in job_data["display_title"].values:
                st.markdown("""
                <div class="warning-panel">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="font-size: 1.25rem;">⚠️</div>
                        <div>
                            <strong style="font-size: 0.95rem;">Invalid Selection</strong><br>
                            <span style="color: #d97706; font-size: 0.8rem;">Please choose a valid job role.</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Progress display
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("""
                    <div class="modern-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.75rem;">🤖</div>
                            <h3 style="color: var(--text-primary); margin-bottom: 0.5rem; font-size: 1.1rem;">AI Analysis in Progress</h3>
                            <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.85rem;">Analyzing your resume...</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    analysis_steps = [
                        "🔍 Extracting skills...",
                        "📊 Analyzing requirements...", 
                        "🎯 Calculating scores...",
                        "📈 Generating insights...",
                        "✅ Complete!"
                    ]
                    
                    for i, step in enumerate(analysis_steps):
                        status_text.markdown(f"""
                        <div style="text-align: center; color: var(--text-primary); font-weight: 500; padding: 0.25rem; font-size: 0.85rem;">
                            {step}
                        </div>
                        """, unsafe_allow_html=True)
                        progress_bar.progress((i + 1) / len(analysis_steps))
                        time.sleep(0.2)
                    
                    progress_bar.empty()
                    status_text.empty()
                
                # Perform analysis with comprehensive error handling
                try:
                    with st.spinner("🔍 Analyzing match..."):
                        # Get selected job data
                        selected_job = job_data[job_data["display_title"] == job_role_display].iloc[0]
                        
                        # Initialize variables with defaults
                        matched_skills = []
                        missing_skills = []
                        overall_score = 0
                        skill_match = 0
                        context_match = 0
                        recommendations = []
                        
                        try:
                            # Basic matching
                            basic_match_score = get_match_score(resume_text, selected_job["skills"])
                            
                            # Extract skills
                            resume_skills = extract_skills_advanced(resume_text)
                            job_skills_text = selected_job["skills"]
                            job_skills = extract_skills_advanced(job_skills_text)
                            
                            # Calculate matches
                            for skill in job_skills:
                                skill_lower = skill.lower()
                                if any(skill_lower in resume_skill.lower() or resume_skill.lower() in skill_lower 
                                      for resume_skill in resume_skills):
                                    matched_skills.append(skill)
                                else:
                                    missing_skills.append(skill)
                            
                            # Calculate skill match percentage
                            if len(job_skills) > 0:
                                skill_match_percentage = (len(matched_skills) / len(job_skills)) * 100
                            else:
                                skill_match_percentage = 0
                            
                            # Try advanced scoring
                            try:
                                match_result = calculate_comprehensive_score(
                                    resume_text, 
                                    selected_job["skills"], 
                                    selected_job.get("job_description", "")
                                )
                                overall_score = match_result["overall_score"]
                                skill_match = match_result["skill_match_score"]
                                context_match = match_result["context_match_score"]
                                matched_skills = match_result.get("matched_skills", matched_skills)
                                missing_skills = match_result.get("missing_skills", missing_skills)
                            
                            except Exception as e:
                                logger.warning(f"Advanced scoring failed, using basic: {str(e)}")
                                overall_score = skill_match_percentage
                                skill_match = skill_match_percentage
                                context_match = basic_match_score
                            
                            # Generate recommendations
                            try:
                                recommendations = generate_skill_recommendations(
                                    missing_skills[:10], 
                                    selected_job["job_title"]
                                )
                            except Exception as e:
                                logger.warning(f"Recommendations failed: {str(e)}")
                                recommendations = [
                                    f"Consider learning {skill}" for skill in missing_skills[:5]
                                ]
                        
                        except Exception as e:
                            st.error(f"❌ Analysis error: {str(e)}")
                            logger.error(f"Analysis error: {str(e)}")
                            st.stop()
                        
                        st.markdown("""
                        <div class="success-panel">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div style="font-size: 1.25rem;">🎉</div>
                                <div>
                                    <strong style="font-size: 0.95rem;">Analysis Complete!</strong><br>
                                    <span style="color: #059669; font-size: 0.8rem;">Resume analyzed successfully</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Results Display
                        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
                        
                        # Main Score Display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            score_color = "🟢" if overall_score >= 70 else "🟡" if overall_score >= 50 else "🔴"
                            st.markdown(f"""
                                <div class="metric-card" style="background: var(--gradient-primary);">
                                    <div class="metric-value">{score_color} {overall_score:.1f}%</div>
                                    <div class="metric-label">Overall Match</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card" style="background: var(--gradient-success);">
                                    <div class="metric-value">🎯 {skill_match:.1f}%</div>
                                    <div class="metric-label">Skill Match</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card" style="background: var(--gradient-secondary);">
                                    <div class="metric-value">🔍 {context_match:.1f}%</div>
                                    <div class="metric-label">Context Match</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Mini metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="mini-metric-card">
                                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">✅</div>
                                <div class="mini-metric-value" style="color: var(--success-color);">{len(matched_skills)}</div>
                                <div class="mini-metric-label" style="color: var(--text-secondary);">Skills Found</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="mini-metric-card">
                                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">❌</div>
                                <div class="mini-metric-value" style="color: var(--error-color);">{len(missing_skills)}</div>
                                <div class="mini-metric-label" style="color: var(--text-secondary);">Missing</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            total_skills = len(matched_skills) + len(missing_skills)
                            st.markdown(f"""
                            <div class="mini-metric-card">
                                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">📋</div>
                                <div class="mini-metric-value" style="color: var(--primary-color);">{total_skills}</div>
                                <div class="mini-metric-label" style="color: var(--text-secondary);">Total Skills</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            grade = "A+" if overall_score >= 90 else "A" if overall_score >= 80 else "B" if overall_score >= 70 else "C" if overall_score >= 60 else "D"
                            st.markdown(f"""
                            <div class="mini-metric-card">
                                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🏆</div>
                                <div class="mini-metric-value" style="color: var(--warning-color);">{grade}</div>
                                <div class="mini-metric-label" style="color: var(--text-secondary);">Grade</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Progress breakdown
                        st.markdown('<div class="section-header">📈 Breakdown</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="modern-card">
                                <h4 style="color: var(--text-primary); margin-bottom: 0.75rem; font-size: 1rem;">Skills Alignment</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(max(0, min(100, skill_match)) / 100)
                            st.write(f"**{len(matched_skills)}** matched out of **{total_skills}** analyzed")
                        
                        with col2:
                            st.markdown("""
                            <div class="modern-card">
                                <h4 style="color: var(--text-primary); margin-bottom: 0.75rem; font-size: 1rem;">Experience Context</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(max(0, min(100, context_match)) / 100)
                            st.write("Based on resume content analysis")
                        
                        # Skills Analysis
                        st.markdown('<div class="section-header">🎯 Skills Analysis</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="modern-card">
                                <h4 style="color: var(--success-color); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; font-size: 1rem;">
                                    <span>✅</span> Matched Skills
                                </h4>
                            """, unsafe_allow_html=True)
                            if matched_skills:
                                skills_html = "".join([f'<span class="skill-badge">{skill}</span>' for skill in matched_skills[:12]])
                                st.markdown(f'<div style="margin: 0.75rem 0;">{skills_html}</div>', unsafe_allow_html=True)
                                
                                if len(matched_skills) > 12:
                                    with st.expander(f"Show all {len(matched_skills)} matched skills"):
                                        remaining_skills = "".join([f'<span class="skill-badge">{skill}</span>' for skill in matched_skills[12:]])
                                        st.markdown(remaining_skills, unsafe_allow_html=True)
                            else:
                                st.warning("No skills matched. Consider updating your resume.")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="modern-card">
                                <h4 style="color: var(--error-color); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; font-size: 1rem;">
                                    <span>⚠️</span> Skills to Develop
                                </h4>
                            """, unsafe_allow_html=True)
                            if missing_skills:
                                missing_html = "".join([f'<span class="missing-badge">{skill}</span>' for skill in missing_skills[:12]])
                                st.markdown(f'<div style="margin: 0.75rem 0;">{missing_html}</div>', unsafe_allow_html=True)
                                
                                if len(missing_skills) > 12:
                                    with st.expander(f"Show all {len(missing_skills)} missing skills"):
                                        remaining_missing = "".join([f'<span class="missing-badge">{skill}</span>' for skill in missing_skills[12:]])
                                        st.markdown(remaining_missing, unsafe_allow_html=True)
                            else:
                                st.success("🎉 All required skills found!")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Recommendations
                        if recommendations and len(recommendations) > 0:
                            st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
                            
                            for i, rec in enumerate(recommendations[:3], 1):
                                priority = "High" if i <= 2 else "Medium"
                                priority_color = "var(--error-color)" if priority == "High" else "var(--warning-color)"
                                icon = "🎯" if i == 1 else "📚" if i == 2 else "💡"
                                
                                st.markdown(f"""
                                <div class="modern-card">
                                    <div style="display: flex; align-items: start; gap: 0.75rem;">
                                        <div style="font-size: 1.25rem;">{icon}</div>
                                        <div style="flex: 1;">
                                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                                                <h4 style="margin: 0; color: var(--text-primary); font-size: 0.95rem;">Recommendation {i}</h4>
                                                <span style="background: {priority_color}; color: white; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.65rem; font-weight: 600;">{priority}</span>
                                            </div>
                                            <p style="margin: 0; color: var(--text-secondary); line-height: 1.4; font-size: 0.85rem;">{rec}</p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Visualization Section
                        st.markdown('<div class="section-header">📊 Visual Analysis</div>', unsafe_allow_html=True)
                        
                        fig_col1, fig_col2 = st.columns(2)
                        
                        with fig_col1:
                            if matched_skills or missing_skills:
                                try:
                                    labels = ['Matched Skills', 'Missing Skills']
                                    values = [len(matched_skills), len(missing_skills)]
                                    colors = ['#10b981', '#ef4444']
                                    
                                    fig = go.Figure(data=[go.Pie(
                                        labels=labels, 
                                        values=values,
                                        hole=.3,
                                        marker_colors=colors
                                    )])
                                    fig.update_layout(
                                        title="Skills Distribution",
                                        font=dict(size=12),
                                        height=300,
                                        margin=dict(t=40, b=20, l=20, r=20),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating pie chart: {str(e)}")
                            else:
                                st.info("No skills data available for visualization")
                        
                        with fig_col2:
                            try:
                                categories = ['Skill Match', 'Context Match', 'Overall Score']
                                scores = [
                                    max(0, min(100, skill_match)),
                                    max(0, min(100, context_match)), 
                                    max(0, min(100, overall_score))
                                ]
                                
                                fig = go.Figure([go.Bar(
                                    x=categories,
                                    y=scores,
                                    marker_color=['#6366f1', '#8b5cf6', '#10b981']
                                )])
                                fig.update_layout(
                                    title="Score Breakdown",
                                    yaxis=dict(range=[0, 100]),
                                    font=dict(size=12),
                                    height=300,
                                    margin=dict(t=40, b=20, l=20, r=20),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating bar chart: {str(e)}")
                        
                        # Job Requirements
                        with st.expander("📋 Job Requirements", expanded=False):
                            st.markdown("**Job Title:** " + job_role_display)
                            st.markdown("**Required Skills:**")
                            st.write(selected_job["skills"])
                            if "job_description" in selected_job.index and pd.notna(selected_job["job_description"]):
                                st.markdown("**Job Description:**")
                                st.write(selected_job["job_description"])
                        
                        # Export Section
                        st.markdown('<div class="section-header">🔥 Export</div>', unsafe_allow_html=True)
                        
                        try:
                            report_data = {
                                "Analysis Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Target Job Role": job_role_display,
                                "Overall Match Score": f"{overall_score:.1f}%",
                                "Skill Match Score": f"{skill_match:.1f}%",
                                "Context Match Score": f"{context_match:.1f}%",
                                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                                "Recommendations": " | ".join(recommendations[:5]) if recommendations else "None"
                            }
                            
                            report_df = pd.DataFrame([report_data])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                csv_buffer = BytesIO()
                                report_df.to_csv(csv_buffer, index=False)
                                st.download_button(
                                    label="📄 CSV Report",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"skillmatch_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                text_report = f"""SKILLMATCH AI ANALYSIS REPORT
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
{chr(10).join([f"- {rec}" for rec in recommendations[:5]]) if recommendations else "None available"}

Generated by SkillMatch AI
"""
                                st.download_button(
                                    label="📝 Text Report",
                                    data=text_report,
                                    file_name=f"skillmatch_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Error preparing export data: {str(e)}")
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-panel">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="font-size: 1.25rem;">⚠️</div>
                            <div>
                                <strong style="font-size: 0.95rem;">Analysis Error</strong><br>
                                <span style="color: #d97706; font-size: 0.8rem;">Error: {str(e)}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    logger.error(f"Analysis error: {str(e)}")
    
    else:
        st.markdown("""
        <div class="warning-panel">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="font-size: 1.25rem;">⚠️</div>
                <div>
                    <strong style="font-size: 0.95rem;">Processing Error</strong><br>
                    <span style="color: #d97706; font-size: 0.8rem;">Unable to extract text. Try a different file format or check if the file is corrupted.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Call-to-action when no file uploaded
    st.markdown("""
    <div class="upload-card">
        <div style="font-size: 3rem; margin-bottom: 0.75rem; opacity: 0.5;">📊</div>
        <h3 style="color: var(--text-primary); margin-bottom: 0.5rem; font-size: 1.1rem;">Ready to Start?</h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;">Upload your resume to begin AI analysis</p>
        <div style="display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap;">
            <div style="background: var(--card-hover); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border-color); color: var(--text-secondary); font-size: 0.75rem;">
                <span style="margin-right: 0.5rem;">📄</span> PDF
            </div>
            <div style="background: var(--card-hover); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border-color); color: var(--text-secondary); font-size: 0.75rem;">
                <span style="margin-right: 0.5rem;">📝</span> DOCX
            </div>
            <div style="background: var(--card-hover); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border-color); color: var(--text-secondary); font-size: 0.75rem;">
                <span style="margin-right: 0.5rem;">📋</span> TXT
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0; color: var(--text-secondary);">
    <p style="margin-bottom: 0.25rem; font-size: 0.8rem;">Made with ❤️ using Streamlit • Powered by AI</p>
    <p style="margin: 0; font-size: 0.75rem;">SkillMatch AI © 2024. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
