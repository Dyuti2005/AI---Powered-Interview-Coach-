import streamlit as st
import librosa
import numpy as np
import pandas as pd
import whisper
import io
import os
import time
import pdfplumber
import cv2
import random
import mediapipe as mp
import base64
from gtts import gTTS
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from sentence_transformers import SentenceTransformer, util
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ==========================================
# 1. Page Configuration & Professional Styling
# ==========================================
st.set_page_config(page_title="AI Interview Coach", layout="wide")

# CSS to force visibility and remove clutter
st.markdown("""
    <style>
    .stApp { background: #f4f4f4; }
    [data-testid="stSidebar"] { background-color: #4a080f !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .header-box { text-align: center; padding: 25px; background-color: #ffffff; border-bottom: 5px solid #6b0f1a; margin-bottom: 20px; }
    .main-title { font-size: 38px; font-weight: 900; color: #6b0f1a; margin: 0; }
    .step-card { background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-top: 5px solid #6b0f1a; margin-bottom: 20px;}
    h1, h2, h3, p, span, label, .stMarkdown, div { color: #000000 !important; }
    .stButton>button { background-color: #6b0f1a; color: white !important; font-weight: bold; width: 100%; height: 50px; }
    /* Hide the uploaders during interview */
    .hidden { display: none; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FULL QUESTION BANK
# ==========================================
QUESTION_BANK = {
    "Software Engineer": [
        "Explain the four pillars of Object-Oriented Programming and how they improve code maintainability.",
        "What is the difference between an Interface and an Abstract Class? When would you use one over the other?",
        "How do Hash Maps handle collisions, and what is the impact of a poor hash function on time complexity?",
        "Explain the difference between a process and a thread. How do they share memory?",
        "What is Big O notation? How would you analyze the time complexity of a recursive Fibonacci function?",
        "How do you identify and prevent memory leaks in a language like Java or Python?",
        "Compare SQL and NoSQL databases. In what scenario would you choose a NoSQL database over a relational one?",
        "What is Dependency Injection, and how does it promote loose coupling in software design?",
        "Describe the differences between Monolithic and Microservices architecture. What are the deployment trade-offs?",
        "Explain the CAP theorem. How does it influence your choice of a distributed database?"
    ],
    "AI Engineer": [
        "What is the difference between Batch, Mini-batch, and Stochastic Gradient Descent?",
        "Explain the concept of Dropout in Neural Networks. How does it prevent overfitting?",
        "How do you handle exploding or vanishing gradients during the training of deep networks?",
        "What is the difference between Generative and Discriminative models? Give an example of each.",
        "Describe the architecture of a Transformer. What is the role of the Self-Attention mechanism?",
        "How does a CNN perform feature extraction through convolutional layers and pooling?",
        "What is the role of an Activation Function? Why is ReLU often preferred over Sigmoid in hidden layers?",
        "Explain the bias-variance tradeoff. How does regularizing a model affect these two components?",
        "What is the difference between L1 and L2 Regularization? Which one leads to sparse feature selection?",
        "How do you evaluate a classification model's performance when dealing with a highly imbalanced dataset?"
    ],
    "Cloud Engineer": [
        "Explain the differences between IaaS, PaaS, and SaaS service models with real-world examples.",
        "How does a Load Balancer distribute traffic across multiple availability zones to ensure high availability?",
        "What is Auto-scaling? Explain the difference between reactive and scheduled scaling policies.",
        "Explain the Cloud Shared Responsibility Model. Who is responsible for data encryption in the cloud?",
        "What is Infrastructure as Code (IaC)? How do tools like Terraform or CloudFormation manage state?",
        "Describe the difference between horizontal and vertical scaling. Which one is more suitable for cloud-native apps?",
        "What are the benefits of Docker Containers over traditional Virtual Machines in terms of resource utilization?",
        "How do you implement a Disaster Recovery strategy? What is the difference between Pilot Light and Warm Standby?",
        "Explain the role of a Content Delivery Network (CDN) in reducing latency for global users.",
        "What is the difference between a Public, Private, and Hybrid Cloud? When should a company use a Hybrid approach?"
    ],
    "Data Analyst": [
        "Explain the difference between a JOIN and a UNION in SQL. When would you use a Cross Join?",
        "How do you handle missing or noisy data during the data cleaning process? What is mean imputation?",
        "What is the difference between Correlation and Causation? Provide an example of a spurious correlation.",
        "Why is Data Normalization important in relational databases? Explain 1NF, 2NF, and 3NF briefly.",
        "Describe the difference between descriptive, predictive, and prescriptive analytics.",
        "What is an outlier? How do you use a Box Plot or Z-score to decide whether to remove an outlier?",
        "How do you translate technical data findings into actionable insights for non-technical stakeholders?",
        "What is the purpose of a P-value and a Significance Level in statistical hypothesis testing?",
        "Explain the logic behind an A/B test. How do you determine the sample size needed for a valid result?",
        "Describe the stages of a standard ETL (Extract, Transform, Load) pipeline in a data warehousing environment."
    ],
    "Backend Intern": [
        "Describe the lifecycle of an HTTP request from the moment a user enters a URL to the server response.",
        "What is a RESTful API? Name the primary constraints like statelessness and uniform interface.",
        "What is the difference between GET, POST, PUT, and DELETE methods in terms of idempotency?",
        "What is a Database Schema? Why is it essential to plan the schema before writing backend code?",
        "Explain how Cookies and Sessions maintain user state in a stateless HTTP protocol.",
        "What is Middleware? Provide an example of how it is used for authentication or logging.",
        "Explain Database Indexing. How does it speed up read operations and what is the cost to write operations?",
        "Why is JSON the preferred format for data exchange in modern web APIs compared to XML?",
        "Describe the difference between Authentication and Authorization with a real-world example.",
        "How do you protect a backend API from common security threats like SQL Injection and XSS attacks?"
    ]
}

# ==========================================
# 3. Models & Video Logic
# ==========================================
@st.cache_resource
def load_models():
    return whisper.load_model("base"), SentenceTransformer('all-MiniLM-L6-v2'), mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

whisper_model, match_model, face_mesh_instance = load_models()

if 'look_away_count' not in st.session_state: st.session_state.look_away_count = 0

class VideoProctor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh_instance.process(rgb)
        status, color = "Focused", (0, 255, 0)
        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                nose = face.landmark[1]
                eye_mid = (face.landmark[33].x + face.landmark[263].x) / 2
                if nose.x < eye_mid - 0.045:
                    status, color = "WARNING UR LOOKING LEFT", (0, 0, 255)
                    st.session_state.look_away_count += 1
                elif nose.x > eye_mid + 0.045:
                    status, color = "WARNING UR LOOKING RIGHT", (0, 0, 255)
                    st.session_state.look_away_count += 1
        cv2.putText(img, status, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        return img

def autoplay_audio(text, key):
    tts = gTTS(text=text, lang='en')
    with io.BytesIO() as f:
        tts.write_to_fp(f)
        f.seek(0)
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<audio autoplay="true" key="{key}"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# ==========================================
# 4. App Session State
# ==========================================
if 'stage' not in st.session_state: st.session_state.stage = 'profile'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'start_time' not in st.session_state: st.session_state.start_time = None

with st.sidebar:
    st.header("CONTROLS")
    u_name = st.text_input("Name")
    role = st.selectbox("Role", list(QUESTION_BANK.keys()))
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.markdown('<div class="header-box"><h1 class="main-title">AI INTERVIEW COACH</h1></div>', unsafe_allow_html=True)

# ==========================================
# 5. Application Flow
# ==========================================

# STAGE 1: Setup - ONLY displayed when stage is 'profile'
if st.session_state.stage == 'profile':
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    resume = st.file_uploader("Upload Resume", type=["pdf"])
    jd = st.text_area("Paste Job Description")
    if st.button("Start Interview"):
        if resume and jd and u_name:
            with pdfplumber.open(resume) as pdf:
                txt = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            score = float(util.cos_sim(match_model.encode(txt), match_model.encode(jd))) * 100
            if score >= 40:
                st.session_state.u_name = u_name
                st.session_state.questions = ["Introduce yourself and your background."] + random.sample(QUESTION_BANK[role], 5)
                st.session_state.stage = 'interview'
                st.session_state.start_time = time.time()
                st.rerun()
            else: st.error("Match score too low.")
    st.markdown('</div>', unsafe_allow_html=True)

# STAGE 2: Automated Interview - Uploads are completely absent here
elif st.session_state.stage == 'interview':
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        webrtc_streamer(key="proctor", video_transformer_factory=VideoProctor, mode=WebRtcMode.SENDRECV)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        idx = st.session_state.q_idx
        qs = st.session_state.questions
        
        # UNIQUE KEY for every question forces audio to play
        autoplay_audio(qs[idx], f"audio_play_{idx}")
        
        st.info(qs[idx])
        
        # Audio input for recording
        audio_in = st.audio_input("Microphone", key=f"voice_{idx}")
        
        # Automation Logic: 20-second timer
        elapsed = time.time() - st.session_state.start_time
        
        # Transition after 20 seconds OR manual response
        if audio_in or elapsed >= 20:
            ans_text = "No response recorded."
            bpm = 0
            if audio_in:
                temp = f"t_{idx}.wav"
                with open(temp, "wb") as f: f.write(audio_in.getvalue())
                y, sr = librosa.load(temp)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = int(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else int(tempo)
                ans_text = whisper_model.transcribe(temp)['text']
                os.remove(temp)
            
            st.session_state.results.append({"q": qs[idx], "a": ans_text, "bpm": bpm})
            
            if idx < 5:
                st.session_state.q_idx += 1
                st.session_state.start_time = time.time()
                st.rerun()
            else:
                # Flow Complete
                if st.button("SUBMIT INTERVIEW"):
                    st.session_state.stage = 'dashboard'
                    st.rerun()
        
        # Required for background timer to work in Streamlit
        time.sleep(1)
        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# STAGE 3: Final Dashboard
elif st.session_state.stage == 'dashboard':
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: #6b0f1a;'>FINAL PERFORMANCE REPORT</h2>", unsafe_allow_html=True)
    
    # 1. TOP LEVEL METRICS (Hardcoded as requested)
    avg_bpm = int(np.mean([item['bpm'] for item in st.session_state.results])) if st.session_state.results else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CONFIDENCE SCORE", "60%")
    with col2:
        st.metric("TOTAL WARNINGS", "2")
    with col3:
        st.metric("AVG SPEECH PACE", f"{avg_bpm} BPM")

    st.divider()

    # 2. COMMUNICATION TREND (BPM GRAPH)
    st.subheader("Speech Communication Trend")
    if st.session_state.results:
        bpm_trend = pd.DataFrame({
            'Question': [f"Q{i+1}" for i in range(len(st.session_state.results))],
            'BPM': [item['bpm'] for item in st.session_state.results]
        })
        st.line_chart(bpm_trend.set_index('Question'), color="#6b0f1a")
    
    

    st.divider()

    # 3. OVERALL REVIEW & KEYWORD SUGGESTIONS
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("###  Overall Review")
        st.write("You demonstrated a solid understanding of the technical requirements, though your delivery suggests room for increased engagement and vocal conviction.")

    with c2:
        
        role_map = {
            "Software Engineer": "Scalability, Latency, Design Patterns, Refactoring",
            "AI Engineer": "Hyperparameters, Overfitting, Inference, Backpropagation",
            "Cloud Engineer": "High Availability, Elasticity, Provisioning, Virtualization",
            "Data Analyst": "Normalization, Aggregation, Hypothesis Testing, Correlation",
            "Backend Intern": "Statelessness, Idempotency, Authentication, CRUD"
        }
        # Safely get the role from session state
        current_role = st.session_state.get('role', 'Software Engineer')
        suggested_keys = role_map.get(current_role, "Technical Terminology")
        
        st.markdown("### Communication Strategy")
        st.write(f"Improve your professional delivery by using keywords like: **{suggested_keys}**.")
        st.info("Using these terms demonstrates senior-level technical communication.")

    st.divider()

    st.markdown("###  Area of Improvement")
    st.warning("Focus on articulating technical trade-offs more clearly while maintaining consistent eye contact to boost your overall impact.")

    st.divider()

    if st.button("START NEW SESSION"):
        st.session_state.clear()
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
