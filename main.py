# =====================================================
# Career Transition AI (Resume-Level Version)
# =====================================================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import re

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Career Transition AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Career Transition AI")
st.caption("AI-powered career feasibility & skill gap analysis")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/candidate_job_role_dataset.csv")
    df["skills"] = df["skills"].str.lower().str.split(",")
    return df

df = load_data()

# -------------------------
# Train Model
# -------------------------
@st.cache_resource
def train_model(df):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["skills"])
    y = df["job_role"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, mlb

model, mlb = train_model(df)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("🧑‍💻 Your Profile")

current_role = st.sidebar.text_input("Current Role")
target_role = st.sidebar.text_input("Target Role")

skills_input = st.sidebar.text_area(
    "Enter Skills (comma separated)",
    placeholder="python, sql, statistics"
)

resume = st.sidebar.file_uploader("📄 Upload Resume (optional)", type=["txt"])

# -------------------------
# Resume Skill Extraction
# -------------------------
def extract_skills_from_text(text, known_skills):
    text = text.lower()
    found = []
    for skill in known_skills:
        if re.search(rf"\b{skill}\b", text):
            found.append(skill)
    return found

if resume:
    text = resume.read().decode("utf-8")
    all_skills = list(set(sum(df["skills"], [])))
    extracted = extract_skills_from_text(text, all_skills)
    if extracted:
        st.sidebar.success(f"Detected Skills: {', '.join(extracted)}")
        skills_input = ", ".join(extracted)

# -------------------------
# Predict Button
# -------------------------
if st.sidebar.button("🚀 Analyze Transition"):

    if not target_role or not skills_input:
        st.warning("Please provide target role and skills")
        st.stop()

    user_skills = [s.strip().lower() for s in skills_input.split(",")]
    user_vector = mlb.transform([user_skills])

    probs = model.predict_proba(user_vector)[0]
    classes = model.classes_

    if target_role in classes:
        feasibility_score = int(probs[list(classes).index(target_role)] * 100)
    else:
        feasibility_score = 0

    # -------------------------
    # Skill Gap Analysis
    # -------------------------
    target_rows = df[df["job_role"] == target_role]

    if len(target_rows) > 0:
        required_skills = set(sum(target_rows["skills"].tolist(), []))
    else:
        required_skills = set()

    user_skill_set = set(user_skills)
    matched = user_skill_set & required_skills
    missing = required_skills - user_skill_set

    # -------------------------
    # Effort Estimation
    # -------------------------
    missing_count = len(missing)

    if missing_count <= 2:
        effort = "🟢 Low (1–2 months)"
    elif missing_count <= 5:
        effort = "🟡 Medium (3–6 months)"
    else:
        effort = "🔴 High (6–12 months)"

    # =========================
    # MAIN DASHBOARD
    # =========================
    st.subheader("📊 Transition Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("🎯 Feasibility Score", f"{feasibility_score}/100")
    col2.metric("✅ Matched Skills", len(matched))
    col3.metric("❌ Missing Skills", len(missing))

    st.progress(feasibility_score / 100)

    # -------------------------
    # Skills Columns
    # -------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ✅ You Already Have")
        for s in sorted(matched):
            st.success(s)

    with c2:
        st.markdown("### 🎯 Skills to Learn")
        for s in sorted(missing):
            st.error(s)

    st.info(f"⏱ Estimated Effort: {effort}")

    # -------------------------
    # Recommendations
    # -------------------------
    st.markdown("### 🧠 AI Recommendations")

    if feasibility_score > 70:
        st.success("Strong transition potential. Start building projects and applying.")
    elif feasibility_score > 40:
        st.warning("Moderate feasibility. Focus on closing top skill gaps.")
    else:
        st.error("Low feasibility. Consider stepping-stone roles first.")

    # -------------------------
    # Alternative Roles
    # -------------------------
    st.markdown("### 🔄 Suggested Alternative Roles")

    similarity_scores = dict(zip(classes, probs))
    top_roles = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[1:4]

    for role, score in top_roles:
        st.write(f"➡ {role} ({int(score*100)}% match)")

    # -------------------------
    # Learning Roadmap
    # -------------------------
    if missing:
        st.markdown("### 📚 Suggested Learning Roadmap")
        for i, skill in enumerate(list(missing)[:5], 1):
            st.write(f"{i}. Learn {skill} → Build 1 project")

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning • Portfolio Ready")
