# ==============================
# Career Transition Predictor
# Academic Final Version
# ==============================

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/candidate_job_role_dataset.csv")

# Expecting columns:
# skills, job_role

# ------------------------------
# 2. Preprocess Skills
# ------------------------------
df["skills"] = df["skills"].str.lower().str.split(",")

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["skills"])
y = df["job_role"]

# ------------------------------
# 3. Train Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully!")

# ------------------------------
# 4. USER INPUT
# ------------------------------
print("\n====== Career Transition Predictor ======")
current_role = input("Enter your CURRENT role: ")
target_role = input("Enter your TARGET role: ")
user_skills = input("Enter your skills (comma separated): ").lower().split(",")

# Clean spaces
user_skills = [s.strip() for s in user_skills]

# ------------------------------
# 5. Predict Role Match
# ------------------------------
user_vector = mlb.transform([user_skills])
predicted_role = model.predict(user_vector)[0]

# Probability for target role
probs = model.predict_proba(user_vector)[0]
classes = model.classes_

if target_role in classes:
    feasibility_score = int(probs[list(classes).index(target_role)] * 100)
else:
    feasibility_score = 0

# ------------------------------
# 6. Skill Gap Analysis
# ------------------------------
# Get skills required by target role from dataset
target_rows = df[df["job_role"] == target_role]

if len(target_rows) > 0:
    required_skills = set(sum(target_rows["skills"].tolist(), []))
else:
    required_skills = set()

user_skill_set = set(user_skills)

matched = user_skill_set & required_skills
missing = required_skills - user_skill_set
partial = set()  # simplified version

# ------------------------------
# 7. Effort Estimation
# ------------------------------
missing_count = len(missing)

if missing_count <= 2:
    effort = "Low (1–2 months)"
elif missing_count <= 5:
    effort = "Medium (3–6 months)"
else:
    effort = "High (6–12 months)"

# ------------------------------
# 8. Final Output
# ------------------------------
print("\n========== RESULTS ==========")
print(f"Current Role: {current_role}")
print(f"Target Role: {target_role}")

print(f"\nFeasibility Score: {feasibility_score}/100")

print("\nMatched Skills:")
for s in matched:
    print(f"✔ {s}")

print("\nMissing Skills:")
for s in missing:
    print(f"✘ {s}")

print(f"\nEstimated Learning Effort: {effort}")

# Explainability
print("\nKey Factors:")
if feasibility_score > 60:
    print("- Strong skill overlap")
else:
    print("- Significant skill gaps detected")

print("\n=============================")