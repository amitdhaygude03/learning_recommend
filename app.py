# app.py — Personalized Learning Recommendation App (EdTech) with Admin, Auth, Guided Path
# -----------------------------------------------------------------------------
# Features:
#  - User login (simple auth)
#  - Admin page: upload CSVs (courses, books, projects)
#  - Personalized recommendations (content-based)
#  - Guided learning path (sequence by difficulty/est. hours)
#  - Progress tracking dashboard
# -----------------------------------------------------------------------------

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "🎓 Personalized Learning Recommender"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

COURSES_FILE = DATA_DIR / "courses.csv"
BOOKS_FILE = DATA_DIR / "books.csv"
PROJECTS_FILE = DATA_DIR / "projects.csv"
PROGRESS_FILE = DATA_DIR / "user_plan.json"

# -----------------------------
# Authentication
# -----------------------------
USERS = {
    "admin": "admin123",  # admin role
    "user": "user123"     # normal user role
}


def login():
    st.sidebar.subheader("🔑 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.user = username
            st.session_state.role = "admin" if username == "admin" else "user"
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid credentials")


def require_login():
    if "user" not in st.session_state:
        st.warning("Please log in to access this feature.")
        st.stop()


# -----------------------------
# Data Handling
# -----------------------------
DEFAULT_COURSES = [
    ["id", "title", "provider", "level", "tags", "description", "url", "est_hours"],
    ["c1", "Python for Data Science", "Coursera", "Beginner", "python, pandas", "Intro to Python", "http://coursera.org", 20]
]
DEFAULT_BOOKS = [["id", "title", "provider", "level", "tags", "description", "url", "est_hours"]]
DEFAULT_PROJECTS = [["id", "title", "provider", "level", "tags", "description", "url", "est_hours"]]


def ensure_csv(file: Path, default_data: List[List]):
    if not file.exists():
        pd.DataFrame(default_data[1:], columns=default_data[0]).to_csv(file, index=False)


for f, d in [(COURSES_FILE, DEFAULT_COURSES), (BOOKS_FILE, DEFAULT_BOOKS), (PROJECTS_FILE, DEFAULT_PROJECTS)]:
    ensure_csv(f, d)


def load_datasets() -> Dict[str, pd.DataFrame]:
    return {
        "courses": pd.read_csv(COURSES_FILE),
        "books": pd.read_csv(BOOKS_FILE),
        "projects": pd.read_csv(PROJECTS_FILE)
    }


# -----------------------------
# Recommender Engine
# -----------------------------

def _combine_text(row: pd.Series) -> str:
    return " ".join([str(row.get("title", "")), str(row.get("tags", "")), str(row.get("description", ""))])


def build_tfidf(df: pd.DataFrame):
    corpus = df.apply(_combine_text, axis=1).values
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


@st.cache_resource(show_spinner=False)
def get_models(datasets: Dict[str, pd.DataFrame]):
    return {name: build_tfidf(df) for name, df in datasets.items()}


def recommend(df, vec, X, query, top_k=5):
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, X).ravel()
    df = df.copy()
    df["similarity"] = sims
    return df.sort_values("similarity", ascending=False).head(top_k)


# -----------------------------
# Guided Learning Path
# -----------------------------

def guided_path(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by level then est_hours
    level_order = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    df = df.copy()
    df["level_num"] = df["level"].map(level_order).fillna(2)
    return df.sort_values(["level_num", "est_hours"])


# -----------------------------
# Progress Persistence
# -----------------------------

def load_plan():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"items": []}


def save_plan(plan):
    PROGRESS_FILE.write_text(json.dumps(plan, indent=2))


# -----------------------------
# Admin Page
# -----------------------------

def admin_page():
    st.subheader("⚙️ Admin Panel: Manage Datasets")
    st.write("Upload CSVs to replace current datasets.")
    for name, file in [("Courses", COURSES_FILE), ("Books", BOOKS_FILE), ("Projects", PROJECTS_FILE)]:
        uploaded = st.file_uploader(f"Upload {name} CSV", type="csv", key=f"u_{name}")
        if uploaded:
            pd.read_csv(uploaded).to_csv(file, index=False)
            st.success(f"{name} dataset updated!")


# -----------------------------
# Main App
# -----------------------------

def main():
    st.set_page_config(page_title="Learning Recommender", page_icon="🎓", layout="wide")
    st.title(APP_TITLE)

    login()
    require_login()

    datasets = load_datasets()
    models = get_models(datasets)

    if st.session_state.role == "admin":
        with st.sidebar:
            if st.button("Go to Admin Page"):
                st.session_state.page = "Admin"

    page = st.session_state.get("page", "Recommend")
    tabs = ["Recommend", "Track Progress", "Guided Path"]
    if st.session_state.role == "admin":
        tabs.append("Admin")

    tab = st.radio("Navigate", tabs, index=tabs.index(page))
    st.session_state.page = tab

    if tab == "Recommend":
        skills = st.text_input("Enter skills (comma-separated)")
        goals = st.text_input("Enter goals")
        query = skills + " " + goals
        for name, df in datasets.items():
            vec, X = models[name]
            st.subheader(f"Recommended {name.title()}")
            recs = recommend(df, vec, X, query)
            st.dataframe(recs[["title", "provider", "level", "similarity"]])

    elif tab == "Track Progress":
        plan = load_plan()
        st.subheader("Your Learning Plan")
        df = pd.DataFrame(plan["items"])
        st.data_editor(df, num_rows="dynamic")
        if st.button("Save Plan"):
            plan["items"] = df.to_dict(orient="records")
            save_plan(plan)

    elif tab == "Guided Path":
        st.subheader("🧭 Suggested Guided Path")
        combined = pd.concat(datasets.values())
        path = guided_path(combined)
        st.dataframe(path[["title", "level", "est_hours"]])

    elif tab == "Admin":
        admin_page()


if __name__ == "__main__":
    main()
