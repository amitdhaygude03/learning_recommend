import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# User Authentication
# ---------------------------
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "learner"},
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = users[username]["role"]
            st.session_state.username = username
            st.success(f"Welcome {username}! Role: {st.session_state.role}")
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

# ---------------------------
# Load or Create Datasets
# ---------------------------
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

def load_data(filename, default_df):
    path = os.path.join(DATA_PATH, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        default_df.to_csv(path, index=False)
        return default_df

courses_df = load_data(
    "courses.csv",
    pd.DataFrame([
        {"title": "Python Basics", "description": "Learn Python fundamentals", "difficulty": "Beginner", "hours": 10},
        {"title": "Data Science 101", "description": "Intro to Data Science", "difficulty": "Intermediate", "hours": 20},
        {"title": "Deep Learning", "description": "Neural networks and deep learning", "difficulty": "Advanced", "hours": 30},
    ])
)

books_df = load_data(
    "books.csv",
    pd.DataFrame([
        {"title": "Hands-On ML", "author": "Geron", "description": "Machine learning with Scikit-learn & TensorFlow"},
        {"title": "Deep Learning", "author": "Goodfellow", "description": "Foundations of deep learning"},
    ])
)

projects_df = load_data(
    "projects.csv",
    pd.DataFrame([
        {"title": "Stock Prediction", "description": "Predict stock prices using ML", "difficulty": "Intermediate", "hours": 25},
        {"title": "Churn Prediction", "description": "Telecom churn model", "difficulty": "Intermediate", "hours": 15},
    ])
)

# ---------------------------
# Recommender System
# ---------------------------
def recommend(user_input, df, text_column="description", top_n=3):
    corpus = df[text_column].fillna("").tolist() + [user_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_idx = similarity[0].argsort()[::-1][:top_n]
    return df.iloc[top_idx]

# ---------------------------
# Guided Path Generator
# ---------------------------
def generate_path(df):
    df_sorted = df.sort_values(by=["difficulty", "hours"], ascending=[True, True])
    return df_sorted

# ---------------------------
# Admin Page
# ---------------------------
def admin_page():
    st.title("⚙️ Admin Panel")
    st.write("Upload new CSVs to update datasets.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    dataset_choice = st.selectbox("Select dataset to replace", ["courses", "books", "projects"])
    if uploaded_file and st.button("Upload"):
        df_new = pd.read_csv(uploaded_file)
        filename = f"{dataset_choice}.csv"
        df_new.to_csv(os.path.join(DATA_PATH, filename), index=False)
        st.success(f"{dataset_choice}.csv updated!")

# ---------------------------
# Learner Dashboard
# ---------------------------
def learner_page():
    st.title("🎓 Personalized Learning Recommender")

    skills = st.text_input("Enter your skills")
    goals = st.text_input("Enter your learning goals")
    if st.button("Get Recommendations"):
        if skills or goals:
            query = skills + " " + goals
            st.subheader("📘 Recommended Courses")
            st.write(recommend(query, courses_df))
            st.subheader("📚 Recommended Books")
            st.write(recommend(query, books_df))
            st.subheader("💻 Recommended Projects")
            st.write(recommend(query, projects_df))
        else:
            st.warning("Please enter some skills or goals.")

    st.subheader("📊 Guided Learning Path")
    st.write(generate_path(courses_df))

# ---------------------------
# App Flow
# ---------------------------
if not st.session_state.logged_in:
    login()
else:
    st.sidebar.write(f"Logged in as {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        logout()
        st.experimental_rerun()

    if st.session_state.role == "admin":
        admin_page()
    else:
        learner_page()
