# 🎓 Personalized Learning Recommender (EdTech) — Streamlit App

This is a high-level **EdTech Streamlit app** designed for placement portfolios.  
It recommends **courses, books, and projects** based on the user’s entered skills and career goals.  
The app also includes **user authentication, admin dataset upload, and guided learning path generation**.

---

## 🚀 Features
- **Personalized Recommendations**
  - Enter your skills & goals → get tailored recommendations (courses, books, projects).
  - Uses TF-IDF similarity on skills & goals.

- **Progress Tracking Dashboard**
  - Add learning items to your plan.
  - Mark them as completed or track progress percentage.
  - Visual analytics (progress bars, charts).

- **User Authentication**
  - Simple login system (User vs Admin roles).

- **Admin Page**
  - Upload new CSV datasets (courses, books, projects).
  - Manage resources dynamically.

- **Guided Learning Path**
  - Sequences items by difficulty & estimated hours.
  - Generates a structured path to follow.

---

## 📂 Project Structure
```
├── app.py                   # Main Streamlit app code
├── recommender_model.pkl    # Pickled recommender model placeholder
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## 🛠️ Installation & Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/edtech-recommender.git
cd edtech-recommender
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate    # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📊 Usage
- **Learners**: Login → Enter skills & goals → Get recommendations → Add to plan → Track progress.
- **Admins**: Login as admin → Upload CSV files with new learning resources.

---

## 📘 Example CSV Format

### courses.csv
```
title,description,difficulty,hours
Python Basics,"Learn Python fundamentals",Beginner,10
Data Science 101,"Intro to Data Science",Intermediate,20
```

### books.csv
```
title,author,description
Hands-On ML,Geron,"Machine learning with Scikit-learn & TensorFlow"
Deep Learning,Goodfellow,"Foundations of deep learning"
```

### projects.csv
```
title,description,difficulty,hours
Stock Prediction,"Predict stock prices using ML",Intermediate,25
Churn Prediction,"Telecom customer churn model",Intermediate,15
```

---

## 📦 Requirements
- Streamlit
- Pandas
- Numpy
- Scikit-learn

(Already listed in `requirements.txt`)

---

## 🎯 Future Enhancements
- Integrate **real datasets / APIs** for courses & books.
- Add **database support** for persistent user data.
- Enhance recommender system with **collaborative filtering** or **hybrid models**.
- Add **graph-based prerequisite paths** for guided learning.

---

## 🧑‍💻 Author
Developed by **[Your Name]**  
Data Science | AI | ML Enthusiast

---

⭐ If you like this project, give it a star on GitHub!
