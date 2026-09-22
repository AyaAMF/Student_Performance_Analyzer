# 🎓 Student Performance Analyzer

An AI-powered web application that predicts student academic performance
using Machine Learning.

## 🚀 Overview

Student Performance Analyzer is an interactive AI application built with
Python and Streamlit.

The system analyzes student demographic, academic, social, and lifestyle
information and uses a trained Machine Learning model to predict the
student's final academic grade.

The application also provides prediction history, analytics, and a
user authentication system.

## ✨ Features

- 🔐 User Registration & Login
- 🎓 Student Profile Management
- 🤖 AI-Based Performance Prediction
- 📊 Prediction Analytics
- 📈 Prediction History
- 📥 Export Prediction History as CSV
- 👤 User Profile
- 🌙 Modern Dark AI Dashboard
- 🗄️ SQLite Database

## 🧠 Machine Learning

The project uses a Random Forest Regression model.

### Target

The model predicts:

`G3`

which represents the student's final grade.

### Leakage Prevention

`G1` and `G2` were excluded from the model input because they are
previous-period grades that could provide direct information about the
final grade.

### Features

The model uses demographic, academic, social, and lifestyle features
such as:

- Age
- Mother's education
- Father's education
- Study time
- Past failures
- Absences
- Health
- Family relationship
- Free time
- Going out
- Travel time
- Parental support
- Internet access
- Extracurricular activities
- And other student attributes

## 📊 Model Evaluation

The baseline Random Forest model achieved approximately:

- MAE: 2.97
- RMSE: 3.75
- R²: 0.31

The model was evaluated using a train/test split and cross-validation.

## 🏗️ Project Structure

```text
Student_Performance_Analyzer/
│
├── data/
│   └── student-mat.csv
│
├── models/
│   └── student_performance_model.pkl
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
