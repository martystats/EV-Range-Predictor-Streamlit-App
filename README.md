# 🚗 EV Range Predictor – End-to-End Machine Learning & Streamlit Deployment

---

# 📌 Project Summary

This project presents a complete end-to-end Machine Learning workflow for predicting Electric Vehicle (EV) driving range using structured vehicle data.  

The objective was not only to build a predictive model, but to demonstrate a **full professional data science lifecycle**, including:

• Data cleaning  
• Exploratory Data Analysis (EDA)  
• Feature engineering  
• Model selection & evaluation  
• Model serialization  
• Streamlit deployment  
• Version control with Git & GitHub  

This repository reflects production-ready project structuring and deployment discipline.

---

# 🎯 Problem Statement

Accurately estimating EV driving range is critical for:

• Consumers making purchase decisions  
• Dealerships assessing vehicle value  
• Automotive analysts  
• Policy and energy planners  

The goal of this project was to:

> Build a supervised Machine Learning model capable of predicting EV driving range based on structured vehicle attributes.

---

# 🧠 Project Workflow (Step-by-Step)

## Step 1 — Data Cleaning & Preparation

• Validated dataset integrity  
• Removed inconsistencies  
• Handled missing values  
• Standardized feature formats  
• Encoded categorical variables  

Clean data foundation ensured reliable downstream modeling.

---

## Step 2 — Exploratory Data Analysis (EDA)

Performed:

• Univariate analysis  
• Bivariate relationship exploration  
• Correlation analysis  
• Feature impact inspection  

Key insights were documented directly in the notebook:
`EV_Range_Prediction_End_to_End.ipynb`

---

## Step 3 — Feature Engineering

• Encoded categorical variables  
• Structured model-ready dataset  
• Ensured proper feature alignment  
• Prepared final training matrix  

---

## Step 4 — Model Training & Evaluation

Multiple regression models were evaluated:

• Linear Regression  
• Decision Tree Regressor  
• Random Forest Regressor  

### ✅ Final Selected Model:
**Random Forest Regressor**

Reason for selection:
• Strong nonlinear handling  
• Reduced overfitting  
• Better generalization performance  
• Stable predictions across validation  

Model was serialized using `joblib` for deployment readiness.

---

## Step 5 — Model Persistence

Saved artifacts:

• `best_random_forest_model_*.pkl`  
• `model_metadata_*.pkl`  

This ensures:
• Reproducibility  
• Deployment consistency  
• Production portability  

---

## Step 6 — Streamlit Web Application

A user-interactive Streamlit interface was built to:

• Accept vehicle feature inputs  
• Generate real-time predictions  
• Provide accessible ML output  

The application file:

`app.py`

This transforms the ML model into a usable application.

---

# 🗂️ Repository Structure

EV-Range-Predictor-Streamlit-App/
│
├── EV_Range_Prediction_End_to_End.ipynb
├── app.py
├── best_random_forest_model_*.pkl
├── model_metadata_*.pkl
├── requirements.txt
└── README.md

This structure follows professional ML project organization standards.

---

# 🚀 How To Run Locally

1️⃣ Clone repository

git clone https://github.com/martystats/EV-Range-Predictor-Streamlit-App.git

2️⃣ Navigate into folder

cd EV-Range-Predictor-Streamlit-App

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Launch Streamlit app

streamlit run app.py

---

# 🛠️ Technologies Used

• Python  
• Pandas  
• NumPy  
• Scikit-learn  
• Joblib  
• Streamlit  
• Git  
• GitHub  

---

# 📈 Core Skills Demonstrated

✔ End-to-End Machine Learning Workflow  
✔ Data Cleaning & Feature Engineering  
✔ Model Comparison & Selection  
✔ Overfitting Awareness  
✔ Model Serialization  
✔ Streamlit Deployment  
✔ Version Control & Repository Structuring  
✔ Professional Documentation  

---

# 🔮 Future Improvements

• Hyperparameter tuning (GridSearchCV)  
• Cross-validation enhancements  
• Feature importance visualization in app  
• Deployment to Streamlit Cloud  
• CI/CD integration  

---

# 👤 Author

Martin Ude  
Data Science & Machine Learning Practitioner  

GitHub: https://github.com/martystats  

---

# ⭐ Closing Note

This project demonstrates structured, professional, and deployment-ready Machine Learning engineering.  

It reflects not only modeling ability, but also workflow discipline, reproducibility, and real-world application design.

If you found this project insightful, consider giving it a ⭐.
