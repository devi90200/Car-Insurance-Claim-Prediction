🚗 Car Insurance Claim Prediction System:

AI-Driven Risk Assessment & Underwriting Intelligence

📌 Project Overview:

    The Car Insurance Claim Prediction project is an end-to-end machine learning solution designed to predict whether a policyholder is likely to file an insurance claim in the next policy period.

    By leveraging demographic data, vehicle specifications, and policy attributes, this system helps insurance companies make data-driven underwriting, pricing, and risk management decisions.

    The project follows a production-ready ML lifecycle, from data exploration and feature engineering to model deployment via an interactive Streamlit dashboard.

🎯 Problem Statement

    Insurance companies face significant financial risk due to unpredictable claim behavior.

The objective of this project is to:

    Predict the probability of a car insurance claim (is_claim) using customer, vehicle, and policy-related features.

    This prediction enables insurers to proactively manage risk, reduce losses, and optimize operational efficiency.

🛠 Skills & Technologies Gained:
📊 Data & Analytics:

Exploratory Data Analysis (EDA)

Data visualization using Matplotlib, Seaborn, Plotly

Feature engineering & selection

🤖 Machine Learning:

Classification algorithms:

Logistic Regression

Decision Tree

Random Forest

XGBoost / LightGBM

Model evaluation:

Accuracy

Precision & Recall

F1-Score

ROC-AUC

Log-Loss (probability calibration)

⚙️ Engineering & Deployment:

Scikit-learn Pipelines

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Streamlit dashboard deployment

AWS-ready / production-structured project

Git version control & documentation

🎯 Target Variable:

    is_claim

        0 → No claim

        1 → Claim filed

🔍 Project Approach

        ->Exploratory Data Analysis (EDA)
        ->Data Preprocessing
        ->Model Development
        ->Model Evaluation
        ->Hyperparameter Tuning
        ->Deployment

📈 Results & Outcomes:

    ✅ Accurate prediction of claim probability

    🔍 Identification of key risk-driving features

    📊 Interactive dashboard for business interpretation

    ⚙️ Reproducible ML pipeline

    🌐 Deployed web application for real-time prediction

📊 Risk Interpretation Framework:

    Probability Range	    Risk Level	         Business Action

   < 5%	                    Low Risk	         Standard Premium
   5–10%	              Moderate Risk	         Monitor Customer
   10–20%	               High Risk	         Higher Premium
   > 20%	             Very High Risk	         Manual Review

🧪 Evaluation Metrics:

              Accuracy

              Precision

              Recall

              F1-Score

              ROC-AUC

🚀 Deployment:

             Framework: Streamlit

             Model Serialization: Joblib

             Deployment Ready: AWS

📁 Project Structure
car_insurance_claim/
│
├── app.py
├── notebook/
│   └── model/
│       └── final_insurance_claim_model.pkl
|   └── eda.ipynb
├── assests/
│   ├── car.gif
│   ├── carj.gif
│   └── insurance_banner.gif
├── requirements.txt
└── README.md
