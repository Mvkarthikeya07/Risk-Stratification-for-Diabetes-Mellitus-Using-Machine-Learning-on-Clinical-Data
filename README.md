🧠 Risk Stratification for Diabetes Mellitus Using Machine Learning on Clinical Data

A Full-Stack Machine Learning Application for Clinical Risk Assessment

📌 Overview

The Clinical Diabetes Prediction System is a machine learning–powered web application that predicts the risk of diabetes using real medical diagnostic features such as glucose level, BMI, age, insulin, and blood pressure.

The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, supervised model training, model serialization, REST-style API integration, and a deployment-ready web interface built using Flask. It highlights the practical application of AI in healthcare decision-support systems.

🎯 Objectives

Build a clinical risk prediction model using supervised machine learning

Apply data preprocessing, imputation, and feature scaling techniques

Deploy the trained model via a Flask-based web application

Provide probability-based predictions for interpretability

Demonstrate applied machine learning in a healthcare context
```
🚀 Key Capabilities

✔ End-to-end ML pipeline (data → model → deployment)
✔ Logistic Regression–based diabetes prediction
✔ Real-time prediction with probability output
✔ Clean and responsive web interface
✔ REST-style API endpoints
✔ Modular, maintainable, and extensible codebase
```
🧠 Machine Learning Workflow
🧹 Data Preprocessing

Handling missing values using imputation

Feature scaling using standardization

Ensuring consistent preprocessing during training and inference

🤖 Model

Logistic Regression for binary classification

Probability-based output to represent risk level

💾 Model Persistence

Trained pipeline saved using Joblib

Enables reproducibility and efficient reuse

🔮 Prediction

User inputs are processed through the preprocessing pipeline

Model outputs:

Diabetes risk probability (%)

Classification (Diabetic / Non-Diabetic)

🧪 Example Predictions
🔴 High-Risk Sample

Input

Glucose: 180

BMI: 36

Age: 45

Insulin: 94

Pregnancies: 2

Output

Probability: 77.7%

Prediction: Diabetes Positive (High Risk)

🟢 Low-Risk Sample

Input

Glucose: 95

BMI: 24

Age: 25

Output

Probability: 12.4%

Prediction: Diabetes Negative (Low Risk)

📡 API Endpoints
Endpoint	Method	Description
/	GET	Loads frontend UI
/predict	POST	Predicts diabetes risk & probability
/health	GET	Checks model availability
/data-head	GET	Returns first 5 rows of dataset
/data-tail	GET	Returns last 5 rows of dataset
🧠 Dataset Description

The dataset (diabetes_user.csv) contains medical diagnostic features:

Feature	Description
Pregnancies	Number of pregnancies
Glucose	Plasma glucose concentration
BloodPressure	Diastolic blood pressure
SkinThickness	Skin fold thickness
Insulin	2-hour serum insulin
BMI	Body mass index
DiabetesPedigreeFunction	Hereditary risk score
Age	Age in years
Outcome	1 = Diabetic, 0 = Non-Diabetic
🛠 Tech Stack
Backend & Machine Learning

Python

Pandas, NumPy

Scikit-learn

Joblib

Flask

Frontend

HTML5

CSS3

JavaScript
````
📂 Project Structure
diabetes_project/
│
├── static/
│   └── style.css                  # Styling
│
├── templates/
│   └── index.html                 # Web UI
│
├── app.py                         # Flask API + frontend
├── train_model.py                 # Model training script
├── diabetes_user.csv              # Dataset
├── model.joblib                   # Trained ML pipeline
├── requirements.txt
└── README.md
````
🖼️ Screenshots
Figure 1: Clean Clinical Input Interface

<img width="1366" height="768" alt="2025-11-26 (2)" src="https://github.com/user-attachments/assets/f29980cb-f5ee-4b6e-87f5-570ba6106c38" />

Figure 2: Low-Risk Diabetes Prediction

<img width="1366" height="768" alt="2025-11-26 (1)" src="https://github.com/user-attachments/assets/6754634a-b823-4f18-b541-f2dc565c67f5" />

Figure 3: High-Risk Diabetes Prediction

<img width="1366" height="768" alt="2025-11-26" src="https://github.com/user-attachments/assets/c723845d-7638-47de-9b93-0f43fd7e453e" />

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train_model.py

4️⃣ Run the Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000/

🏢 Internship Context

This project was developed during my AI/ML Internship at InternPe.

The work focused on applying practical machine learning and full-stack development concepts, including:

Data preprocessing and feature engineering

Supervised classification using Logistic Regression

Backend development using Flask

REST-style API design for ML inference

Real-time probability-based risk prediction

Integration of ML models into healthcare web applications

This project represents academic and practical work completed during the internship period, demonstrating the application of machine learning in clinical decision-support systems.

🔮 Future Enhancements

Deployment on Render / Heroku

User authentication and prediction history

SHAP-based explainability visualizations

Feature distribution and statistical analysis plots

UI upgrade using Bootstrap or Material UI

👤 Author

M V Karthikeya
Machine Learning & Python Developer

GitHub: https://github.com/Mvkarthikeya07

📜 License

This project is licensed under the MIT License.

⭐ Final Remarks

This project demonstrates a production-ready healthcare machine learning system, highlighting strong capabilities in data preprocessing, supervised modeling, backend development, and real-time deployment, making it suitable for academic evaluation, professional portfolios, and research-oriented applications.
