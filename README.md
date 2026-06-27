<div align="center">

<h1>🩺 Diabetes Risk Stratification System</h1>
<h3>Clinical Risk Prediction Using Supervised Machine Learning on Diagnostic Biomarker Data</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Dataset-98%20Clinical%20Records-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Features-8%20Biomarkers-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.736-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Context-AI%2FML%20Internship%20%40%20InternPe-red?style=flat-square"/>
</p>

> A **full-stack clinical decision-support web application** — real-time diabetes risk stratification from 8 diagnostic biomarkers using a median-imputed, standardized Logistic Regression pipeline, deployed via Flask with a JSON REST API and probability-calibrated predictions.

</div>

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Live Screenshots](#-live-screenshots)
- [System Architecture](#-system-architecture)
- [ML Pipeline — How It Works](#-ml-pipeline--how-it-works)
- [Dataset Details](#-dataset-details)
- [Model Comparison & Benchmarks](#-model-comparison--benchmarks)
- [Classification Report](#-classification-report)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Example Predictions](#-example-predictions)
- [Engineering Highlights](#-engineering-highlights)
- [Internship Context](#-internship-context)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

Diabetes Mellitus is a global health crisis affecting over 500 million adults. Early detection dramatically improves outcomes — but clinical assessment is time-consuming, expertise-dependent, and inaccessible in resource-constrained settings. Routine diagnostic data (glucose, BMI, blood pressure, insulin) is widely collected but rarely leveraged for automated risk stratification.

---

## 💡 Solution Overview

This system trains a **scikit-learn pipeline** (median imputation → standard scaling → Logistic Regression) on 98 clinical records with 8 biomarker features. The trained model is serialized and served via a Flask REST API — accepting form or JSON input, returning a binary prediction and a calibrated probability score in real time. A clean two-panel web UI displays risk level with color-coded output.

---

## 🖥️ Live Screenshots

### Figure 1 — Clinical Input Interface

<img width="1366" height="768" alt="Clinical Input Interface" src="https://github.com/user-attachments/assets/f29980cb-f5ee-4b6e-87f5-570ba6106c38"/>

> Eight biomarker fields pre-populated with median values. The 2-column grid layout keeps all inputs visible without scrolling. JSON is posted to `/predict` via `fetch()` — no page reload.

---

### Figure 2 — Low-Risk Prediction Result

<img width="1366" height="768" alt="Low Risk Prediction" src="https://github.com/user-attachments/assets/6754634a-b823-4f18-b541-f2dc565c67f5"/>

> Glucose: 95, BMI: 24, Age: 25 → **12.4% probability** → `Diabetes Negative (Low Risk)` in green.

---

### Figure 3 — High-Risk Prediction Result

<img width="1366" height="768" alt="High Risk Prediction" src="https://github.com/user-attachments/assets/c723845d-7638-47de-9b93-0f43fd7e453e"/>

> Glucose: 180, BMI: 36, Age: 45, Insulin: 94 → **77.7% probability** → `Diabetes Positive (High Risk)` in red.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                             │
│   index.html  →  fetch POST /predict (JSON)  →  display result      │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  HTTP JSON POST
┌─────────────────────────────▼────────────────────────────────────────┐
│                      Flask Application (app.py)                      │
│                                                                      │
│   /health     → model load status check                             │
│   /predict    → build_input_dataframe() → model.predict_proba()     │
│   /reload-model → hot-reload model without restart                  │
│   /data-head  → first 5 rows of dataset (JSON)                      │
│   /data-tail  → last 5 rows of dataset (JSON)                       │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│                  sklearn Pipeline (model.joblib)                      │
│                                                                      │
│   Step 1: SimpleImputer(strategy="median")                          │
│           → replaces medically impossible 0s (NaN) with median      │
│                                                                      │
│   Step 2: StandardScaler()                                          │
│           → zero-mean, unit-variance normalization                  │
│                                                                      │
│   Step 3: LogisticRegression(max_iter=2000)                         │
│           → binary classification + predict_proba()                 │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     🔴 Diabetic (1)                🟢 Non-Diabetic (0)
     + probability %               + probability %
```

---

## 🔬 ML Pipeline — How It Works

### Step 1 — Data Ingestion & Zero-Imputation

Medically impossible zero values exist in 5 features — a patient cannot have zero glucose, zero BMI, or zero blood pressure. These are treated as missing data and replaced with `np.nan` before any processing:

```python
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col] = df[col].replace(0, np.nan)
```

**Zero-value counts in the dataset:**

| Feature | Zero Count | Clinical Meaning |
|---|---|---|
| Insulin | 55 / 98 | Missing lab measurement |
| SkinThickness | 32 / 98 | Missing anthropometric reading |
| BloodPressure | 6 / 98 | Missing vital |
| BMI | 4 / 98 | Missing anthropometric reading |
| Glucose | 1 / 98 | Implausible — missing |

### Step 2 — Median Imputation

```python
SimpleImputer(strategy="median")
```

Median is chosen over mean because clinical biomarker distributions (especially Insulin) are heavily right-skewed. Median imputation is robust to the extreme outliers present in the data (e.g., Insulin max = 846).

### Step 3 — Standard Scaling

```python
StandardScaler()
```

Logistic Regression uses gradient-based optimization — features on vastly different scales (Glucose: 0–197, Pregnancies: 0–15) would cause unequal gradient contributions. StandardScaler normalizes each feature to zero mean and unit variance.

### Step 4 — Logistic Regression

```python
LogisticRegression(max_iter=2000)
```

Binary classifier outputting both a class label and a calibrated probability. `max_iter=2000` ensures full convergence on the imputed, scaled data.

### Step 5 — Atomic Model Serialization

```python
# Atomic save: write to .tmp then os.replace() — prevents corrupt model on crash
joblib.dump(model, "model.joblib.tmp")
os.replace("model.joblib.tmp", "model.joblib")
```

### Step 6 — Named-Column Inference (Prevents sklearn Warnings)

```python
def build_input_dataframe(data):
    # Builds DataFrame with correct column names — not a raw numpy array
    # Ensures sklearn pipeline receives expected feature names at inference time
    df = pd.DataFrame([values], columns=FEATURES)
    return df
```

---

## 📊 Dataset Details

**Source:** Pima Indians Diabetes Dataset (clinical subset — `diabetes_user.csv`)

| Property | Value |
|---|---|
| Total records | 98 |
| Features | 8 biomarkers |
| Target | `Outcome` (0 = Non-Diabetic, 1 = Diabetic) |
| Class distribution | 62 Non-Diabetic (63.3%) · 36 Diabetic (36.7%) |
| Train / Test split | 78 train · 20 test (80/20, stratified) |

**Feature Summary:**

| Feature | Mean | Std | Clinical Significance |
|---|---|---|---|
| Pregnancies | 4.63 | 3.61 | Gestational diabetes risk factor |
| Glucose | 118.15 | 33.96 | Primary diagnostic indicator |
| BloodPressure | 68.03 | 22.05 | Hypertension correlation |
| SkinThickness | 19.48 | 16.13 | Body fat proxy |
| Insulin | 72.97 | 126.87 | Pancreatic function indicator |
| BMI | 30.83 | 9.52 | Obesity risk factor |
| DiabetesPedigreeFunction | 0.47 | 0.37 | Hereditary risk score |
| Age | 34.57 | 11.26 | Risk increases with age |

---

## ⚖️ Model Comparison & Benchmarks

All six models were benchmarked on the **same train/test split** (80/20, stratified, `random_state=42`) with the **same preprocessing pipeline** (median imputation + standard scaling) to ensure a fair, apples-to-apples comparison. Results are computed on the actual `diabetes_user.csv` dataset.

| Model | Test Accuracy | 5-Fold CV Accuracy | ROC-AUC | Interpretable | Used |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.650** | **0.674** | **0.736** | ✅ Yes | ✅ **★** |
| Gradient Boosting | 0.700 | 0.675 | **0.769** | ⚠️ Partial | ❌ |
| Decision Tree | 0.700 | 0.593 | 0.637 | ✅ Yes | ❌ |
| SVM (RBF) | 0.650 | 0.634 | 0.703 | ❌ No | ❌ |
| Random Forest | 0.600 | 0.664 | 0.687 | ⚠️ Partial | ❌ |
| KNN (k=5) | 0.550 | 0.645 | 0.659 | ❌ No | ❌ |

> Metrics: Test Accuracy on held-out 20% · 5-Fold CV Accuracy on full dataset · ROC-AUC on held-out 20%.

### Why Logistic Regression was chosen

| Criterion | Reasoning |
|---|---|
| **Calibrated probabilities** | LR natively outputs well-calibrated probabilities — essential for clinical risk communication ("77.7% risk") rather than a bare label |
| **Interpretability** | Each feature has a learned coefficient — clinicians can understand *why* a prediction was made |
| **Regularization** | L2 regularization prevents overfitting on the 98-record dataset; tree-based models overfit heavily on small data (Decision Tree CV = 0.593 vs test = 0.700 — clear overfitting) |
| **Stable CV accuracy** | LR has the smallest gap between CV (0.674) and test (0.650) accuracy — most generalizable model on this dataset size |
| **Clinical standard** | Logistic regression is the clinical gold standard for binary risk prediction — directly comparable to AUROC-reported clinical studies |
| **Convergence guarantee** | `max_iter=2000` ensures full optimization convergence on the scaled feature space |

### ROC-AUC Analysis

ROC-AUC is the **primary evaluation metric** for medical classifiers — it measures discrimination ability independent of the decision threshold.

| AUC Range | Clinical Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent |
| 0.80 – 0.90 | Good |
| **0.70 – 0.80** | **Fair ← LR: 0.736** |
| 0.60 – 0.70 | Poor |
| 0.50 – 0.60 | Fail |

> **Note on dataset size:** With only 98 records and a 20-sample test set, all accuracy metrics carry high variance. The 5-fold CV accuracy (0.674) is the more reliable estimate of generalization performance. Gradient Boosting achieves higher AUC (0.769) but shows CV instability — it would be the preferred model with a larger dataset.

---

## 📈 Classification Report (Logistic Regression)

Evaluated on the 20-sample held-out test set:

```
              precision    recall    f1-score   support
─────────────────────────────────────────────────────
Non-Diabetic     0.71      0.77      0.74        13
    Diabetic     0.50      0.43      0.46         7
─────────────────────────────────────────────────────
    accuracy                         0.65        20
   macro avg     0.61      0.60      0.60        20
weighted avg     0.64      0.65      0.64        20
```

**Confusion Matrix:**

```
                 Predicted Non-Diabetic   Predicted Diabetic
Actual Non-Diabetic        10                    3
Actual Diabetic             4                    3
```

**Clinical interpretation:**
- **10 True Negatives** — correctly cleared as non-diabetic
- **3 True Positives** — correctly flagged as diabetic
- **4 False Negatives** — missed diabetic cases (recall gap — addressed in future roadmap via threshold tuning)
- **3 False Positives** — non-diabetic flagged as at-risk (acceptable in a screening context)

> In a clinical screening system, **recall on the Diabetic class** (sensitivity) is more important than precision — missing a diabetic patient is more costly than a false alarm. Threshold lowering (e.g., classifying as Diabetic when `prob > 0.35`) would improve recall at a small precision cost.

---

## 🔌 API Reference

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/` | Render web UI | HTML |
| `GET` | `/health` | Model load status | `{"ok": true, "model_loaded": true, "model_path": "model.joblib"}` |
| `POST` | `/predict` | Predict diabetes risk | `{"prediction": 1, "probability": 0.777}` |
| `POST` | `/reload-model` | Hot-reload model file | `{"reloaded": true, "model_path": "model.joblib"}` |
| `GET` | `/data-head` | First 5 dataset rows | `{"head": [...]}` |
| `GET` | `/data-tail` | Last 5 dataset rows | `{"tail": [...]}` |

### POST `/predict` — Request & Response

**Request** (JSON or form):
```json
{
  "Pregnancies": 2,
  "Glucose": 180,
  "BloodPressure": 74,
  "SkinThickness": 24,
  "Insulin": 94,
  "BMI": 36.0,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 45
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.777
}
```

**Error response:**
```json
{
  "error": "Model not loaded. Run training and then /reload-model or ensure model file exists."
}
```

> The API accepts both `application/json` and `multipart/form-data`. Missing fields are imputed with `np.nan` and handled by the pipeline's `SimpleImputer`.

---

## 🛠️ Tech Stack

| Library | Role |
|---------|------|
| **Python 3.10+** | Core language |
| **Flask** | Web server, REST API routing, Jinja2 templating |
| **Pandas** | CSV ingestion, DataFrame construction, zero-replacement |
| **NumPy** | NaN handling, array operations |
| **Scikit-learn** | `SimpleImputer`, `StandardScaler`, `LogisticRegression`, `Pipeline`, `train_test_split` |
| **Joblib** | Atomic model serialization / deserialization |
| **JavaScript (vanilla)** | Async `fetch()` POST to `/predict`, dynamic result rendering |

---

## 📁 Project Structure

```
diabetes_project/
│
├── app.py                    # Flask routes + model loading + REST API
├── train_model.py            # End-to-end training: load → impute → scale →
│                             #   train LR → evaluate → atomic save to model.joblib
│
├── diabetes_user.csv         # 98 clinical records, 8 features + Outcome
├── model.joblib              # Serialized sklearn Pipeline (imputer+scaler+LR)
│
├── static/
│   └── style.css             # 2-column grid layout, card container, color tokens
│
├── templates/
│   └── index.html            # Form UI + fetch() JS + color-coded result display
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mvkarthikeya07/diabetes-prediction.git
cd diabetes-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py
# Expected output:
# Training complete. Validation accuracy: 0.650
# Saved model to model.joblib (XXXXX bytes)

# 4. Launch the application
python app.py

# 5. Open in browser
# http://127.0.0.1:5000
```

### Verify the API is healthy

```bash
curl http://127.0.0.1:5000/health
# {"model_loaded": true, "model_path": "model.joblib", "ok": true}
```

### Run a prediction from the command line

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":180,"BloodPressure":74,"SkinThickness":24,"Insulin":94,"BMI":36,"DiabetesPedigreeFunction":0.627,"Age":45}'
# {"prediction": 1, "probability": 0.777}
```

---

## 🧪 Example Predictions

### 🔴 High-Risk Patient

| Feature | Value |
|---|---|
| Glucose | 180 |
| BMI | 36.0 |
| Age | 45 |
| Insulin | 94 |
| Pregnancies | 2 |
| **Prediction** | **Diabetes Positive (High Risk)** |
| **Probability** | **77.7%** |

### 🟢 Low-Risk Patient

| Feature | Value |
|---|---|
| Glucose | 95 |
| BMI | 24.0 |
| Age | 25 |
| **Prediction** | **Diabetes Negative (Low Risk)** |
| **Probability** | **12.4%** |

---

## 🔬 Engineering Highlights

| Highlight | Detail |
|---|---|
| **Zero-as-missing detection** | 5 features have medically impossible zeros replaced with `np.nan` before any computation — clinically correct preprocessing |
| **Atomic model save** | `joblib.dump()` to `.tmp` then `os.replace()` — prevents a corrupt `model.joblib` if the process crashes mid-write |
| **Multi-filename fallback** | App checks `model.joblib` → `model_pipeline.joblib` → `best_model.joblib` at startup — resilient to filename changes |
| **Hot-reload endpoint** | `POST /reload-model` re-reads the model file without restarting Flask — supports live retraining workflows |
| **Named DataFrame inference** | `build_input_dataframe()` constructs a properly-named `pd.DataFrame` — eliminates sklearn feature-name warnings |
| **Dual input format** | `/predict` accepts both `application/json` and form-encoded data — compatible with both the web UI and curl/API clients |
| **Stratified split** | `train_test_split(stratify=y)` preserves the 63/37 class ratio in both train and test sets — prevents split-induced class imbalance |

---

## ## 🏢 Internship Context

| Field | Detail |
|---|---|
| Organization | InternPe |
| Role | AI/ML Intern |
| Focus | Applied machine learning for healthcare decision-support |
| Status | ✅ Completed |

**Skills applied during this internship project:**
- Clinical data preprocessing & biomarker feature engineering (zero-as-missing detection, median imputation)
- Supervised binary classification with probability-calibrated output (not just binary labels)
- Comparative evaluation of 6 supervised algorithms with full metric reporting (Accuracy, CV, ROC-AUC)
- Sklearn Pipeline design for leakage-safe preprocessing and serialization (atomic save with `os.replace()`)
- Flask REST API integration for real-time ML inference (JSON & form input, hot-reload endpoint)
- Responsible AI practices — clear medical disclaimers, no false clinical claims, recall-prioritized screening logic

🔗 **Certificate:** [View on LinkedIn](https://www.linkedin.com/posts/m-v-karthikeya-b26a2131b_internshipcompletion-aiml-machinelearning-activity-7408819858177724416-m1Rt?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFEhlw4BT-6V0rnLIZSzBIoK7YvV2QlbHLc)

🔗 **Watch the Demo:** [LinkedIn Video](https://www.linkedin.com/posts/m-v-karthikeya-b26a2131b_internpe-machinelearning-datascience-activity-7399398252476944384-3XE6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFEhlw4BT-6V0rnLIZSzBIoK7YvV2QlbHLc)

---

## 🔮 Future Roadmap

- [ ] **Threshold Tuning** — Lower classification threshold (e.g., 0.35) to improve Diabetic recall for screening use-cases
- [ ] **SHAP Explainability** — Per-prediction feature importance waterfall charts for clinical interpretability
- [ ] **Larger Dataset** — Retrain on full Pima Indians dataset (768 records) for more reliable generalization
- [ ] **Gradient Boosting Upgrade** — Switch to GBM (ROC-AUC 0.769) when dataset is scaled up
- [ ] **Feature Distribution Plots** — Interactive histograms and box plots per biomarker, stratified by Outcome
- [ ] **User Authentication** — Login system with per-user prediction history and export
- [ ] **Cloud Deployment** — Dockerized Flask app on Render / AWS Elastic Beanstalk
- [ ] **Bootstrap UI** — Responsive mobile-friendly redesign with Material or Bootstrap 5

---

## 👤 Author

**M V Karthikeya**
Machine Learning & Python Developer

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=flat-square&logo=github)](https://github.com/Mvkarthikeya07)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

---

<div align="center">

**Clinical intelligence at the point of care — fast, interpretable, and deployable.**

*Logistic Regression · Median Imputation · StandardScaler · Flask REST API · Probability Calibration*

© 2026 M V Karthikeya

</div>
