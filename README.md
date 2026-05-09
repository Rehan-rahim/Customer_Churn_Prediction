# 📡 Customer Churn Prediction: Hybrid ML & Generative AI Pipeline

---

## 👨‍🏫 Course Information

| Field | Details |
|---|---|
| **Professor** | Syed Jawad Shah |
| **Course** | [Principle of Data Science] |
| **Institution** | University of Missouri-Kansas City |

---

## 👥 Group Members

| # | Name |
|---|---|
| 1 | [Rehan Ali] 
| 2 | [Rasagyna Peddapalli]
| 3 | [Utkarsh Choudhary]
| 4 | [Yashwanth Nanjuti]

---

## 📌 Project Overview

This project develops an **advanced churn prediction pipeline** for the **telecommunications sector**. It employs a **hybrid approach** combining:

- 🌲 **Random Forest** (Supervised Learning) — to predict whether a customer will churn
- 🤖 **BERT** (Generative AI / NLP) — to understand the customer's psychological state through **Sentiment & Semantic Risk Profiling**

The goal is not only to predict churn, but to provide **actionable, explainable insights** about customer behavior that business teams can act upon in real time.

---

## 🏗️ Methodology — 4-Phase Pipeline

### Phase 1 — Data Ingestion & Inspection
- Loaded the **IBM Telco Customer Churn Dataset** (7,043 records)
- Performed **schema validation** and **statistical profiling** to understand data distribution and quality

### Phase 2 — Preprocessing & Exploratory Data Analysis (EDA)
- Cleaned the `TotalCharges` column and removed null/missing values
- Applied **One-Hot Encoding** and **Binary Encoding** to categorical features
- Expanded feature space to **31 dimensions** for model readiness
- Generated visualizations to surface churn patterns and class imbalance

### Phase 3 — Supervised Learning
- Trained a **Random Forest Classifier** with **100 estimators**
- Evaluated against a **Decision Tree baseline** to quantify ensemble gains
- Used train/test split with stratified sampling to preserve class ratios

### Phase 4 — Generative AI Augmentation (BERT)
- Constructed natural-language **customer profile sentences** from structured features
- Fed profiles into `bert-base-uncased` to extract **768-dimensional semantic embeddings**
- Combined embeddings with rule-based logic to generate **Semantic Risk Tiers**

---

## 📊 Performance & Results

> Evaluated on test set: **n = 1,407 samples**

| Metric | Score |
|---|---|
| **Overall Accuracy** | 78.75% |
| **ROC-AUC Score** | 0.8184 |
| **Non-Churner F1-Score** | 0.86 |
| **Churner Recall** | 0.48 *(limited by class imbalance)* |

### Semantic Risk Profiling — BERT-Derived Customer Tiers

| Risk Tier | Profile Description | Churn Risk |
|---|---|---|
| 🟢 **Neutral** | Long-term contracts, low monthly charges | Low |
| 🟡 **Negative** | Month-to-month subscribers, moderate charges | Medium |
| 🔴 **Frustrated** | Month-to-month + high charges + no support services | **Highest** |

> The **Frustrated** tier represents the most actionable segment for targeted retention campaigns.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Core Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Random Forest, Decision Tree) |
| **Deep Learning / NLP** | Hugging Face Transformers (BERT), PyTorch |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Deployment** | Gradio (Interactive Assessment UI), Streamlit (Business Dashboard) |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telco_customer_churn.csv       # IBM Telco Dataset (7,043 records)
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # Phase 1: Loading & Inspection
│   ├── 02_preprocessing_eda.ipynb     # Phase 2: Cleaning & EDA
│   ├── 03_random_forest.ipynb         # Phase 3: Supervised Learning
│   └── 04_bert_augmentation.ipynb     # Phase 4: BERT Embeddings & Risk Tiers
│
├── app/
│   ├── gradio_app.py                  # Interactive Customer Assessment Tool
│   └── streamlit_dashboard.py         # Business Intelligence Dashboard
│
├── models/
│   ├── random_forest_model.pkl        # Saved RF Classifier
│   └── bert_embeddings/               # Cached BERT Embedding Outputs
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone (https://github.com/Rehan-rahim/Customer_Churn_Prediction.git)

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Future Roadmap

- [ ] **Class Imbalance Handling** — Apply SMOTE or `class_weight='balanced'` to improve Churner Recall beyond 0.48
- [ ] **Advanced Ensemble Models** — Benchmark XGBoost and LightGBM against the current Random Forest baseline
- [ ] **Real-Time CRM Integration** — Deploy the pipeline via **FastAPI** for live integration with CRM systems
- [ ] **Fine-Tuned BERT** — Fine-tune `bert-base-uncased` on domain-specific telecom churn text for richer embeddings
- [ ] **Explainability Layer** — Integrate SHAP values for feature-level model interpretability

---
---

> *"Predicting churn is not just a machine learning problem — it's a human understanding problem."*
