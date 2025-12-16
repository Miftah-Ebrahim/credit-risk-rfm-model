
# Credit Risk Model (RFM & Alternative Data)

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square) ![Python 3.9](https://img.shields.io/badge/Python-3.9-blue?style=flat-square) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=flat-square) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square)

An end-to-end Machine Learning pipeline for assessing creditworthiness using alternative data. Uses **Recency, Frequency, Monetary (RFM)** analysis and **Unsupervised Learning** (K-Means) to create a proxy target variable for supervised training.

---

## 🏗️ Architecture

1.  **Ingestion**: Raw transaction logs.
2.  **Feature Store**: `src/features.py` extracts temporal profiles, categorical modes (`ChannelId`), and RFM stats.
3.  **Labeling**: `src/features.py` uses **3-Cluster K-Means** to identify high-risk behaviors (High Recency, Low Activity).
4.  **Training**: `src/train.py` builds an **sklearn Pipeline** (Scaler + Encoder + Classifier) tracked via **MLflow**.
5.  **Inference**: `src/predict.py` loads the production model via MLflow Native Loading.
6.  **API**: `api/main.py` serves predictions via FastAPI.

---

## 🚀 Quick Start

### Docker (Recommended)
```bash
docker-compose -f docker/docker-compose.yml up --build
# API: http://localhost:8000/docs
```

### Local Development
```bash
pip install -r requirements.txt
python -m src.data_processing  # ETL
python -m src.train            # Train & Register
uvicorn api.main:app --reload  # Serve
```

---

## 🛠️ Components

| Component | Description |
| :--- | :--- |
| **`src/features.py`** | Pure functional feature logic (RFM, WoE, Temporal). |
| **`src/data_processing.py`** | Minimal orchestrator for Data cleaning & preparation. |
| **`src/train.py`** | Pipeline Construction, GridSearch, MLflow Logging. |
| **`src/predict.py`** | Model Loader & Inference Wrapper. |
| **`api/`** | FastAPI endpoints and strict Pydantic schemas. |

---

## 📊 Results

| Model | AUC | F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **1.000** | **1.000** | 🏆 Production |
| **Logistic Regression** | 0.999 | 0.978 | Challenger |

*Note: High scores reflect the nature of proxy modeling where ground truth is derived from features.*

---

## ⚖️ License
MIT License.
