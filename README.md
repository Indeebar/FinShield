# 🏦 FinShield — Financial Fraud & Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![MLflow](https://img.shields.io/badge/MLflow-2.x-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)
![License](https://img.shields.io/badge/license-MIT-green)

> An end-to-end fraud detection platform serving real-time predictions via FastAPI, with RAG-powered explainability, MLflow experiment tracking, and CI/CD deployment to AWS EC2 / Render.

---

## 🎯 What It Does

- **Detects fraud** in financial transactions using XGBoost and PyTorch TabNet models
- **Explains decisions** using a RAG engine (sentence-transformers + ChromaDB) over a fraud case knowledge base
- **Tracks experiments** with MLflow (metrics, artifacts, model registry)
- **Ships to production** via Docker + GitHub Actions CI/CD

---

## 🏗 Architecture

```
FastAPI Gateway
    ├── POST /predict       → XGBoost / TabNet inference + SHAP
    ├── GET  /explain/{id}  → RAG explainability engine
    └── GET  /health        → system health

Feature Engineering → PostgreSQL → Model Inference → SHAP → RAG → Response
                                        ↕
                               MLflow Tracking Server
```

---

## 🛠 Tech Stack

| Layer          | Technology                        |
|:---------------|:----------------------------------|
| ML             | XGBoost, scikit-learn, SHAP       |
| Deep Learning  | PyTorch + TabNet                  |
| NLP / RAG      | sentence-transformers, ChromaDB   |
| MLOps          | MLflow (tracking + registry)      |
| API            | FastAPI + Uvicorn                 |
| Database       | PostgreSQL + SQLAlchemy           |
| CI/CD          | GitHub Actions                    |
| Containers     | Docker + Docker Compose           |
| Cloud          | AWS EC2 / Render                  |

---

## 📁 Project Structure

```
finshield/
├── data/
│   ├── raw/                    # Raw dataset (gitignored)
│   └── processed/              # Processed features
├── src/
│   ├── features/               # Feature engineering
│   ├── models/                 # XGBoost + TabNet training
│   ├── serving/                # FastAPI application
│   ├── rag/                    # RAG explainability engine
│   └── mlops/                  # MLflow utilities
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/                  # EDA and experiments
├── .github/workflows/          # CI/CD pipelines
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Quick Start

### Local Development

```bash
# Clone & setup
git clone https://github.com/Indeebar/FinShield.git
cd finshield
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start services
docker-compose up -d

# Run API
uvicorn src.serving.app:app --reload
```

### API Endpoints

```bash
# Predict fraud
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500.0, "merchant": "Online Store", "hour": 2}'

# Get explanation
curl http://localhost:8000/explain/txn_12345

# Health check
curl http://localhost:8000/health
```

---

## 📊 Model Performance

| Model   | AUC-ROC | F1 (Fraud) | Precision | Recall |
|:--------|:--------|:-----------|:----------|:-------|
| XGBoost | TBD     | TBD        | TBD       | TBD    |
| TabNet  | TBD     | TBD        | TBD       | TBD    |

*Updated after training on Kaggle Credit Card Fraud dataset (284,807 transactions)*

---

## 🔄 CI/CD Pipeline

```
push to main
    → Lint (ruff)
    → Unit Tests (pytest)
    → Docker Build
    → Push to Docker Hub
    → Deploy to Render / AWS EC2
```

---

## 📄 License

MIT © [Indeebar](https://github.com/Indeebar)
