# 📓 FinShield — Notebooks

Exploratory notebooks for the FinShield fraud detection project.

---

## Notebook Index

| Notebook                          | Purpose                                          | Status     |
|:----------------------------------|:-------------------------------------------------|:-----------|
| `01_eda.ipynb`                    | Exploratory Data Analysis on raw transaction data | 🔲 Planned |
| `02_feature_engineering.ipynb`    | Feature crafting experiments (velocity, rolling)  | 🔲 Planned |
| `03_xgboost_baseline.ipynb`       | XGBoost training, tuning, and evaluation          | 🔲 Planned |
| `04_tabnet_training.ipynb`        | PyTorch TabNet model training                     | 🔲 Planned |
| `05_model_comparison.ipynb`       | Compare XGBoost vs TabNet with MLflow logs        | 🔲 Planned |
| `06_shap_explainability.ipynb`    | SHAP feature importance analysis                  | 🔲 Planned |
| `07_rag_prototype.ipynb`          | RAG engine prototyping with ChromaDB              | 🔲 Planned |

---

## How to Run

```bash
cd F:\FinShield
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

> ⚠️ Notebooks are for experimentation only. Final code lives in `src/`.
