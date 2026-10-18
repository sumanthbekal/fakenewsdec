# Veritas Fake News Detector

An AI-powered Fake News Detection web application built with FastAPI and Scikit-Learn. The system is trained on the **WELFake dataset** (~72,000 articles) using word n-gram **TF-IDF Vectorization** and a calibrated **Linear Support Vector Classifier (LinearSVC)**.

---

## Architecture Overview

- **Model Pipeline**: 
  - Text preprocessing & leakage removal (strips news agency wire tags like Reuters/AP/AFP and URLs).
  - TF-IDF Vectorizer with unigrams & bigrams (`max_features=50,000`, `sublinear_tf=True`).
  - `LinearSVC` calibrated with 3-fold `CalibratedClassifierCV` to yield reliable probabilities.
- **Backend**: FastAPI with Uvicorn, serving the RESTful API and static assets.
- **Frontend**: Responsive, modern dark-mode glassmorphic interface with vanilla CSS & JavaScript.

---

## Model Evaluation on WELFake Test Benchmark (N = 5,000 articles)

- **Accuracy**: 95.96%
- **Precision**: 96.32%
- **Recall**: 96.45%
- **F1-Score**: 96.39%
- **ROC-AUC**: 0.9932
- **Confusion Matrix**:
  - True Negatives (Correct FAKE): 2,105
  - False Positives: 103
  - False Negatives: 99
  - True Positives (Correct REAL): 2,693

---

## Quick Start & Running the Project

### 1. Prerequisites
- Python 3.11+ installed.

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r backend/requirements.txt
```

### 3. (Optional) Train / Re-evaluate Model
The trained model pipeline and evaluation plots are saved in `backend/model/model_pipeline.joblib` and `frontend/static/figs/evaluation_metrics.png`. To train again:
```powershell
.venv\Scripts\python backend/train.py
```

### 4. Run the Web Application
```powershell
.venv\Scripts\uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## API Endpoints

- `GET /` — Serves the interactive web interface.
- `GET /health` — Returns system health status, model load status, and benchmark evaluation metrics.
- `POST /predict` — Analyzes news text.
  - **Request Body**: `{"text": "Your news article text here (minimum 10 words)"}`
  - **Response**:
    ```json
    {
      "label": "FAKE",
      "confidence": 0.9862,
      "probabilities": {
        "FAKE": 0.9862,
        "REAL": 0.0138
      },
      "words": 54,
      "truncated": false
    }
    ```

---

## Mandatory Disclaimer
> **Important Model Disclaimer:**
> This is a statistical estimate from a model trained on the WELFake dataset only. It checks writing patterns, not facts, and can be wrong on news from other sources. Verify important claims with trusted sources.
