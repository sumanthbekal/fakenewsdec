import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="WELFake News Detector API",
    description="Real vs Fake News detector using TF-IDF and Calibrated LinearSVC trained on the WELFake dataset.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "model_pipeline.joblib")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global model state
pipeline_bundle = None

def load_model():
    global pipeline_bundle
    if os.path.exists(MODEL_PATH):
        try:
            pipeline_bundle = joblib.load(MODEL_PATH)
            print("Model pipeline successfully loaded into memory.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            pipeline_bundle = None
    else:
        print(f"Model file not found at {MODEL_PATH}")

@app.on_event("startup")
def startup_event():
    load_model()

# Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., description="The news article text or headline to analyze")

class Probabilities(BaseModel):
    FAKE: float
    REAL: float

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Probabilities
    words: int
    truncated: bool = False

def clean_input_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"^.{0,60}\((reuters|ap|afp)\)\s*[-–—]?\s*", "", t, flags=re.I)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

@app.get("/")
def serve_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "WELFake News Detector API is operational. Place index.html in frontend/templates."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if pipeline_bundle is not None else "degraded",
        "model_loaded": pipeline_bundle is not None,
        "metrics": pipeline_bundle.get("metrics") if pipeline_bundle else None,
        "disclaimer": "This is a statistical estimate from a model trained on the WELFake dataset only. It checks writing patterns, not facts, and can be wrong on news from other sources. Verify important claims with trusted sources."
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if pipeline_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please ensure training has completed.")

    raw_text = request.text
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty. Please provide the news headline or article body."
        )

    # Word count check
    words_list = raw_text.strip().split()
    word_count = len(words_list)
    if word_count < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Input text has only {word_count} word(s). Please provide at least 10 words for a meaningful linguistic analysis."
        )

    # Clean input
    cleaned = clean_input_text(raw_text)
    if not cleaned or len(cleaned.split()) < 3:
        cleaned = raw_text.strip()

    vectorizer = pipeline_bundle["vectorizer"]
    model = pipeline_bundle["model"]

    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0] # [P(FAKE), P(REAL)]
    
    fake_prob = float(proba[0])
    real_prob = float(proba[1])

    if real_prob >= fake_prob:
        predicted_label = "REAL"
        confidence = real_prob
    else:
        predicted_label = "FAKE"
        confidence = fake_prob

    # Round values nicely
    confidence = round(confidence, 4)
    fake_prob = round(fake_prob, 4)
    real_prob = round(real_prob, 4)

    return PredictResponse(
        label=predicted_label,
        confidence=confidence,
        probabilities=Probabilities(
            FAKE=fake_prob,
            REAL=real_prob,
        ),
        words=word_count,
        truncated=False
    )
