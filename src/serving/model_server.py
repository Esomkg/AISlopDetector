"""Model inference server for AISlopDetector.

Exposes a FastAPI endpoint for single-image classification.
Run with: uvicorn src.serving.model_server:app --host 0.0.0.0 --port 8080
"""

import io
import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from src.models.classifier import AISlopClassifier
from src.data.transforms import get_val_transforms


app = FastAPI(
    title="AISlopDetector",
    description="AI-generated image detection API",
    version="0.1.0",
)

MODEL = None
DEVICE = None
TRANSFORM = None
CLASS_LABELS = {0: "REAL", 1: "FAKE"}
PREDICTION_COUNT = Counter("aislop_predictions_total", "Total predictions", ["class"])
ERROR_COUNT = Counter("aislop_errors_total", "Total prediction errors")
LOW_CONFIDENCE_THRESHOLD = 0.6


class ModelService:
    """Manages model lifecycle for inference."""
    
    def __init__(self, checkpoint_path: str, backbone: str = "efficientnet_b3", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AISlopClassifier(num_classes=2, backbone_name=backbone, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint["model_state_dict"]

        # Remap keys from Kaggle notebook format (backbone.classifier.x → head.x)
        remapped = {}
        for key, value in state_dict.items():
            new_key = key.replace("backbone.classifier.", "head.")
            remapped[new_key] = value

        self.model.load_state_dict(remapped)
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_val_transforms(image_size=224)
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Run inference on a single PIL image.
        
        Returns dict with:
            - predicted_class: "REAL" or "FAKE"
            - confidence: float (probability of predicted class)
            - probabilities: {"REAL": float, "FAKE": float}
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted = int(np.argmax(probs))
        
        return {
            "predicted_class": CLASS_LABELS[predicted],
            "confidence": round(float(probs[predicted]), 4),
            "probabilities": {
                "REAL": round(float(probs[0]), 4),
                "FAKE": round(float(probs[1]), 4),
            },
            "needs_review": float(probs[predicted]) < LOW_CONFIDENCE_THRESHOLD,
        }
    
    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """Run inference on a batch of PIL images."""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        outputs = self.model(tensors)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted = np.argmax(probs, axis=1)
        
        results = []
        for i in range(len(images)):
            results.append({
                "predicted_class": CLASS_LABELS[int(predicted[i])],
                "confidence": round(float(probs[i][int(predicted[i])]), 4),
                "probabilities": {
                    "REAL": round(float(probs[i][0]), 4),
                    "FAKE": round(float(probs[i][1]), 4),
                },
                "needs_review": float(probs[i][int(predicted[i])]) < LOW_CONFIDENCE_THRESHOLD,
            })
        return results


def init_model(checkpoint_path: str, backbone: str = "efficientnet_b3"):
    """Initialize the global model service."""
    global MODEL, DEVICE, TRANSFORM
    MODEL = ModelService(checkpoint_path, backbone=backbone)
    DEVICE = MODEL.device
    TRANSFORM = MODEL.transform


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AISlopDetector</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #0d1117; color: #c9d1d9; min-height: 100vh;
  display: flex; justify-content: center; padding: 40px 20px;
}
.container { max-width: 720px; width: 100%; }
h1 { font-size: 1.8rem; font-weight: 600; margin-bottom: 4px; }
.subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 32px; }
.upload-zone {
  border: 2px dashed #30363d; border-radius: 12px;
  padding: 48px 24px; text-align: center; cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  margin-bottom: 24px;
}
.upload-zone:hover, .upload-zone.dragover {
  border-color: #58a6ff; background: rgba(88,166,255,0.06);
}
.upload-zone p { color: #8b949e; font-size: 0.95rem; }
.upload-zone .icon { font-size: 2.5rem; margin-bottom: 12px; display: block; }
.preview-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 12px; margin-bottom: 24px; }
.preview-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  overflow: hidden; position: relative;
}
.preview-card img { width: 100%; height: 140px; object-fit: cover; display: block; }
.preview-card .remove {
  position: absolute; top: 6px; right: 6px;
  background: rgba(0,0,0,0.6); color: #fff; border: none;
  border-radius: 50%; width: 24px; height: 24px;
  cursor: pointer; font-size: 14px; line-height: 24px; text-align: center;
}
.actions { display: flex; gap: 12px; margin-bottom: 32px; flex-wrap: wrap; }
.btn {
  padding: 10px 24px; border-radius: 8px; border: none;
  font-size: 0.95rem; font-weight: 600; cursor: pointer;
  transition: opacity 0.2s;
}
.btn:hover { opacity: 0.85; }
.btn-primary { background: #238636; color: #fff; }
.btn-secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.results { display: flex; flex-direction: column; gap: 16px; }
.result-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 12px;
  padding: 20px; display: flex; align-items: center; gap: 16px;
}
.result-card.real { border-left: 4px solid #238636; }
.result-card.fake { border-left: 4px solid #da3633; }
.result-card img { width: 100px; height: 100px; object-fit: cover; border-radius: 8px; }
.result-info { flex: 1; }
.result-label {
  font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;
}
.result-label.real { color: #3fb950; }
.result-label.fake { color: #f85149; }
.result-filename { color: #8b949e; font-size: 0.8rem; margin-bottom: 8px; }
.bar-group { margin-bottom: 6px; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 2px; }
.bar-outer { background: #21262d; border-radius: 4px; height: 8px; overflow: hidden; }
.bar-inner { height: 100%; border-radius: 4px; transition: width 0.4s; }
.bar-inner.real-bar { background: #238636; }
.bar-inner.fake-bar { background: #da3633; }
.loading { text-align: center; color: #8b949e; padding: 20px; }
.error { background: rgba(218,54,51,0.1); border: 1px solid #da3633; border-radius: 8px; padding: 12px 16px; color: #f85149; margin-bottom: 16px; }
.status-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
  background: #21262d; margin-left: 8px;
}
.status-badge.on { background: rgba(63,185,80,0.15); color: #3fb950; }
.status-badge.off { background: rgba(248,81,73,0.15); color: #f85149; }
.stat { font-size: 0.75rem; color: #8b949e; margin-top: 8px; }
.file-input { display: none; }
@media (max-width: 500px) {
  .result-card { flex-direction: column; align-items: flex-start; }
}
</style>
</head>
<body>
<div class="container">
  <h1>AISlopDetector</h1>
  <p class="subtitle">AI-generated image detection<span class="status-badge" id="statusBadge">checking...</span></p>

  <div class="upload-zone" id="dropZone">
    <span class="icon">&#x1f4c1;</span>
    <p>Drop images here or click to browse</p>
    <p style="font-size:0.8rem;margin-top:8px">Supports JPG, PNG, WEBP — single or batch</p>
  </div>
  <input type="file" id="fileInput" class="file-input" accept="image/*" multiple>

  <div class="preview-grid" id="previewGrid"></div>

  <div class="actions" id="actions" style="display:none">
    <button class="btn btn-primary" id="predictBtn" onclick="runPrediction()">Detect</button>
    <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
  </div>

  <div class="results" id="results"></div>

  <p class="stat" style="margin-top:48px;text-align:center">EfficientNet-B3 &bull; CLIP ViT-B/32 embeddings &bull; MMD drift monitoring</p>
</div>

<script>
const dropZone = document.getElementById('dropZone')
const fileInput = document.getElementById('fileInput')
const previewGrid = document.getElementById('previewGrid')
const actions = document.getElementById('actions')
const results = document.getElementById('results')
const statusBadge = document.getElementById('statusBadge')
let files = []

async function checkHealth() {
  try { const r = await fetch('/health'); const d = await r.json()
    statusBadge.className = 'status-badge ' + (d.status==='healthy'?'on':'off')
    statusBadge.textContent = d.status==='healthy' ? 'ready' : 'offline'
  } catch(e) { statusBadge.className = 'status-badge off'; statusBadge.textContent = 'offline' }
}
checkHealth(); setInterval(checkHealth, 30000)

dropZone.addEventListener('click', () => fileInput.click())
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover') })
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'))
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragover'); addFiles(Array.from(e.dataTransfer.files)) })
fileInput.addEventListener('change', () => { addFiles(Array.from(fileInput.files)); fileInput.value = '' })

function addFiles(newFiles) {
  newFiles.forEach(f => { if (f.type.startsWith('image/') && !files.find(x => x.name===f.name)) files.push(f) })
  renderPreviews()
}

function renderPreviews() {
  previewGrid.innerHTML = ''; actions.style.display = files.length ? 'flex' : 'none'
  files.forEach((f, i) => {
    const card = document.createElement('div'); card.className = 'preview-card'
    const img = document.createElement('img'); img.src = URL.createObjectURL(f)
    const btn = document.createElement('button'); btn.className = 'remove'; btn.textContent = 'x'
    btn.onclick = () => { files.splice(i,1); renderPreviews() }
    card.appendChild(img); card.appendChild(btn); previewGrid.appendChild(card)
  })
}

function clearAll() { files = []; results.innerHTML = ''; renderPreviews() }

async function runPrediction() {
  if (!files.length) return
  results.innerHTML = '<div class="loading">Running detection...</div>'
  const predictBtn = document.getElementById('predictBtn')
  predictBtn.disabled = true; predictBtn.textContent = 'Analyzing...'

  try {
    const formData = new FormData()
    files.forEach(f => formData.append('files', f))
    const res = await fetch('/predict/batch', { method: 'POST', body: formData })
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Prediction failed') }
    const predictions = await res.json()
    results.innerHTML = predictions.map((p, i) => `
      <div class="result-card ${p.predicted_class.toLowerCase()}">
        <img src="${URL.createObjectURL(files[i])}" alt="preview">
        <div class="result-info">
          <div class="result-label ${p.predicted_class.toLowerCase()}">${p.predicted_class}</div>
          <div class="result-filename">${p.filename || files[i].name}</div>
          <div class="bar-group">
            <div class="bar-label"><span>REAL</span><span>${(p.probabilities.REAL*100).toFixed(1)}%</span></div>
            <div class="bar-outer"><div class="bar-inner real-bar" style="width:${(p.probabilities.REAL*100).toFixed(0)}%"></div></div>
          </div>
          <div class="bar-group">
            <div class="bar-label"><span>FAKE</span><span>${(p.probabilities.FAKE*100).toFixed(1)}%</span></div>
            <div class="bar-outer"><div class="bar-inner fake-bar" style="width:${(p.probabilities.FAKE*100).toFixed(0)}%"></div></div>
          </div>
        </div>
      </div>
    `).join('')
  } catch(e) {
    results.innerHTML = '<div class="error">' + e.message + '</div>'
  } finally {
    predictBtn.disabled = false; predictBtn.textContent = 'Detect'
  }
}
</script>
</body>
</html>"""
async def health():
    """Health check endpoint."""
    if MODEL is None:
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return {"status": "healthy", "device": str(DEVICE)}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify a single uploaded image as REAL or FAKE.
    
    Accepts: multipart/form-data with an image file.
    Returns JSON with predicted_class, confidence, and probabilities.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    result = MODEL.predict(image)
    PREDICTION_COUNT.labels(result["predicted_class"]).inc()
    result["filename"] = file.filename
    return JSONResponse(content=result)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify multiple images.
    
    Accepts: multipart/form-data with multiple image files.
    Returns JSON array of predictions.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    images = []
    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(img)
        except Exception:
            ERROR_COUNT.inc()
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
    
    results = MODEL.predict_batch(images)
    for r in results:
        PREDICTION_COUNT.labels(r["predicted_class"]).inc()
    for i, (result, file) in enumerate(zip(results, files)):
        result["filename"] = file.filename
    return JSONResponse(content=results)


@app.post("/v1/models/aislop:predict")
async def kserve_predict(payload: dict):
    """KServe V2-compatible prediction endpoint.
    
    Accepts JSON: {"inputs": [{"name": "image", "data": [base64_encoded_image], ...}]}
    Returns JSON: {"predictions": [...]}
    """
    import base64
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        instances = payload.get("instances", [])
        results = []
        for instance in instances:
            img_data = base64.b64decode(instance["image"])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            result = MODEL.predict(image)
            PREDICTION_COUNT.labels(result["predicted_class"]).inc()
            results.append(result)
        return JSONResponse(content={"predictions": results})
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/review")
async def predict_with_review(file: UploadFile = File(...)):
    """Classify an image and route low-confidence predictions to review."""
    import requests

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = MODEL.predict(image)

    if result["needs_review"]:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            from src.data.active_learning import send_to_review
            send_to_review([tmp_path])
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            result["review_error"] = str(e)

    result["filename"] = file.filename
    return JSONResponse(content=result)


def main():
    parser = argparse.ArgumentParser(description="AISlopDetector inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--backbone", default="efficientnet_b3", help="Model backbone")
    args = parser.parse_args()
    
    init_model(args.checkpoint, args.backbone)
    print(f"Model loaded from {args.checkpoint}")
    print(f"Device: {DEVICE}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
