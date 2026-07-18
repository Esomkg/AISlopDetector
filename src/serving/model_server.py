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
<title>AI Slop Detector</title>
<style>
:root {
  --bg: #000;
  --surface: #0a0a0a;
  --border: #1a2e1a;
  --text: #e8e8e8;
  --muted: #6b8b6b;
  --green: #00e676;
  --red: #ff3d3d;
  --radius: 10px;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #000;
  color: var(--text);
  min-height: 100vh;
  line-height: 1.6;
  position: relative;
  overflow-x: hidden;
}

body::before {
  content: '';
  position: fixed;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background:
    radial-gradient(ellipse at 30% 20%, rgba(0,230,118,0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 70% 80%, rgba(0,230,118,0.04) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.02) 0%, transparent 70%);
  animation: bgShift 20s ease-in-out infinite;
  z-index: 0;
  pointer-events: none;
}

@keyframes bgShift {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  33% { transform: translate(1%, -1%) rotate(0.5deg); }
  66% { transform: translate(-1%, 1%) rotate(-0.5deg); }
}

.header {
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 10;
  background: rgba(0,0,0,0.85);
  backdrop-filter: blur(12px);
}

.header-left {
  display: flex; align-items: center; gap: 10px;
}

.header h1 {
  font-size: 1.05rem; font-weight: 600;
  letter-spacing: 0.02em;
  color: #fff;
}

.status {
  font-size: 0.7rem; padding: 4px 10px; border-radius: 20px;
  font-weight: 500; letter-spacing: 0.03em; text-transform: uppercase;
}
.status.ready { background: rgba(0,230,118,0.12); color: var(--green); }
.status.offline { background: rgba(255,61,61,0.12); color: var(--red); }

.container { max-width: 780px; margin: 0 auto; padding: 36px 24px; position: relative; z-index: 1; }

.upload-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 24px;
  transition: border-color 0.3s;
}
.upload-section:focus-within { border-color: var(--green); }

.upload-zone {
  padding: 52px 24px; text-align: center; cursor: pointer;
  transition: background 0.2s;
}

.upload-zone:hover, .upload-zone.dragover {
  background: rgba(0,230,118,0.03);
}

.upload-icon {
  width: 56px; height: 56px; margin: 0 auto 16px;
  border-radius: 50%;
  background: rgba(0,230,118,0.06);
  border: 1px solid rgba(0,230,118,0.15);
  display: flex; align-items: center; justify-content: center;
  transition: border-color 0.2s, background 0.2s;
}
.upload-zone:hover .upload-icon, .upload-zone.dragover .upload-icon {
  border-color: var(--green);
  background: rgba(0,230,118,0.1);
}

.upload-icon svg {
  width: 24px; height: 24px;
  stroke: var(--green);
}

.upload-zone p { color: var(--muted); font-size: 0.9rem; }
.upload-zone p span { color: var(--green); cursor: pointer; font-weight: 500; }

.preview-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 8px; padding: 16px;
  border-top: 1px solid var(--border);
  display: none;
}
.preview-grid.has-files { display: grid; }

.preview-card {
  position: relative; border-radius: 6px; overflow: hidden;
  aspect-ratio: 1;
}
.preview-card img { width: 100%; height: 100%; object-fit: cover; display: block; }
.preview-card .remove {
  position: absolute; top: 4px; right: 4px;
  width: 20px; height: 20px; border-radius: 50%;
  background: rgba(0,0,0,0.8); color: #fff; border: none;
  cursor: pointer; font-size: 12px; display: flex;
  align-items: center; justify-content: center;
  opacity: 0; transition: opacity 0.15s;
}
.preview-card:hover .remove { opacity: 1; }

.actions {
  display: flex; gap: 10px; padding: 0 16px 16px;
  display: none;
}
.actions.visible { display: flex; }

.btn {
  padding: 10px 20px; border-radius: 8px; border: none;
  font-size: 0.85rem; font-weight: 600; cursor: pointer;
  transition: all 0.15s; font-family: inherit;
}
.btn-primary { background: var(--green); color: #000; }
.btn-primary:hover { background: #00c853; }
.btn-secondary {
  background: transparent; color: var(--muted);
  border: 1px solid var(--border);
}
.btn-secondary:hover { color: var(--text); border-color: var(--muted); }
.btn:disabled { opacity: 0.3; pointer-events: none; }

.results { display: flex; flex-direction: column; gap: 12px; }

.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  display: flex;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.result-card.real { border-left: 3px solid var(--green); }
.result-card.fake { border-left: 3px solid var(--red); }

.result-preview {
  width: 120px; min-width: 120px;
  background: #111;
  display: flex; align-items: center; justify-content: center;
}

.result-preview img { width: 100%; height: 120px; object-fit: cover; }

.result-body { padding: 16px 20px; flex: 1; min-width: 0; }

.result-verdict {
  font-size: 1.1rem; font-weight: 700;
  letter-spacing: 0.02em; margin-bottom: 2px;
}
.result-verdict.real { color: var(--green); }
.result-verdict.fake { color: var(--red); }

.result-file {
  font-size: 0.75rem; color: var(--muted);
  margin-bottom: 12px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.gauge-row { display: flex; gap: 12px; }
.gauge { flex: 1; }

.gauge-header {
  display: flex; justify-content: space-between;
  font-size: 0.72rem; margin-bottom: 4px;
  color: var(--muted);
}

.gauge-track {
  height: 5px; background: rgba(255,255,255,0.06);
  border-radius: 3px; overflow: hidden;
}

.gauge-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.5s ease;
}
.gauge-fill.real-bar { background: var(--green); }
.gauge-fill.fake-bar { background: var(--red); }

.gauge-value { font-weight: 600; font-variant-numeric: tabular-nums; }

.review-badge {
  display: inline-block; font-size: 0.68rem; padding: 2px 8px;
  border-radius: 12px; margin-top: 8px;
  background: rgba(255,193,7,0.1); color: #ffc107;
}

.loading {
  text-align: center; padding: 40px; color: var(--muted);
}

.error {
  background: rgba(255,61,61,0.06);
  border: 1px solid rgba(255,61,61,0.2);
  border-radius: var(--radius); padding: 14px 18px;
  color: var(--red); font-size: 0.85rem;
}

.file-input { display: none; }

.footer {
  text-align: center; padding: 40px 24px;
  color: var(--muted); font-size: 0.72rem;
  position: relative; z-index: 1;
}

@media (max-width: 600px) {
  .result-card { flex-direction: column; }
  .result-preview { width: 100%; min-width: 100%; }
  .result-preview img { height: 200px; }
  .header { padding: 12px 16px; }
  .container { padding: 24px 16px; }
}
</style>
</head>
<body>

<header class="header">
  <div class="header-left">
    <h1>AI Slop Detector</h1>
  </div>
  <span class="status" id="statusBadge">checking</span>
</header>

<div class="container">

  <div class="upload-section">
    <div class="upload-zone" id="dropZone">
      <div class="upload-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <p>Drop images here or <span>click to browse</span></p>
      <p style="font-size:0.75rem;margin-top:6px">JPG, PNG, WEBP — single or batch</p>
    </div>
    <input type="file" id="fileInput" class="file-input" accept="image/*" multiple>
    <div class="preview-grid" id="previewGrid"></div>
    <div class="actions" id="actions">
      <button class="btn btn-primary" id="predictBtn" onclick="runPrediction()">Run Detection</button>
      <button class="btn btn-secondary" onclick="clearAll()">Clear All</button>
    </div>
  </div>

  <div class="results" id="results"></div>

</div>

<footer class="footer">
  EfficientNet-B3 &bull; 98.4% accuracy &bull; MMD drift monitoring &bull; Self-healing detection
</footer>

<script>
const dropZone = document.getElementById('dropZone')
const fileInput = document.getElementById('fileInput')
const previewGrid = document.getElementById('previewGrid')
const actions = document.getElementById('actions')
const results = document.getElementById('results')
const statusBadge = document.getElementById('statusBadge')
let files = []

async function checkHealth() {
  try {
    const r = await fetch('/health')
    if (r.ok) {
      statusBadge.className = 'status ready'
      statusBadge.textContent = 'Ready'
    } else { throw new Error() }
  } catch(e) {
    statusBadge.className = 'status offline'
    statusBadge.textContent = 'Offline'
  }
}
checkHealth(); setInterval(checkHealth, 30000)

dropZone.addEventListener('click', () => fileInput.click())
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover') })
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'))
dropZone.addEventListener('drop', e => {
  e.preventDefault()
  dropZone.classList.remove('dragover')
  addFiles(Array.from(e.dataTransfer.files))
})
fileInput.addEventListener('change', () => {
  addFiles(Array.from(fileInput.files))
  fileInput.value = ''
})

function addFiles(newFiles) {
  newFiles.forEach(f => {
    if (f.type.startsWith('image/') && !files.find(x => x.name === f.name)) files.push(f)
  })
  renderPreviews()
}

function renderPreviews() {
  previewGrid.innerHTML = ''
  files.forEach((f, i) => {
    const card = document.createElement('div'); card.className = 'preview-card'
    const img = document.createElement('img'); img.src = URL.createObjectURL(f)
    const btn = document.createElement('button'); btn.className = 'remove'; btn.textContent = 'x'
    btn.onclick = () => { files.splice(i, 1); renderPreviews() }
    card.appendChild(img); card.appendChild(btn); previewGrid.appendChild(card)
  })
  const has = files.length > 0
  previewGrid.classList.toggle('has-files', has)
  actions.classList.toggle('visible', has)
}

function clearAll() {
  files = []
  results.innerHTML = ''
  renderPreviews()
}

async function runPrediction() {
  if (!files.length) return
  results.innerHTML = '<div class="loading">Running detection</div>'
  const predictBtn = document.getElementById('predictBtn')
  predictBtn.disabled = true
  predictBtn.textContent = 'Analyzing...'

  try {
    const formData = new FormData()
    files.forEach(f => formData.append('files', f))
    const res = await fetch('/predict/batch', { method: 'POST', body: formData })
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Prediction failed') }
    const predictions = await res.json()

    results.innerHTML = predictions.map((p, i) => `
      <div class="result-card ${p.predicted_class.toLowerCase()}">
        <div class="result-preview">
          <img src="${URL.createObjectURL(files[i])}" alt="preview">
        </div>
        <div class="result-body">
          <div class="result-verdict ${p.predicted_class.toLowerCase()}">${p.predicted_class}</div>
          <div class="result-file">${p.filename || files[i].name}</div>
          <div class="gauge-row">
            <div class="gauge">
              <div class="gauge-header"><span>Real</span><span class="gauge-value">${(p.probabilities.REAL*100).toFixed(1)}%</span></div>
              <div class="gauge-track"><div class="gauge-fill real-bar" style="width:${(p.probabilities.REAL*100).toFixed(0)}%"></div></div>
            </div>
            <div class="gauge">
              <div class="gauge-header"><span>Fake</span><span class="gauge-value">${(p.probabilities.FAKE*100).toFixed(1)}%</span></div>
              <div class="gauge-track"><div class="gauge-fill fake-bar" style="width:${(p.probabilities.FAKE*100).toFixed(0)}%"></div></div>
            </div>
          </div>
          ${p.needs_review ? '<div class="review-badge">Needs review</div>' : ''}
        </div>
      </div>
    `).join('')
  } catch(e) {
    results.innerHTML = '<div class="error">' + e.message + '</div>'
  } finally {
    predictBtn.disabled = false
    predictBtn.textContent = 'Run Detection'
  }
}
</script>
</body>
</html>"""
@app.get("/health")
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
