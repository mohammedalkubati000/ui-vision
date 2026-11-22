import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from skimage.metrics import structural_similarity as ssim_fn

from predict import ROOT, SSIM_T, SiameseNet, load, prep

app = FastAPI(title="UI-Vision API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SiameseNet()
model.load_state_dict(torch.load(ROOT / "models" / "best.pt"))
model.eval()


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


@app.post("/compare")
async def compare(design: UploadFile = File(...), code: UploadFile = File(...)):
    import io

    d_bytes, c_bytes = await design.read(), await code.read()
    d, c = decode(d_bytes), decode(c_bytes)
    if c.shape != d.shape:
        c = cv2.resize(c, (d.shape[1], d.shape[0]))

    score = ssim_fn(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))

    a = load(io.BytesIO(d_bytes)).unsqueeze(0)
    b = load(io.BytesIO(c_bytes)).unsqueeze(0)
    with torch.no_grad():
        p_match = torch.softmax(model(prep(a), prep(b)), dim=1)[0, 1].item()

    return {
        "ssim": round(float(score), 4),
        "ssim_verdict": "High Match" if score >= SSIM_T else "Visual Mismatch",
        "cnn_match_prob": round(p_match, 4),
        "cnn_verdict": "High Match" if p_match >= 0.5 else "Visual Mismatch",
    }