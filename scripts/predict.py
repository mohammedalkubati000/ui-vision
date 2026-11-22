import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
W, H, SCALE = 704, 256, 0.5
SSIM_T = 0.921

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(path):
    img = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (W, H), (243, 244, 246))
    canvas.paste(img, (0, 0))
    canvas = canvas.resize((int(W * SCALE), int(H * SCALE)), Image.BILINEAR)
    return torch.from_numpy(np.array(canvas)).permute(2, 0, 1)


def prep(x):
    return (x.float() / 255 - MEAN) / STD


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )

    def forward(self, a, b):
        f = self.backbone(torch.cat([a, b]))
        fa, fb = f.chunk(2)
        return self.head(torch.cat([(fa - fb).abs(), fa * fb], dim=1))


def compare(design_path, code_path):
    d = cv2.imread(str(design_path))
    c = cv2.imread(str(code_path))
    if c.shape != d.shape:
        c = cv2.resize(c, (d.shape[1], d.shape[0]))
    score = ssim_fn(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))

    model = SiameseNet()
    model.load_state_dict(torch.load(ROOT / "models" / "best.pt"))
    model.eval()
    with torch.no_grad():
        out = model(prep(load(design_path).unsqueeze(0)), prep(load(code_path).unsqueeze(0)))
    p_match = torch.softmax(out, dim=1)[0, 1].item()

    return {
        "ssim": round(score, 4),
        "ssim_verdict": "High Match" if score >= SSIM_T else "Visual Mismatch",
        "cnn_match_prob": round(p_match, 4),
        "cnn_verdict": "High Match" if p_match >= 0.5 else "Visual Mismatch",
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/predict.py design.png code.png")
    for k, v in compare(sys.argv[1], sys.argv[2]).items():
        print(f"{k:16s}: {v}")