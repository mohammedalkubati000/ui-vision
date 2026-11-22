import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "dataset" / "samples"
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


model = SiameseNet()
model.load_state_dict(torch.load(ROOT / "models" / "best.pt"))
model.eval()

with open(ROOT / "dataset" / "baseline_results.csv", encoding="utf-8") as f:
    ssim = {r["id"]: float(r["ssim"]) for r in csv.DictReader(f)}

with open(ROOT / "dataset" / "labels.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["component"] == "form"]

A = torch.stack([load(SAMPLES / f"{r['id']}_design.png") for r in rows])
B = torch.stack([load(SAMPLES / f"{r['id']}_code.png") for r in rows])
with torch.no_grad():
    pred = torch.cat([model(prep(A[i:i + 32]), prep(B[i:i + 32])).argmax(1)
                      for i in range(0, len(A), 32)])

total, cnn_miss, ssim_miss = Counter(), Counter(), Counter()
print("== mismatch فاتت الـ CNN ==")
for r, p in zip(rows, pred):
    if r["label"] != "mismatch":
        continue
    change = json.loads(r["change"])
    cnn_missed = p.item() == 1
    ssim_missed = ssim[r["id"]] >= SSIM_T
    for prop in change:
        total[prop] += 1
        cnn_miss[prop] += cnn_missed
        ssim_miss[prop] += ssim_missed
    if cnn_missed:
        print(f"{r['id']}  ssim={ssim[r['id']]:.3f}  {r['change']}")

print("\n== حسب الخاصية: (عدد الـ mismatch اللي فيها) / فاتت الـ CNN / فاتت SSIM ==")
for prop in sorted(total):
    print(f"{prop:12s} total={total[prop]:2d}  CNN missed={cnn_miss[prop]:2d}  SSIM missed={ssim_miss[prop]:2d}")