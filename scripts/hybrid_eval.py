import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "dataset" / "samples"
W, H, SCALE = 704, 256, 0.5
SSIM_T = 0.921   # من train+val

SPLIT = {"alert": "val", "form": "test", "profile": "holdout"}

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
    all_rows = list(csv.DictReader(f))


def report(name, y, pred):
    acc = (pred == y).float().mean().item()
    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    print(f"  {name:8s} acc={acc:.2%}  TP={tp} TN={tn} FP(missed bug)={fp} FN(false alarm)={fn}")


for comp, split in SPLIT.items():
    rows = [r for r in all_rows if r["component"] == comp]
    A = torch.stack([load(SAMPLES / f"{r['id']}_design.png") for r in rows])
    B = torch.stack([load(SAMPLES / f"{r['id']}_code.png") for r in rows])
    y = torch.tensor([1 if r["label"] == "match" else 0 for r in rows])

    with torch.no_grad():
        cnn = torch.cat([model(prep(A[i:i + 32]), prep(B[i:i + 32])).argmax(1)
                         for i in range(0, len(A), 32)])
    ssim_pred = torch.tensor([1 if ssim[r["id"]] >= SSIM_T else 0 for r in rows])
    hybrid = cnn & ssim_pred   # match فقط لو الاثنين قالوا match

    print(f"\n== {split} ({comp}) ==")
    report("SSIM", y, ssim_pred)
    report("CNN", y, cnn)
    report("Hybrid", y, hybrid)