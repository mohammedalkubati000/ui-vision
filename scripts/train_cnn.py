import csv
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models

torch.manual_seed(0)
np.random.seed(0)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "dataset" / "samples"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SPLIT = {
    "card": "train", "button": "train", "navbar": "train",
    "alert": "val",
    "form": "test",
}

W, H = 704, 256      # حجم اللوحة الثابتة
SCALE = 0.5          # نصغّرها للنص عشان التدريب أسرع
EPOCHS = 8
BATCH = 16
LR = 1e-4

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(path):
    img = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (W, H), (243, 244, 246))
    canvas.paste(img, (0, 0))
    canvas = canvas.resize((int(W * SCALE), int(H * SCALE)), Image.BILINEAR)
    return torch.from_numpy(np.array(canvas)).permute(2, 0, 1)  # uint8


def prep(x):
    return (x.float() / 255 - MEAN) / STD


print("Loading images...")
raw = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
with open(ROOT / "dataset" / "labels.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        A, B, Y = raw[SPLIT[row["component"]]]
        A.append(load(SAMPLES / f"{row['id']}_design.png"))
        B.append(load(SAMPLES / f"{row['id']}_code.png"))
        Y.append(1 if row["label"] == "match" else 0)

data = {k: (torch.stack(v[0]), torch.stack(v[1]), torch.tensor(v[2])) for k, v in raw.items()}
for k, v in data.items():
    print(f"{k}: {len(v[2])} pairs")


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )

    def forward(self, a, b):
        f = self.backbone(torch.cat([a, b]))
        fa, fb = f.chunk(2)
        return self.head(torch.cat([(fa - fb).abs(), fa * fb], dim=1))


@torch.no_grad()
def predict(model, A, B):
    model.eval()
    preds = []
    for i in range(0, len(A), 32):
        out = model(prep(A[i:i + 32]), prep(B[i:i + 32]))
        preds.append(out.argmax(1))
    return torch.cat(preds)


model = SiameseNet()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

A_tr, B_tr, Y_tr = data["train"]
A_va, B_va, Y_va = data["val"]
best_val, best_epoch = 0.0, 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()   # نثبّت إحصائيات الـ BatchNorm المدرّبة مسبقاً
    perm = torch.randperm(len(Y_tr))
    total_loss = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        out = model(prep(A_tr[idx]), prep(B_tr[idx]))
        loss = loss_fn(out, Y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)

    pv = predict(model, A_va, B_va)
    val_acc = (pv == Y_va).float().mean().item()
    print(f"   val predicted_match = {pv.float().mean().item():.0%}")
    print(f"epoch {epoch}/{EPOCHS}  loss={total_loss / len(Y_tr):.4f}  "
          f"val_acc={val_acc:.2%}  ({time.time() - t0:.0f}s)")

    if val_acc > best_val:
        best_val, best_epoch = val_acc, epoch
        torch.save(model.state_dict(), MODEL_DIR / "best.pt")

print(f"\nbest val_acc = {best_val:.2%} (epoch {best_epoch})")

# التقييم النهائي على الـ test (مرة وحدة بس)
model.load_state_dict(torch.load(MODEL_DIR / "best.pt"))
A_te, B_te, Y_te = data["test"]
pred = predict(model, A_te, B_te)

acc = (pred == Y_te).float().mean().item()
tp = ((pred == 1) & (Y_te == 1)).sum().item()
tn = ((pred == 0) & (Y_te == 0)).sum().item()
fp = ((pred == 1) & (Y_te == 0)).sum().item()
fn = ((pred == 0) & (Y_te == 1)).sum().item()

print(f"TEST accuracy (form) = {acc:.2%}")
print(f"Test confusion: TP(match ok)={tp}  TN(mismatch ok)={tn}  FP(missed bug)={fp}  FN(false alarm)={fn}")