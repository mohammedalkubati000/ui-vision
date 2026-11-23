import csv

import torch

from predict import ROOT, SSIM_T, SiameseNet, load, prep

SAMPLES = ROOT / "dataset" / "samples"

model = SiameseNet()
model.load_state_dict(torch.load(ROOT / "models" / "best.pt"))
model.eval()

with open(ROOT / "dataset" / "baseline_results.csv", encoding="utf-8") as f:
    ssim = {r["id"]: float(r["ssim"]) for r in csv.DictReader(f)}
with open(ROOT / "dataset" / "labels.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["component"] == "profile"]

groups = {
    "CNN right, SSIM wrong": [],
    "SSIM right, CNN wrong": [],
    "both wrong": [],
}

for r in rows:
    a = load(SAMPLES / f"{r['id']}_design.png").unsqueeze(0)
    b = load(SAMPLES / f"{r['id']}_code.png").unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(prep(a), prep(b)), dim=1)[0, 1].item()
    truth = r["label"] == "match"
    ssim_ok = (ssim[r["id"]] >= SSIM_T) == truth
    cnn_ok = (p >= 0.5) == truth
    line = f"{r['id']}  {r['label']:8s} ssim={ssim[r['id']]:.3f}  cnn={p:.2f}  {r['change']}"
    if cnn_ok and not ssim_ok:
        groups["CNN right, SSIM wrong"].append(line)
    elif ssim_ok and not cnn_ok:
        groups["SSIM right, CNN wrong"].append(line)
    elif not ssim_ok and not cnn_ok:
        groups["both wrong"].append(line)

for name, lines in groups.items():
    print(f"\n== {name}: {len(lines)} ==")
    for line in lines[:4]:
        print(line)