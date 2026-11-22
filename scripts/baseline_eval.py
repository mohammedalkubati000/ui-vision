import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPLIT = {
    "card": "train", "button": "train", "navbar": "train",
    "alert": "val",
    "form": "test",
}

rows = []
with open(ROOT / "dataset" / "baseline_results.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((r["component"], r["label"], float(r["ssim"])))


def accuracy(sub, t):
    return sum((r[2] >= t) == (r[1] == "match") for r in sub) / len(sub)


fit = [r for r in rows if SPLIT[r[0]] in ("train", "val")]
test = [r for r in rows if SPLIT[r[0]] == "test"]

best_t = max((t / 1000 for t in range(500, 1000)), key=lambda t: accuracy(fit, t))

print(f"threshold (من train+val) = {best_t:.3f}")
print(f"train+val accuracy       = {accuracy(fit, best_t):.2%}")
print(f"TEST accuracy (form)     = {accuracy(test, best_t):.2%}")

tp = sum(r[2] >= best_t and r[1] == "match" for r in test)
tn = sum(r[2] < best_t and r[1] == "mismatch" for r in test)
fp = sum(r[2] >= best_t and r[1] == "mismatch" for r in test)
fn = sum(r[2] < best_t and r[1] == "match" for r in test)
print(f"\nTest confusion: TP(match ok)={tp}  TN(mismatch ok)={tn}  FP(missed bug)={fp}  FN(false alarm)={fn}")