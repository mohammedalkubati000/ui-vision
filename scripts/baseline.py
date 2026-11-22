import csv
import cv2
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "dataset" / "samples"

results = []  # (id, component, label, score)
with open(ROOT / "dataset" / "labels.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        d = cv2.imread(str(SAMPLES / f"{row['id']}_design.png"))
        c = cv2.imread(str(SAMPLES / f"{row['id']}_code.png"))

        # لو الأبعاد مختلفة نوحّدها
        if c.shape != d.shape:
            c = cv2.resize(c, (d.shape[1], d.shape[0]))

        d_gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        score = ssim(d_gray, c_gray)
        results.append((row["id"], row["component"], row["label"], score))

with open(ROOT / "dataset" / "baseline_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "component", "label", "ssim"])
    w.writerows(results)


def accuracy(rows, thresh):
    ok = sum((r[3] >= thresh) == (r[2] == "match") for r in rows)
    return ok / len(rows)


print("== SSIM لكل label ==")
for lab in ["match", "mismatch"]:
    s = [r[3] for r in results if r[2] == lab]
    print(f"{lab:9s} mean={sum(s)/len(s):.4f}  min={min(s):.4f}  max={max(s):.4f}")

print("\n== الدقة عند حد 0.95 ==")
print(f"all: {accuracy(results, 0.95):.2%}")
for comp in sorted({r[1] for r in results}):
    sub = [r for r in results if r[1] == comp]
    print(f"{comp:8s}: {accuracy(sub, 0.95):.2%}")

best_t = max((t / 1000 for t in range(500, 1000)), key=lambda t: accuracy(results, t))
print(f"\n== أفضل حد فاصل ==")
print(f"threshold={best_t:.3f}  accuracy={accuracy(results, best_t):.2%}")