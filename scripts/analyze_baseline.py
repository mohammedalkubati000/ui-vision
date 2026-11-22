import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

changes = {}
with open(ROOT / "dataset" / "labels.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        changes[row["id"]] = row["change"]

rows = []
with open(ROOT / "dataset" / "baseline_results.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append((row["id"], row["label"], float(row["ssim"])))

THRESH = 0.95

print("== mismatch لكن SSIM عالي (الـ baseline غلطان) ==")
fn = sorted([r for r in rows if r[1] == "mismatch" and r[2] >= THRESH], key=lambda r: -r[2])
for r in fn[:8]:
    print(f"{r[0]}  ssim={r[2]:.4f}  {changes[r[0]]}")
print(f"المجموع: {len(fn)}\n")

print("== match لكن SSIM منخفض (الـ baseline غلطان) ==")
fp = sorted([r for r in rows if r[1] == "match" and r[2] < THRESH], key=lambda r: r[2])
for r in fp[:8]:
    print(f"{r[0]}  ssim={r[2]:.4f}  {changes[r[0]]}")
print(f"المجموع: {len(fp)}")