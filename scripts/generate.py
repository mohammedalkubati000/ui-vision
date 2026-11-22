import csv
import json
import random
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

random.seed(42)

COMPONENTS = ["card", "button", "navbar", "alert", "form"]
N_PER_COMPONENT = 100      # نص match ونص mismatch
MIN_COLOR_DIST = 90        # أقل مسافة لون عشان يعتبر mismatch واضح

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "samples"

# نمسح الداتا القديمة ونبدأ من جديد
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

DEFAULTS = {
    "--pad": "24px",
    "--bg": "#2563eb",
    "--weight": "600",
    "--radius": "12px",
    "--font-size": "16px",
}
BASE_RGB = (0x25, 0x63, 0xEB)


def rgb_to_hex(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, v)) for v in c)


def far_color():
    """لون عشوائي يبعد عن الأزرق الأصلي بشكل واضح"""
    while True:
        c = [random.randint(0, 255) for _ in range(3)]
        dist = sum((a - b) ** 2 for a, b in zip(c, BASE_RGB)) ** 0.5
        if dist >= MIN_COLOR_DIST:
            return rgb_to_hex(c)


def tiny_change():
    """اهتزاز بسيط جداً = Match"""
    ch = {}
    if random.random() < 0.3:
        ch["--pad"] = f"{24 + random.choice([-1, 1])}px"
    if random.random() < 0.3:
        ch["--bg"] = rgb_to_hex([v + random.randint(-3, 3) for v in BASE_RGB])
    if random.random() < 0.3:
        ch["--radius"] = f"{12 + random.choice([-1, 1])}px"
    if random.random() < 0.3:
        ch["--font-size"] = f"{16 + random.choice([-0.5, 0.5])}px"
    return ch


def big_change():
    """تغيير واضح في خاصية أو خاصيتين = Mismatch"""
    options = {
        "--pad": lambda: f"{random.choice([6, 8, 12, 36, 40, 48])}px",
        "--bg": far_color,
        "--weight": lambda: random.choice(["300", "400", "800", "900"]),
        "--radius": lambda: f"{random.choice([0, 2, 24, 32, 40])}px",
        "--font-size": lambda: f"{random.choice([11, 12, 13, 20, 22, 24])}px",
    }
    props = random.sample(list(options), k=random.choice([1, 1, 2]))
    return {p: options[p]() for p in props}


def shot(page, url, overrides, path):
    page.goto(url)
    values = {**DEFAULTS, **overrides}
    for key, val in values.items():
        page.evaluate(
            "([k, v]) => document.documentElement.style.setProperty(k, v)",
            [key, val],
        )
    page.locator("#target").screenshot(path=str(path))


rows = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 900, "height": 600}, device_scale_factor=1)

    for comp in COMPONENTS:
        url = (ROOT / "components" / f"{comp}.html").as_uri()
        for i in range(N_PER_COMPONENT):
            label = "match" if i % 2 == 0 else "mismatch"
            change = tiny_change() if label == "match" else big_change()

            sample_id = f"{comp}_{i:04d}"
            shot(page, url, {}, OUT_DIR / f"{sample_id}_design.png")
            shot(page, url, change, OUT_DIR / f"{sample_id}_code.png")

            rows.append([sample_id, comp, label, json.dumps(change)])
        print("finished:", comp)

    browser.close()

with open(ROOT / "dataset" / "labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "component", "label", "change"])
    writer.writerows(rows)

print("Done:", len(rows), "pairs")