import csv
import json
import random
from pathlib import Path
from playwright.sync_api import sync_playwright

random.seed(123)

COMP = "profile"
N = 100
MIN_COLOR_DIST = 90

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "samples"
LABELS = ROOT / "dataset" / "labels.csv"

with open(LABELS, encoding="utf-8") as f:
    if any(r["component"] == COMP for r in csv.DictReader(f)):
        raise SystemExit("profile already generated, aborting")

DEFAULTS = {"--pad": "24px", "--bg": "#2563eb", "--weight": "600",
            "--radius": "12px", "--font-size": "16px"}
BASE_RGB = (0x25, 0x63, 0xEB)


def rgb_to_hex(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, v)) for v in c)


def far_color():
    while True:
        c = [random.randint(0, 255) for _ in range(3)]
        if sum((a - b) ** 2 for a, b in zip(c, BASE_RGB)) ** 0.5 >= MIN_COLOR_DIST:
            return rgb_to_hex(c)


def tiny_change():
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
    for key, val in {**DEFAULTS, **overrides}.items():
        page.evaluate("([k, v]) => document.documentElement.style.setProperty(k, v)", [key, val])
    page.locator("#target").screenshot(path=str(path))


rows = []
url = (ROOT / "components" / f"{COMP}.html").as_uri()
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 900, "height": 600}, device_scale_factor=1)
    for i in range(N):
        label = "match" if i % 2 == 0 else "mismatch"
        change = tiny_change() if label == "match" else big_change()
        sid = f"{COMP}_{i:04d}"
        shot(page, url, {}, OUT_DIR / f"{sid}_design.png")
        shot(page, url, change, OUT_DIR / f"{sid}_code.png")
        rows.append([sid, COMP, label, json.dumps(change)])
    browser.close()

with open(LABELS, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

print("Done:", len(rows), "profile pairs added")