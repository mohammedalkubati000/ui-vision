from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
html_file = ROOT / "components" / "card.html"
out_file = ROOT / "dataset" / "test_original.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 800, "height": 600}, device_scale_factor=1)
    page.goto(html_file.as_uri())
    page.locator("#target").screenshot(path=str(out_file))
    browser.close()

print("Saved:", out_file)