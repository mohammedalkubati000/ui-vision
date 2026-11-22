# UI-Vision: Automated Design-to-Code Visual Match and Mismatch Classifier

**By:** Mohammed Alkubati  
**Date:** November 2026  

---

## 1. Project Idea & Problem Statement

Frontend developers and UI/UX designers spend considerable time manually comparing implemented web pages against original Figma designs to detect visual regressions—such as incorrect padding or margins, wrong font weights, color shifts, or misaligned components. Manual visual QA is slow, repetitive, and prone to human oversight before shipping to production.

**UI-Vision** automates this comparison workflow by taking two input images:
1. **Figma design** snapshot (`design`).
2. **Coded page screenshot** (`code`).

The system evaluates both inputs and returns similarity scores and classifications across two approaches:
- **High Match (Production Ready)**
- **Visual Mismatch (Needs Fix)**

---

## 2. Computer Vision Task, Dataset & Architecture

### Task & Classification
- **Image Classification** (supported by classical structural/pixel comparison) to classify visual consistency into **High Match** vs. **Visual Mismatch** without the manual mask-labeling overhead of segmentation.

### Dataset & Splitting Strategy
- Paired UI component designs and rendered code screenshots (including controlled synthetic visual defects such as color shifts and layout/style differences), with `labels.csv` tracking sample ground truth.
- Partitioned across training, validation, and test sets by **UI component type** to prevent data leakage across similar component variants.

### Tech Stack & Models
- **Baseline (Classical CV):** Python, OpenCV, and `scikit-image` computing Structural Similarity Index (**SSIM**) against `SSIM_T` and pixel-level differences.
- **Deep Learning Classifier:** **SiameseNet** backed by **ResNet18** (`best.pt`), outputting match probability (`cnn_match_prob`) and verdict (`High Match` / `Visual Mismatch`).
- **Backend API:** **FastAPI** (`scripts/api.py`) served via **Uvicorn** on `http://localhost:8000` with endpoint `POST /compare`.
- **Frontend Dashboard:** **Next.js** (App Router) and **Tailwind CSS** (`dashboard/app/page.tsx`) running on `http://localhost:3000`, providing side-by-side image upload, preview, and dual comparison cards.

---

## 3. Step-by-Step Progress

1. **Proposal & Scope Definition:** Defined the COMP 491 project scope for automated design-to-code visual verification (`UI-Vision`).
2. **Dataset & CLI Inference (`scripts/predict.py`):** Prepared paired samples and `labels.csv`, and ran CLI inference comparing SSIM and the CNN (`SiameseNet` / `ResNet18`) on `High Match` and `Visual Mismatch` pairs.
3. **Backend API (`scripts/api.py`):** Built the FastAPI `POST /compare` endpoint loading `models/best.pt`, handling image decoding/shape alignment, computing SSIM and CNN match probability, and returning JSON responses (`HTTP 200`).
4. **End-to-End Dashboard (`dashboard/`):** Initialized the Next.js + Tailwind CSS app and implemented the main comparison page (`app/page.tsx`) with dual image upload, preview, `Compare` trigger, and side-by-side result cards for **Baseline (SSIM)** and **CNN (ResNet18)**.

---

## 4. Challenges, Root Causes & Improvements

### Grayscale Blindness in SSIM (`profile_0001`: Blue vs. Turquoise Card)
- **Observation on `profile_0001` (`mismatch`):** A UI card whose color changed from **blue to turquoise** produced:
  - **SSIM:** `0.9307` (`~0.931`) $\rightarrow$ **High Match** *(missed the color change)*
  - **CNN (`ResNet18`):** `0.0` (`0.0%`) $\rightarrow$ **Visual Mismatch** *(caught the visual mismatch)*
- **Comparison on `profile_0002` (`match`):** Produced `SSIM = 0.9999` (`High Match`) and `cnn_match_prob = 0.8868` (`High Match`).
- **Root Cause & Improvement:**
