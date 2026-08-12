# SpectraTwin

SpectraTwin is a Streamlit application for **spectral data analysis and machine learning** —
experimental design, visualization, preprocessing, model training (classical ML + neural
networks), prediction, control charts, and real-time monitoring — all from one interface.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [How to Use](#how-to-use)
   - [Experimental Design](#1-experimental-design)
   - [Data Visualization](#2-data-visualization)
   - [Preprocessing](#3-preprocessing)
   - [Model Training](#4-model-training)
   - [Neural Network Builder](#5-neural-network-builder)
   - [Model Prediction](#6-model-prediction)
   - [Full Pipeline & One-Click Pipeline](#7-full-pipeline--one-click-pipeline)
   - [Control Charts](#8-control-charts)
   - [Real-Time Transfer](#9-real-time-transfer)
   - [HPLC](#10-hplc)
3. [Data Format](#data-format)
4. [Troubleshooting](#troubleshooting)
5. [Project Structure](#project-structure)
6. [License](#license)

---

## Quick Start

**Requirements:** Python **3.10** recommended.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (macOS only, for XGBoost) install the OpenMP runtime
#    conda install -c conda-forge llvm-openmp   # if using conda
#    brew install libomp                        # if using Homebrew

# 3. Run the app
streamlit run Home.py
```

The app opens at `http://localhost:8501`. Use the **left sidebar** to move between pages.

**Optional — AI chatbot:** to enable the in-app assistant, get a free key from
[Groq Console](https://console.groq.com) and set it as an environment variable
(`GROQ_API_KEY`) or paste it into the chatbot box when prompted. The app works fully
without it.

---

## How to Use

A typical workflow is: **Preprocess → Train a model → Predict**. Each page also works
on its own. Most pages start by asking you to **upload a data file** and **select the
target column(s)** to model.

> **Tip — everything is editable / re-runnable.** You can go back a step, change a
> selection, and re-run at any time. In Experimental Design you can rename or change
> components after creating a project (see below).

### 1. Experimental Design
Plan which experiments to run next using active learning (Gaussian-process based).

1. **New Project** → enter your output/**component names** (comma-separated, e.g.
   `Component_A, Component_B`) → **Create Project**.
2. **Edit anytime:** in the project menu open **"✏️ Edit Project — component names"** to
   rename components (data is kept) or change how many there are (this resets collected
   data, with a confirmation).
3. **Data Management** tab → upload your existing results (CSV/Excel/TXT) or enter them
   manually.
4. **Design & Analysis** tab → define target ranges and **generate a Y-space plan**, or
   let the model **suggest the next most informative experiment**.
5. Use **← Back to Home** to return, or **Save Project** to download it as a `.pkl`.

### 2. Data Visualization
Explore spectra interactively.

1. Upload a spectral file and pick the **target column(s)**.
2. Browse tabs: **spectra overlay**, **statistics**, **correlation heatmap**,
   **PCA**, and **peak detection** (adjust prominence/distance to find peaks).

### 3. Preprocessing
Clean and transform spectra, then export ready-to-model data.

1. **Upload** data and **select target column(s)** (non-numeric feature columns are
   dropped automatically).
2. Optional steps, in order: **Outlier removal → Wavelet denoising → Standard
   preprocessing → Advanced (FFT / OPLS) → Export**.
3. **Standard preprocessing** offers a **technique** (General / Raman / NIR / FTIR) and
   two modes:
   - **Manual** — pick exactly which steps to apply (baseline, smoothing,
     normalization, derivatives, …).
   - **Automated** — Optuna searches the best preprocessing pipeline for you (a live
     progress bar shows trial-by-trial status).
4. **Export** downloads the preprocessed **X**, the **targets (y)**, and a
   **parameters JSON** describing every step (used later for prediction).

### 4. Model Training
Train and compare classical ML models.

1. **Data source:** upload a file **or** choose **"Use Session State (Preprocessed)"**
   to reuse the data you just made on the Preprocessing page — no re-upload needed.
2. Select **target column(s)** (one or more), then do a **Train/Test split**.
3. Choose a **run type**:
   - **Manual** — you set the hyperparameters.
   - **Defined Hypertuning** — pick one model; Optuna tunes its hyperparameters
     (via cross-validation on the training data).
   - **Automated** — searches across all models and picks the best.
   You can set the **number of tuning iterations**.
4. **Save** the trained model — you get `model.pkl`, `parameters.json`, and
   (for pipelines) a `fitted_objects.pkl`. Keep all of these for prediction.

### 5. Neural Network Builder
Build neural networks — no coding.

1. Upload **CSV or Excel** (or use session-state preprocessed data) and select
   **one or more target columns**.
2. **Visual Block Builder (Manual)** — stack Dense/Conv1D/Pooling/Dropout/Flatten
   layers and train (uses TensorFlow/Keras).
3. **AutoML Tuner** — fast, TensorFlow-free random search over a scikit-learn network:
   - **MLP (Dense)**, or
   - **1D-CNN (Conv features + MLP)** — captures local spectral patterns.
   Set the number of trials; the best model is exportable as a `.pkl` that works
   directly on the Prediction page.

### 6. Model Prediction
Apply a saved model to new data.

1. Upload the **Model (.pkl)**, the **Parameters (.json)**, and — for best results —
   the **Fitted objects (.pkl)**.
2. Choose the **model source** (Auto-detect / One-Click / General) — auto-detect usually
   works.
3. Upload the **new data**, then **Run Prediction**. Download the results as CSV.

> Uploading the **fitted objects** reproduces the training preprocessing exactly. Without
> it, preprocessing is reconstructed approximately and you'll see a warning.

### 7. Full Pipeline & One-Click Pipeline
- **Full Pipeline** — the whole workflow (upload → preprocess → dimensionality reduction
  → augmentation → train → evaluate → save) in one guided page.
- **One-Click Pipeline** — the fastest route: upload, pick targets, choose a technique
  (General/Raman/NIR/FTIR), optionally **skip preprocessing** if your data is already
  processed, and it runs preprocessing + automated model selection end-to-end.

### 8. Control Charts
Batch-process monitoring with **Multiway PLS (MPLS)** and statistical process control
(DModX, Hotelling's T², score charts). Upload data, set the batch structure, fit the
MPLS model, and review the SPC charts.

### 9. Real-Time Transfer
Monitor a folder for incoming spectra and broadcast live predictions over your local
network (LAN). **This feature is for local / self-hosted use only.**

**On the analysis machine (server):**
1. Open the **Real-Time Transfer** page, set the folder to watch and a port.
2. Upload your **Model (.pkl)**, **Parameters (.json)**, and optional **Fitted objects
   (.pkl)**, and pick the model source.
3. Click **Start Server** and copy the shown WebSocket URL (`ws://…:8765`).

**On the instrument machine (client):**
```bash
python realtime_client.py
```
Paste the WebSocket URL, click **Connect**, then drop spectra files (`.csv`, `.txt`,
`.xlsx`) into the watched folder — predictions appear live. Use **Stop Server** to shut
it down cleanly (the port is released so you can restart).

### 10. HPLC
Tools for High-Performance Liquid Chromatography data analysis.

---

## Data Format

- **Rows = samples, columns = features (wavelengths) + target(s).**
- Feature/wavelength column headers should be **numeric** (e.g. `800.0, 805.0, …`).
  Non-numeric feature columns (sample IDs, labels) are detected and dropped.
- Supported files: **CSV, Excel (.xlsx/.xls), TXT (tab-delimited)**.
- Select which column(s) are the **target(s)** on each page (multiple targets are
  supported and train a single multi-output model).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| XGBoost `libomp.dylib` error (macOS) | `conda install -c conda-forge llvm-openmp` (or `brew install libomp`) |
| Dependency conflicts | Use **Python 3.10**; reinstall with `pip install -r requirements.txt` |
| Chatbot not responding | Set `GROQ_API_KEY` (or paste a key in the chatbot box); the app still works without it |
| Prediction feature-count mismatch | Upload the **fitted objects (.pkl)** saved with the model |
| Port 8501 already in use | `streamlit run Home.py --server.port 8502` |
| Real-time "port in use" after stop | Use **Stop Server** (it frees the port); pick another port if needed |

---

## Project Structure

```
SpectraTwin/
├── Home.py                     # App entry point
├── pages/                      # One file per feature (sidebar pages 00–10)
├── preprocess.py               # SpectralData + preprocessing operations
├── midel.py                    # Models, tuning (optuna), AutoModelSelector, NN
├── prediction_utils.py         # Shared prediction helpers (used by 07 & 09)
├── target_utils.py             # Single/multi-target handling
├── pca.py / mpls.py / opls.py  # PCA / Multiway-PLS / Orthogonal-PLS
├── FFT.py / hplc.py            # FFT filtering / HPLC
├── data_augmentation.py        # Augmentation techniques
├── spectra_specific/           # Technique-specific preprocessing optimizers
├── realtime_client.py          # Real-time client (run on the instrument PC)
├── chatbot.py                  # Optional AI assistant (Groq)
└── requirements.txt
```

---

## License

MIT License — see the `LICENSE` file.
