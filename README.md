# Automated Malaria Detection Using YOLOv11 and a Stacked Ensemble of Convolutional Neural Networks with Grad-CAM Visualization

## Authors

* **Shawn Jurgen Mayol**
* **Elgen Mar Arinasa**

## Advisor

* **Archival J. Sebial, DIT**

## Introduction

This repository contains all code, notebooks, documentation, and experimental results for our undergraduate thesis project at the University of San Carlos.

**Thesis Title:**
*Automated Malaria Detection Using YOLOv11 and a Stacked Ensemble of Convolutional Neural Networks with Grad-CAM Visualization*

Malaria remains a major health burden, especially in low-resource regions where access to skilled laboratory diagnosis is limited. Manual microscopic examination is time-consuming, labor-intensive, and prone to human error. This project aims to develop an automated malaria detection pipeline by leveraging deep learning and computer vision:

* **YOLOv11** is used for rapid and accurate detection/localization of red blood cells and parasite candidates in blood smear images.
* A **stacked ensemble of CNN classifiers** distinguishes between parasitized and uninfected cells, improving robustness and accuracy.
* **Grad-CAM visualization** highlights image regions that most influence the model's predictions, increasing interpretability for clinicians.

The system is designed to be reproducible, explainable, and feasible for deployment in resource-limited settings.

## Repository Structure

```
malaria-detection-thesis/
│
├── data/                # Datasets
│   ├── raw/
│   └── processed/
│
├── notebooks/           # Jupyter/Colab notebooks for experiments & EDA
│
├── src/                 # Source code (modules, models, training scripts)
│   ├── models/          # YOLO, CNN ensemble, Grad-CAM scripts
│   ├── utils/           # Helper functions (preprocessing, evaluation, etc.)
│   └── main.py          # Entry point if needed
│
├── results/             # Outputs: predictions, logs, visualizations, Grad-CAM heatmaps
│
├── reports/             # Thesis docs, literature summaries, references, presentations
│
├── requirements.txt     # Python dependencies
│
├── .gitignore           # Git ignore file
│
└── README.md            # This file
```

## Quickstart

1. **Set up your environment:**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate         # On Windows
   pip install -r requirements.txt
   ```

2. **Run Jupyter notebooks:**

   * Open VS Code or run `jupyter notebook` from the project root.
   * Start with `notebooks/eda.ipynb` for data exploration.

3. **Data:**

   * Place raw datasets in `data/raw/`.
   * Preprocessed/augmented data should be saved in `data/processed/`.

4. **Results and reports:**

   * All output images, visualizations, and logs are in `results/`.
   * Documentation and thesis write-ups are in `reports/`.

---

This code is intended for academic and research use only.
