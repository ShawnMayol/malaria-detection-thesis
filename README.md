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
│
├── requirements.txt     # Python dependencies
│
│
├── .gitignore
│
└── README.md


python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt