# End-to-End Sales Forecasting Project

Simple sales forecasting pipeline: data preprocessing, model training, and a minimal Flask/Dash/Plotly UI.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run training pipeline:

```bash
python src/model_training.py
```

3. Start the app (if implemented):

```bash
python src/app.py
```

Files

- `src/data_preprocessing.py`: preprocessing utilities
- `src/model_training.py`: model training and saving
- `sales_data.csv`: example dataset

Notes

- `requirements.txt` lists pinned dependencies. Update if you add packages.
