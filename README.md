# Máltækni lokaverkefni

Final project workspace for NLP / máltækni.

Author: Sölvi

## Status

This repository is in setup mode. The actual project implementation and write-up have not started yet.

## Structure

- `src/maltaekni_lokaverkefni/`: reusable Python code
- `notebooks/`: exploratory notebooks
- `data/raw/`: local raw data, not committed
- `data/processed/`: local processed data, not committed
- `models/`: local model artifacts, not committed
- `reports/figures/`: generated figures, not committed by default
- `docs/`: project notes and non-private documentation

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name maltaekni-lokaverkefni --display-name "Python (maltaekni-lokaverkefni)"
```

## Notes

The local PDF handouts are intentionally ignored so the public GitHub repository starts clean.
