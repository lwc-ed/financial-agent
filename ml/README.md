# ML workspace

This folder is for offline ML research, dataset preparation, training, and evaluation.

## Suggested environment split

- `backend/requirements.txt`: production API and app runtime dependencies
- `ml/requirements.txt`: research and training dependencies for a separate `venv_ml`

Example setup:

```bash
python3 -m venv venv_ml
source venv_ml/bin/activate
pip install -r ml/requirements.txt
```

## Directory purpose

- `training/make_dataset.py`: clean a raw CSV into a training-ready dataset
- `training/train.py`: train a baseline model and sync artifacts into `backend/ml_inference/artifacts`
- `training/evaluate.py`: run a basic evaluation against a labeled dataset
- `artifacts/`: local training outputs before or while syncing to backend
- `notebooks/`: optional experiments and EDA

## Baseline workflow

```bash
python ml/training/make_dataset.py --input data/raw.csv --output ml/artifacts/train.csv
python ml/training/train.py --dataset ml/artifacts/train.csv
python ml/training/evaluate.py --model ml/artifacts/risk_model.pkl --dataset ml/artifacts/train.csv
```

## Important note

`backend/ml_inference/inference.py` loads `risk_model.pkl` directly with `pickle`.
If the model is trained with scikit-learn, the backend runtime also needs a compatible `scikit-learn` version installed before inference can load that artifact.

