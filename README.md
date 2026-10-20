# AI-Based Phishing Email Detection System

A runnable multimodal deep-learning system combining DistilBERT email-text embeddings with a neural metadata branch and late fusion. It provides real-time predictions, attention diagnostics, metadata SHAP explanations, `.eml`/`.txt` import, and a Flask dashboard.

## Verified model

The included artifacts were trained from all 3,065 metadata-complete rows in `Nazario_5.csv` using a stratified 70/15/15 split (seed 42). The held-out 460-email test set produced:

| Metric | Verified value |
|---|---:|
| Accuracy | 98.26% |
| Precision | 99.56% |
| Recall | 97.02% |
| F1 | 98.28% |
| ROC-AUC | 99.75% |

The confusion matrix was `[[224, 1], [7, 228]]`. See [MODEL_CARD.md](MODEL_CARD.md), `model/config.json`, and `model/metrics.json` for provenance, limitations, exact losses, split sizes, and checksums. These values are separate from the dissertation's originally reported results.

## Architecture

- Frozen `distilbert/distilbert-base-uncased` contextual text encoder
- Fifteen standardized sender, receiver, message, and URL features
- Two-layer metadata neural network
- Late-fusion binary neural classifier trained with BCE-with-logits loss and AdamW
- Kernel SHAP metadata attribution conditional on the current text embedding
- Final-layer DistilBERT attention visualization
- Automatic heuristic fallback only when model artifacts are absent

## Install and run

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements-ml.txt
python download_text_encoder.py  # fetches the frozen distilbert-base-uncased weights (git-ignored, ~255MB)
python app.py
```

Open `http://127.0.0.1:5000`. The banner and `/api/health` should report `distilbert-multimodal` and `model_loaded: true`.

Environment overrides: `HOST`, `PORT`, `FLASK_DEBUG`, `MODEL_DIR`, and `SECRET_KEY`.

### Docker

The `Dockerfile` installs the CPU build of PyTorch, fetches the DistilBERT weights at build time and serves the app with gunicorn on port 7860:

```bash
docker build -t phishguard-ai .
docker run --rm -p 7860:7860 phishguard-ai
```

Then open `http://127.0.0.1:7860`. The first build is large because of PyTorch.

## Reproduce training

The Kaggle archive has no published license metadata. Review its terms before redistribution. Raw data is excluded from Git.

```bash
source venv/bin/activate
python download_dataset.py
python train_model.py --max-length 128 --batch-size 32 --epochs 15
```

The downloader verifies the source archive checksum. Training writes the local DistilBERT weights, tokenizer, trained fusion head, fitted scaler, SHAP background, configuration, training history, and held-out metrics under `model/`. Use `--max-samples N` only for development smoke runs.

## Dataset facts

The cited Kaggle archive contains:

- `Nazario_5.csv`: 3,065 rows with sender, receiver, subject, body, URL, and label fields
- `email_text.csv`: 53,668 rows with text and label only
- Total archive rows: 56,733, not the 82,500+ stated in the dissertation

Only `Nazario_5.csv` is used for multimodal training because the text-only file lacks structural metadata. Mixing it into the held-out multimodal evaluation would misrepresent the architecture.

## Testing

```bash
source venv/bin/activate
python -m unittest -v
python -m py_compile app.py test_app.py phishing_model/*.py train_model.py download_dataset.py
node --check static/js/script.js
```

## Main routes

- `/` — detection dashboard
- `/analyze` — model-backed JSON prediction endpoint
- `/upload` — `.eml`/`.txt` parsing endpoint
- `/results` — metrics loaded from the live artifact
- `/about` — methodology and project context
- `/api/health` — engine and model-load status

## Responsible use

This detector should support, not replace, human security review. Its evaluation uses a random split of one historical dataset rather than a separate contemporary corpus, messages are truncated to 128 tokens, and attention weights are diagnostic rather than causal explanations.

This project is part of a Northumbria University MSc Cyber Security dissertation by Adeoye Christopher Ayomiposi (ID 24072790), supervised by Hamid Jahankhani and Umair Chaudhry.
