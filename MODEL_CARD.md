# PhishGuard Multimodal Model Card

## Model

- Text branch: frozen `distilbert/distilbert-base-uncased`, maximum 128 tokens
- Metadata branch: 15 scaled sender, receiver, message, and URL features processed by a two-layer neural network
- Fusion: projected DistilBERT embedding concatenated with the metadata embedding, followed by a binary neural classifier
- Explainability: final-layer DistilBERT attention weights for text and Kernel SHAP for metadata

## Training data

The model was trained on all 3,065 labelled rows in `Nazario_5.csv` from the cited Kaggle archive. The archive also contains 53,668 text-only rows, but those were excluded because they do not contain the metadata required for multimodal evaluation.

- Labels: 1,565 phishing and 1,500 legitimate
- Split: stratified 70% train, 15% validation, 15% test
- Seed: 42
- Source CSV SHA-256: `afef69828165619ced6661ca14a95444a1db82c7b25ae9c2619bbbb108844a44`
- Held-out test support: 460

## Held-out results

| Metric | Value |
|---|---:|
| Accuracy | 0.9826 |
| Precision | 0.9956 |
| Recall | 0.9702 |
| F1 | 0.9828 |
| ROC-AUC | 0.9975 |

Confusion matrix (rows actual, columns predicted): `[[224, 1], [7, 228]]`.

The exact machine-readable configuration, training history, metrics, and dataset checksum are stored in `model/config.json` and `model/metrics.json`.

## Limitations

- The model was evaluated on a random held-out split of one source dataset, not on a separate contemporary external corpus.
- DistilBERT is frozen; only the projection, metadata, and fusion layers are trained.
- Inputs longer than 128 tokens are truncated.
- Attention weights are diagnostic signals, not causal explanations.
- SHAP values explain the metadata branch conditional on the current text embedding.
- The model must support—not replace—human security review.
