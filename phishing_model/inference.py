import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from .features import FEATURE_NAMES, feature_vector
from .model import MultimodalPhishingModel


class PhishingInferenceEngine:
    def __init__(self, artifact_dir='model'):
        self.artifact_dir = Path(artifact_dir)
        self.config = json.loads((self.artifact_dir / 'config.json').read_text())
        self.metrics = json.loads((self.artifact_dir / 'metrics.json').read_text())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifact_dir / 'tokenizer')
        self.scaler = joblib.load(self.artifact_dir / 'metadata_scaler.joblib')
        self.background = np.load(self.artifact_dir / 'shap_background.npy')
        self.model = MultimodalPhishingModel(
            str(self.artifact_dir / 'text_encoder'), len(FEATURE_NAMES), freeze_text_encoder=True,
        )
        head_state = torch.load(self.artifact_dir / 'multimodal_head.pt', map_location='cpu', weights_only=True)
        missing, unexpected = self.model.load_state_dict(head_state, strict=False)
        if unexpected or any(not key.startswith('text_encoder.') for key in missing):
            raise RuntimeError(f'Invalid model artifact: missing={missing}, unexpected={unexpected}')
        self.model.to(self.device).eval()

    @classmethod
    def artifacts_available(cls, artifact_dir='model'):
        root = Path(artifact_dir)
        required = [
            'config.json', 'metrics.json', 'metadata_scaler.joblib', 'shap_background.npy',
            'multimodal_head.pt', 'tokenizer/tokenizer.json', 'text_encoder/config.json',
        ]
        return all((root / name).exists() for name in required)

    def _inputs(self, sender, subject, body, urls, receiver=''):
        text = f'{subject} [SEP] {body}'
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.config['max_length'], return_tensors='pt',
        ).to(self.device)
        raw_metadata = feature_vector(
            sender=sender, receiver=receiver, subject=subject, body=body, urls=urls,
        )
        scaled = self.scaler.transform(raw_metadata.reshape(1, -1)).astype(np.float32)
        return tokens, raw_metadata, scaled

    def _metadata_shap(self, text_embedding, scaled_metadata):
        import shap

        def predict(values):
            values = np.asarray(values, dtype=np.float32)
            metadata = torch.tensor(values, device=self.device)
            text = text_embedding.repeat(len(values), 1)
            with torch.no_grad():
                logits = self.model.classify_embeddings(text, metadata)
                return torch.sigmoid(logits).cpu().numpy()

        explainer = shap.KernelExplainer(predict, self.background)
        values = np.asarray(explainer.shap_values(scaled_metadata, nsamples=64, silent=True)).reshape(-1)
        ranked = sorted(zip(FEATURE_NAMES, values), key=lambda item: abs(item[1]), reverse=True)
        top = ranked[:8]
        total = sum(abs(float(value)) for _, value in top) or 1.0
        return [
            {
                'feature': name.replace('_', ' ').title(),
                'value': round(float(value), 4),
                'contribution': round(abs(float(value)) / total, 4),
                'direction': 'phishing' if value >= 0 else 'legitimate',
            }
            for name, value in top
        ]

    def predict(self, sender, subject, body, urls='', receiver=''):
        tokens, _, scaled = self._inputs(sender, subject, body, urls, receiver)
        with torch.no_grad():
            text_embedding, attentions = self.model.encode_text(
                tokens['input_ids'], tokens['attention_mask'], output_attentions=True,
            )
            metadata = torch.tensor(scaled, device=self.device)
            probability = torch.sigmoid(self.model.classify_embeddings(text_embedding, metadata)).item()

        token_ids = tokens['input_ids'][0]
        tokens_text = self.tokenizer.convert_ids_to_tokens(token_ids)
        attention_scores = attentions[-1][0, :, 0, :].mean(dim=0).cpu().numpy()
        attention_items = []
        for token, score in sorted(zip(tokens_text, attention_scores), key=lambda item: item[1], reverse=True):
            if token in self.tokenizer.all_special_tokens or token.startswith('##') or not any(char.isalnum() for char in token):
                continue
            attention_items.append({
                'text': token, 'reason': f'DistilBERT attention weight: {float(score):.3f}',
                'attention': round(float(score), 4),
            })
            if len(attention_items) == 10:
                break

        threshold = float(self.config.get('threshold', 0.5))
        return {
            'prediction': 'Phishing' if probability >= threshold else 'Legitimate',
            'probability': round(probability * 100, 2),
            'confidence': 'High' if abs(probability - threshold) > 0.3 else 'Medium' if abs(probability - threshold) > 0.15 else 'Low',
            'text_explanations': attention_items,
            'feature_contributions': self._metadata_shap(text_embedding, scaled),
            'engine': 'distilbert-multimodal',
            'model_metrics': self.metrics.get('test', {}),
        }
