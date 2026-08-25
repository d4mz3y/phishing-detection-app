import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from phishing_model.features import FEATURE_NAMES, feature_vector
from phishing_model.model import MultimodalPhishingModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train the multimodal phishing detector.')
    parser.add_argument('--data', default='data/raw/dataset/Nazario_5.csv')
    parser.add_argument('--output', default='model')
    parser.add_argument('--model-name', default='distilbert/distilbert-base-uncased')
    parser.add_argument('--max-samples', type=int, default=None, help='Stratified sample limit for quick CPU runs.')
    parser.add_argument('--max-length', type=int, default=192)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path, max_samples, seed):
    frame = pd.read_csv(path).dropna(subset=['label']).copy()
    for column in ('sender', 'receiver', 'subject', 'body', 'urls'):
        frame[column] = frame.get(column, '').fillna('').astype(str)
    frame['label'] = frame['label'].astype(int)
    frame = frame[frame['label'].isin([0, 1])]
    if max_samples and max_samples < len(frame):
        frame, _ = train_test_split(frame, train_size=max_samples, stratify=frame['label'], random_state=seed)
    return frame.reset_index(drop=True)


def encode_texts(model, tokenizer, texts, max_length, batch_size, device):
    embeddings = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            encoded, _ = model.encode_text(tokens['input_ids'], tokens['attention_mask'])
        embeddings.append(encoded.cpu())
        print(f'Encoded {min(start + batch_size, len(texts))}/{len(texts)}', flush=True)
    return torch.cat(embeddings)


def metrics_for(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(labels, probabilities),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
        'support': int(len(labels)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_data(args.data, args.max_samples, args.seed)
    indices = np.arange(len(frame))
    train_idx, remainder = train_test_split(indices, test_size=0.30, stratify=frame['label'], random_state=args.seed)
    val_idx, test_idx = train_test_split(remainder, test_size=0.50, stratify=frame.iloc[remainder]['label'], random_state=args.seed)

    metadata = np.vstack([
        feature_vector(sender=row.sender, receiver=row.receiver, subject=row.subject, body=row.body, urls=row.urls)
        for row in frame.itertuples()
    ])
    scaler = StandardScaler().fit(metadata[train_idx])
    metadata_scaled = scaler.transform(metadata).astype(np.float32)
    texts = (frame['subject'] + ' [SEP] ' + frame['body']).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalPhishingModel(args.model_name, len(FEATURE_NAMES), freeze_text_encoder=True).to(device)
    text_embeddings = encode_texts(model, tokenizer, texts, args.max_length, args.batch_size, device)
    labels = torch.tensor(frame['label'].to_numpy(), dtype=torch.float32)
    metadata_tensor = torch.tensor(metadata_scaled)

    def loader(selected, shuffle=False):
        dataset = TensorDataset(text_embeddings[selected], metadata_tensor[selected], labels[selected])
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    optimizer = torch.optim.AdamW(
        list(model.text_projection.parameters()) + list(model.metadata_network.parameters()) + list(model.classifier.parameters()),
        lr=args.learning_rate,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_val_loss = float('inf')
    patience = 3
    stale_epochs = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for text_batch, metadata_batch, label_batch in loader(train_idx, shuffle=True):
            optimizer.zero_grad()
            logits = model.classify_embeddings(text_batch.to(device), metadata_batch.to(device))
            loss = criterion(logits, label_batch.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for text_batch, metadata_batch, label_batch in loader(val_idx):
                logits = model.classify_embeddings(text_batch.to(device), metadata_batch.to(device))
                val_losses.append(criterion(logits, label_batch.to(device)).item())
        entry = {'epoch': epoch + 1, 'train_loss': float(np.mean(train_losses)), 'validation_loss': float(np.mean(val_losses))}
        history.append(entry)
        print(entry, flush=True)
        if entry['validation_loss'] < best_val_loss:
            best_val_loss = entry['validation_loss']
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device).eval()
    probabilities = []
    with torch.no_grad():
        for text_batch, metadata_batch, _ in loader(test_idx):
            logits = model.classify_embeddings(text_batch.to(device), metadata_batch.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    test_labels = frame.iloc[test_idx]['label'].to_numpy()
    metrics = metrics_for(test_labels, np.asarray(probabilities))
    source_bytes = Path(args.data).read_bytes()
    config = {
        'model_name': args.model_name, 'metadata_features': FEATURE_NAMES,
        'max_length': args.max_length, 'threshold': 0.5, 'seed': args.seed,
        'dataset_path': str(args.data), 'dataset_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'dataset_rows': len(frame), 'split_rows': {'train': len(train_idx), 'validation': len(val_idx), 'test': len(test_idx)},
        'trained_at': datetime.now(timezone.utc).isoformat(), 'device': str(device),
        'text_encoder_frozen': True,
    }
    head_state = {key: value for key, value in best_state.items() if not key.startswith('text_encoder.')}
    torch.save(head_state, output / 'multimodal_head.pt')
    model.text_encoder.save_pretrained(output / 'text_encoder')
    tokenizer.save_pretrained(output / 'tokenizer')
    joblib.dump(scaler, output / 'metadata_scaler.joblib')
    np.save(output / 'shap_background.npy', metadata_scaled[train_idx[:min(32, len(train_idx))]])
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'metrics.json').write_text(json.dumps({'test': metrics, 'history': history}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
