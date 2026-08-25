import torch
from torch import nn
from transformers import AutoModel


class MultimodalPhishingModel(nn.Module):
    def __init__(self, model_name, metadata_features, freeze_text_encoder=True):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        # Attention matrices are required for token-level explanations; SDPA omits them.
        self.text_encoder.set_attn_implementation('eager')
        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.metadata_network = nn.Sequential(
            nn.Linear(metadata_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 1),
        )

    def encode_text(self, input_ids, attention_mask, output_attentions=False):
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        return output.last_hidden_state[:, 0], output.attentions

    def classify_embeddings(self, text_embedding, metadata):
        text_features = self.text_projection(text_embedding)
        metadata_features = self.metadata_network(metadata)
        return self.classifier(torch.cat((text_features, metadata_features), dim=1)).squeeze(1)

    def forward(self, input_ids, attention_mask, metadata):
        text_embedding, _ = self.encode_text(input_ids, attention_mask)
        return self.classify_embeddings(text_embedding, metadata)
