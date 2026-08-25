"""Regenerate model/text_encoder/, which is git-ignored because it is an
unmodified copy of the public distilbert-base-uncased weights (the encoder
is frozen during training, so nothing project-specific lives in this file).
"""
from transformers import AutoModel

if __name__ == '__main__':
    AutoModel.from_pretrained('distilbert-base-uncased').save_pretrained('model/text_encoder')
    print('Saved frozen encoder to model/text_encoder/')
