import ast
import re
from urllib.parse import urlparse

import numpy as np


FEATURE_NAMES = [
    'url_count', 'unique_domain_count', 'mean_url_length', 'max_url_length',
    'has_ip_url', 'has_suspicious_tld', 'sender_domain_digit_count',
    'has_suspicious_sender', 'subject_length', 'body_length', 'caps_ratio',
    'special_char_count', 'has_urgency_words', 'has_threat_words',
    'receiver_count',
]

SUSPICIOUS_TLDS = {'xyz', 'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'click', 'download', 'link', 'fit'}


def normalize_urls(value, body=''):
    if isinstance(value, (list, tuple)):
        candidates = [str(item) for item in value]
    else:
        text = '' if value is None else str(value)
        try:
            parsed = ast.literal_eval(text)
            candidates = [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            candidates = []
        candidates.extend(re.findall(r'https?://[^\s<>"\']+', text))
    candidates.extend(re.findall(r'https?://[^\s<>"\']+', body or ''))
    return list(dict.fromkeys(url.rstrip('.,;:!?)\'"') for url in candidates if url.startswith(('http://', 'https://'))))


def extract_feature_dict(sender='', receiver='', subject='', body='', urls=''):
    sender = str(sender or '')
    receiver = str(receiver or '')
    subject = str(subject or '')
    body = str(body or '')
    url_list = normalize_urls(urls, body)
    domains = [(urlparse(url).hostname or '').lower() for url in url_list]
    sender_domain = sender.rsplit('@', 1)[-1].lower() if '@' in sender else sender.lower()
    combined = f'{subject} {body}'
    lengths = [len(url) for url in url_list]

    return {
        'url_count': len(url_list),
        'unique_domain_count': len(set(filter(None, domains))),
        'mean_url_length': float(np.mean(lengths)) if lengths else 0.0,
        'max_url_length': max(lengths, default=0),
        'has_ip_url': int(any(re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', domain) for domain in domains)),
        'has_suspicious_tld': int(any(domain.rsplit('.', 1)[-1] in SUSPICIOUS_TLDS for domain in domains)),
        'sender_domain_digit_count': sum(char.isdigit() for char in sender_domain),
        'has_suspicious_sender': int(bool(re.search(r'(secure|verify|account|login|update|confirm)', sender, re.I))),
        'subject_length': len(subject),
        'body_length': len(body),
        'caps_ratio': sum(char.isupper() for char in combined) / max(len(combined), 1),
        'special_char_count': len(re.findall(r'[^\w\s]', combined)),
        'has_urgency_words': int(bool(re.search(r'(urgent|immediately|act now|verify|action required|within \d+ hours)', combined, re.I))),
        'has_threat_words': int(bool(re.search(r'(suspend|limited|compromise|unauthorized|terminate|blocked)', combined, re.I))),
        'receiver_count': max(1, len([part for part in re.split(r'[,;]', receiver) if part.strip()])) if receiver else 0,
    }


def feature_vector(**fields):
    values = extract_feature_dict(**fields)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)

