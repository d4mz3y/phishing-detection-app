from flask import Flask, render_template, request, jsonify
import os
import re
import json
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html import unescape
from urllib.parse import urlparse

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-change-me')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL_DIR = os.environ.get('MODEL_DIR', 'model')
_model_engine = None
_model_error = None


def model_artifacts_available():
    required = ('config.json', 'metrics.json', 'metadata_scaler.joblib', 'multimodal_head.pt')
    return all(os.path.exists(os.path.join(MODEL_DIR, name)) for name in required)


def get_model_engine():
    global _model_engine, _model_error
    if _model_engine is None and _model_error is None and model_artifacts_available():
        try:
            from phishing_model.inference import PhishingInferenceEngine
            _model_engine = PhishingInferenceEngine(MODEL_DIR)
        except Exception as exc:
            app.logger.exception('Unable to load trained model artifacts')
            _model_error = str(exc)
    return _model_engine

SAMPLE_LEGITIMATE = {
    'sender': 'hr@company.com',
    'subject': 'Team Building Event - Next Friday',
    'body': 'Hi Team,\n\nI hope this email finds you well. I wanted to remind everyone about our team building event scheduled for next Friday at 3:00 PM in the main conference room.\n\nWe will have various activities and light refreshments. Please let me know if you have any dietary restrictions.\n\nBest regards,\nHR Department',
    'urls': 'https://company.com/events'
}

SAMPLE_PHISHING = {
    'sender': 'security@paypa1-secure.com',
    'subject': 'Urgent: Your Account Has Been Compromised',
    'body': 'Dear Valued Customer,\n\nWe have detected unusual activity on your account. Your account has been temporarily limited. To restore access, please verify your identity immediately by clicking the link below:\n\nhttp://paypa1-secure.com/verify?id=8823\n\nFailure to verify within 24 hours will result in permanent account suspension.\n\nPayPal Security Team',
    'urls': 'http://paypa1-secure.com/verify?id=8823'
}

ALLOWED_EMAIL_EXTENSIONS = {'.eml', '.txt'}


def parse_uploaded_email(file_storage):
    """Convert an RFC 5322 .eml file (or plain-text message) into form fields."""
    filename = (file_storage.filename or '').lower()
    extension = os.path.splitext(filename)[1]
    if extension not in ALLOWED_EMAIL_EXTENSIONS:
        raise ValueError('Upload an .eml or .txt email file')

    raw = file_storage.read()
    if not raw:
        raise ValueError('The uploaded file is empty')

    message = BytesParser(policy=policy.default).parsebytes(raw)
    sender_header = str(message.get('From', '')).strip()
    sender = parseaddr(sender_header)[1] or sender_header
    subject = str(message.get('Subject', '')).strip()
    body_parts = []

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_disposition() == 'attachment':
                continue
            if part.get_content_type() in {'text/plain', 'text/html'}:
                try:
                    content = str(part.get_content())
                    if part.get_content_type() == 'text/html':
                        content = unescape(re.sub(r'<[^>]+>', ' ', content))
                    body_parts.append(content)
                except (LookupError, UnicodeError):
                    continue
    else:
        try:
            content = str(message.get_content())
            if message.get_content_type() == 'text/html':
                content = unescape(re.sub(r'<[^>]+>', ' ', content))
            body_parts.append(content)
        except (LookupError, UnicodeError):
            body_parts.append(raw.decode('utf-8', errors='replace'))

    body = '\n'.join(str(part) for part in body_parts).strip()
    # A headerless .txt file is treated as the message body.
    if extension == '.txt' and not body:
        body = raw.decode('utf-8', errors='replace').strip()
    urls = '\n'.join(dict.fromkeys(
        url.rstrip('.,;:!?)') for url in re.findall(r'https?://[^\s<>"\']+', body)
    ))

    return {'sender': sender, 'subject': subject, 'body': body, 'urls': urls}

def extract_metadata_features(sender, subject, body, urls):
    features = {}
    
    features['url_count'] = len(re.findall(r'https?://[^\s]+', urls)) if urls else 0
    features['url_length'] = len(urls) if urls else 0
    features['has_ip_url'] = 1 if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', urls or '') else 0
    suspicious_tlds = {'xyz', 'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'click', 'download', 'link', 'fit'}
    features['has_suspicious_tld'] = int(any(
        (urlparse(url).hostname or '').rsplit('.', 1)[-1].lower() in suspicious_tlds
        for url in re.findall(r'https?://[^\s]+', urls or '')
    ))
    features['subject_length'] = len(subject) if subject else 0
    features['body_length'] = len(body) if body else 0
    features['sender_domain'] = sender.split('@')[-1] if '@' in sender else ''
    features['has_suspicious_sender'] = 1 if re.search(r'(secure|verify|account|login|update|confirm)', sender, re.IGNORECASE) else 0
    features['has_urgency_words'] = 1 if re.search(r'(urgent|immediately|suspend|limited|verify|action required|within \d+ hours)', subject + ' ' + body, re.IGNORECASE) else 0
    features['has_threat_words'] = 1 if re.search(r'(suspend|limited|compromise|unauthorized|suspended|terminate)', subject + ' ' + body, re.IGNORECASE) else 0
    
    suspicious_chars = len(re.findall(r'[!$%^&*()_+={}\[\]:;\'\"<>,.?/\\|`~]', subject + ' ' + body))
    features['special_char_count'] = suspicious_chars
    features['caps_ratio'] = sum(1 for c in subject + ' ' + body if c.isupper()) / max(len(subject + body), 1)
    
    return features

def analyze_text_explainability(body, subject):
    explanations = []
    text = subject + ' ' + body
    
    urgency_patterns = [
        (r'(urgent|immediately|right away|as soon as possible|act now|don\'t delay)', 'Urgency language detected'),
        (r'(verify|verification|confirm|confirm your)', 'Verification request detected'),
        (r'(suspend|suspended|limited|restricted|blocked|terminate)', 'Account threat language detected'),
        (r'(password|credential|login|sign in|authenticate)', 'Credential request detected'),
        (r'(click here|click the link|follow this link|download)', 'Direct action instruction detected'),
        (r'(within \d+ hours|within \d+ days|expires|expiration)', 'Time pressure tactic detected'),
        (r'(dear (?:valued|customer|user|member|sir|madam))', 'Impersonal greeting detected'),
        (r'(congratulations|you have (?:won|been selected|been chosen))', 'Prize/ reward tactic detected'),
        (r'(bank|paypal|amazon|netflix|apple|microsoft|google|facebook|instagram)', 'Brand impersonation detected'),
        (r'(update your|provide your|enter your|confirm your details)', 'Information solicitation detected')
    ]
    
    for pattern, explanation in urgency_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            explanations.append({
                'text': match.group(),
                'reason': explanation,
                'start': match.start(),
                'end': match.end()
            })
    
    return explanations

def analyze_url_explainability(urls):
    url_explanations = []
    if not urls:
        return url_explanations
    
    url_list = re.findall(r'https?://[^\s]+', urls)
    for url in url_list:
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            url_explanations.append({'url': url, 'reason': 'IP address used instead of domain name'})
        
        domain = re.search(r'https?://([^/]+)', url)
        if domain:
            domain = domain.group(1)
            if re.search(r'(paypa1|g00gle|amaz0n|faceb00k|micros0ft)', domain, re.IGNORECASE):
                url_explanations.append({'url': url, 'reason': 'Suspicious domain spelling (typosquatting)'})
            if re.search(r'\.(xyz|tk|ml|ga|cf|gq|top|click|download|link|fit)$', domain):
                url_explanations.append({'url': url, 'reason': 'Suspicious top-level domain'})
            if re.search(r'(secure|verify|account|login|update|confirm)', domain, re.IGNORECASE):
                url_explanations.append({'url': url, 'reason': 'Suspicious domain keywords'})
        
        if len(url) > 100:
            url_explanations.append({'url': url, 'reason': 'Unusually long URL (potential obfuscation)'})
    
    return url_explanations

def predict_phishing(features):
    weights = {
        'url_count': 0.15,
        'has_ip_url': 0.25,
        'has_suspicious_tld': 0.20,
        'has_suspicious_sender': 0.15,
        'has_urgency_words': 0.15,
        'has_threat_words': 0.10,
        'caps_ratio': 0.05,
        'special_char_count': 0.05,
        'subject_length': 0.02,
        'body_length': 0.02,
        'url_length': 0.01
    }
    active = []

    if features['url_count'] > 0:
        active.append('url_count')
    if features['has_ip_url']:
        active.append('has_ip_url')
    if features['has_suspicious_tld']:
        active.append('has_suspicious_tld')
    if features['has_suspicious_sender']:
        active.append('has_suspicious_sender')
    if features['has_urgency_words']:
        active.append('has_urgency_words')
    if features['has_threat_words']:
        active.append('has_threat_words')
    if features['caps_ratio'] > 0.3:
        active.append('caps_ratio')
    if features['special_char_count'] > 10:
        active.append('special_char_count')
    if features['subject_length'] > 100:
        active.append('subject_length')
    if features['body_length'] < 50 or features['body_length'] > 5000:
        active.append('body_length')

    score = sum(weights[name] for name in active)
    probability = min(max(score, 0.0), 1.0)
    is_phishing = probability >= 0.5

    contributions = [
        {
            'feature': name.replace('_', ' ').title(),
            'value': features[name],
            'contribution': weights[name],
        }
        for name in active
    ]
    contributions.sort(key=lambda item: item['contribution'], reverse=True)
    return is_phishing, probability, contributions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    fields = {name: data.get(name, '') for name in ('sender', 'subject', 'body', 'urls')}
    if any(not isinstance(value, str) for value in fields.values()):
        return jsonify({'error': 'sender, subject, body, and urls must be strings'}), 400

    sender, subject, body, urls = (fields[name].strip() for name in ('sender', 'subject', 'body', 'urls'))
    missing = [name for name, value in (('sender', sender), ('subject', subject), ('body', body)) if not value]
    if missing:
        return jsonify({'error': f"Missing required fields: {', '.join(missing)}"}), 400

    body_urls = re.findall(r'https?://[^\s<>"\']+', body)
    urls = '\n'.join(dict.fromkeys(filter(None, [urls, *body_urls])))

    try:
        
        features = extract_metadata_features(sender, subject, body, urls)
        engine = get_model_engine()
        if engine is not None:
            model_result = engine.predict(sender=sender, subject=subject, body=body, urls=urls)
            is_phishing = model_result['prediction'] == 'Phishing'
            probability = model_result['probability'] / 100
            feature_contributions = model_result['feature_contributions']
        else:
            is_phishing, probability, feature_contributions = predict_phishing(features)
        
        text_explanations = analyze_text_explainability(body, subject)
        url_explanations = analyze_url_explainability(urls)
        
        result = {
            'prediction': 'Phishing' if is_phishing else 'Legitimate',
            'probability': round(probability * 100, 2),
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.15 else 'Low',
            'text_explanations': model_result['text_explanations'] if engine is not None else text_explanations,
            'url_explanations': url_explanations,
            'feature_contributions': feature_contributions[:8],
            'features': features,
            'engine': model_result['engine'] if engine is not None else 'heuristic',
            'model_metrics': model_result.get('model_metrics', {}) if engine is not None else {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
    
    except Exception:
        app.logger.exception('Email analysis failed')
        return jsonify({'error': 'Unable to analyze this email'}), 500

@app.route('/sample/legitimate')
def get_legitimate_sample():
    return jsonify(SAMPLE_LEGITIMATE)

@app.route('/sample/phishing')
def get_phishing_sample():
    return jsonify(SAMPLE_PHISHING)


@app.route('/upload', methods=['POST'])
def upload_email():
    uploaded = request.files.get('email_file')
    if uploaded is None or not uploaded.filename:
        return jsonify({'error': 'Choose an .eml or .txt email file'}), 400
    try:
        return jsonify(parse_uploaded_email(uploaded))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results')
def results():
    metrics = None
    config = None
    try:
        with open(os.path.join(MODEL_DIR, 'metrics.json'), encoding='utf-8') as handle:
            metrics = json.load(handle).get('test')
        with open(os.path.join(MODEL_DIR, 'config.json'), encoding='utf-8') as handle:
            config = json.load(handle)
    except (FileNotFoundError, OSError, ValueError):
        pass
    if metrics and len(metrics.get('confusion_matrix', [])) == 2:
        tn, fp = metrics['confusion_matrix'][0]
        fn, tp = metrics['confusion_matrix'][1]
        def class_values(correct, predicted_total, actual_total):
            precision = correct / predicted_total if predicted_total else 0
            recall = correct / actual_total if actual_total else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            return {'precision': precision, 'recall': recall, 'f1': f1, 'support': actual_total}
        metrics['class_report'] = {
            'legitimate': class_values(tn, tn + fn, tn + fp),
            'phishing': class_values(tp, tp + fp, tp + fn),
        }
    return render_template('results.html', live_metrics=metrics, model_config=config)

@app.route('/api/health')
def health():
    engine = get_model_engine()
    payload = {
        'status': 'healthy' if _model_error is None else 'degraded',
        'engine': 'distilbert-multimodal' if engine is not None else 'heuristic',
        'model_loaded': engine is not None,
    }
    if _model_error:
        payload['model_error'] = _model_error
    return jsonify(payload), 200 if _model_error is None else 503

if __name__ == '__main__':
    app.run(
        debug=os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'},
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', '5000')),
    )
