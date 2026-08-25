document.addEventListener('DOMContentLoaded', function() {
    const emailForm = document.getElementById('emailForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const emailFile = document.getElementById('emailFile');

    if (emailFile) {
        emailFile.addEventListener('change', uploadEmailFile);
    }
    updateEngineStatus();
    
    if (emailForm) {
        emailForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                sender: document.getElementById('sender').value,
                subject: document.getElementById('subject').value,
                body: document.getElementById('body').value,
                urls: document.getElementById('urls').value
            };
            
            if (!formData.sender || !formData.subject || !formData.body) {
                showNotification('Please fill in all required fields', 'error');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.error || 'Analysis failed');
                }

                displayResults(result);
                showNotification('Analysis completed successfully', 'success');
                
            } catch (error) {
                console.error('Error:', error);
                showNotification(error.message || 'Error analyzing email. Please try again.', 'error');
            } finally {
                showLoading(false);
            }
        });
    }
});

async function updateEngineStatus() {
    const note = document.getElementById('engineNote');
    if (!note) return;
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        note.innerHTML = health.model_loaded
            ? '<i class="fas fa-circle-check"></i> Live engine: trained DistilBERT + metadata neural fusion'
            : '<i class="fas fa-circle-info"></i> Live engine: explainable lightweight heuristic (model artifacts not loaded)';
    } catch (_) {
        note.textContent = 'Detection engine status unavailable';
    }
}

async function uploadEmailFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    showLoading(true);
    try {
        const payload = new FormData();
        payload.append('email_file', file);
        const response = await fetch('/upload', { method: 'POST', body: payload });
        const parsed = await response.json();
        if (!response.ok) throw new Error(parsed.error || 'Unable to read email file');

        document.getElementById('sender').value = parsed.sender || '';
        document.getElementById('subject').value = parsed.subject || '';
        document.getElementById('body').value = parsed.body || '';
        document.getElementById('urls').value = parsed.urls || '';
        showNotification('Email loaded. Review the fields, then analyze it.', 'success');
    } catch (error) {
        console.error('Upload error:', error);
        showNotification(error.message || 'Unable to read email file', 'error');
        event.target.value = '';
    } finally {
        showLoading(false);
    }
}

async function loadSample(type) {
    showLoading(true);
    
    try {
        const response = await fetch(`/sample/${type}`);
        if (!response.ok) {
            throw new Error('Unable to load sample');
        }
        const sample = await response.json();
        
        document.getElementById('sender').value = sample.sender || '';
        document.getElementById('subject').value = sample.subject || '';
        document.getElementById('body').value = sample.body || '';
        document.getElementById('urls').value = sample.urls || '';
        
        showNotification(`${type === 'legitimate' ? 'Legitimate' : 'Phishing'} sample loaded`, 'success');
        
        const form = document.getElementById('emailForm');
        form.dispatchEvent(new Event('submit'));
        
    } catch (error) {
        console.error('Error loading sample:', error);
        showNotification('Error loading sample. Please try again.', 'error');
        showLoading(false);
    }
}

function displayResults(result) {
    const placeholderState = document.getElementById('placeholderState');
    const resultsContent = document.getElementById('resultsContent');
    const resultHeader = document.getElementById('resultHeader');
    const resultIcon = document.getElementById('resultIcon');
    const resultPrediction = document.getElementById('resultPrediction');
    const resultConfidence = document.getElementById('resultConfidence');
    const probabilityFill = document.getElementById('probabilityFill');
    const probabilityValue = document.getElementById('probabilityValue');
    const resultEngine = document.getElementById('resultEngine');
    const timestamp = document.getElementById('timestamp');

    const requiredElements = {
        placeholderState, resultsContent, resultHeader, resultIcon,
        resultPrediction, resultConfidence, probabilityFill, probabilityValue
    };
    const missingElements = Object.entries(requiredElements)
        .filter(([, element]) => !element)
        .map(([name]) => name);
    if (missingElements.length) {
        throw new Error(`Result interface is out of date (missing: ${missingElements.join(', ')}). Refresh the page.`);
    }
    
    placeholderState.style.display = 'none';
    resultsContent.style.display = 'block';
    
    const isPhishing = result.prediction === 'Phishing';
    const probabilityClass = isPhishing ? 'phishing' : 'legitimate';
    
    resultHeader.className = `result-header ${probabilityClass}`;
    resultIcon.className = `result-icon ${probabilityClass}`;
    resultIcon.innerHTML = isPhishing 
        ? '<i class="fas fa-exclamation-triangle"></i>' 
        : '<i class="fas fa-check-circle"></i>';
    resultPrediction.textContent = result.prediction;
    resultPrediction.style.color = isPhishing ? '#ef4444' : '#10b981';
    resultConfidence.textContent = `Confidence: ${result.confidence}`;
    const engineLabel = result.engine === 'distilbert-multimodal'
        ? 'DistilBERT + metadata neural fusion'
        : 'Lightweight heuristic';
    if (resultEngine) {
        resultEngine.textContent = `Engine: ${engineLabel}`;
    }
    
    setTimeout(() => {
        probabilityFill.style.width = `${result.probability}%`;
        probabilityFill.className = `probability-fill ${probabilityClass}`;
        probabilityValue.textContent = `${result.probability}%`;
    }, 100);
    
    displayTextExplanations(result.text_explanations);
    displayUrlExplanations(result.url_explanations);
    displayFeatureContributions(result.feature_contributions);
    
    if (timestamp) {
        timestamp.textContent = result.timestamp || 'just now';
    }
    
    resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayTextExplanations(explanations) {
    const container = document.getElementById('textExplanations');
    
    if (!explanations || explanations.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.875rem;">No suspicious text patterns detected</p>';
        return;
    }
    
    container.innerHTML = explanations.map(exp => `
        <div class="explain-item">
            <i class="fas fa-exclamation-circle"></i>
            <div class="explain-text">
                <strong>"${escapeHtml(exp.text)}"</strong>
                <span>${escapeHtml(exp.reason)}</span>
            </div>
        </div>
    `).join('');
}

function displayUrlExplanations(explanations) {
    const container = document.getElementById('urlExplanations');
    
    if (!explanations || explanations.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.875rem;">No suspicious URL patterns detected</p>';
        return;
    }
    
    container.innerHTML = explanations.map(exp => `
        <div class="explain-item">
            <i class="fas fa-link"></i>
            <div class="explain-text">
                <strong>${escapeHtml(exp.url)}</strong>
                <span>${escapeHtml(exp.reason)}</span>
            </div>
        </div>
    `).join('');
}

function displayFeatureContributions(contributions) {
    const container = document.getElementById('featureContributions');
    
    if (!contributions || contributions.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.875rem;">No significant feature contributions</p>';
        return;
    }
    
    container.innerHTML = contributions.map(feature => {
        const direction = feature.direction
            ? ` ${feature.direction === 'phishing' ? '↑ phishing' : '↓ legitimate'}`
            : '';
        return `
        <div class="feature-item">
            <div class="feature-name">${escapeHtml(feature.feature)}</div>
            <div class="feature-bar-container">
                <div class="feature-bar" style="width: ${feature.contribution * 100}%"></div>
            </div>
            <div class="feature-value" title="Raw SHAP value: ${escapeHtml(String(feature.value))}">${(feature.contribution * 100).toFixed(0)}%${escapeHtml(direction)}</div>
        </div>
    `;
    }).join('');
}

function showLoading(show) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (show) {
        loadingOverlay.classList.add('active');
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        }
    } else {
        loadingOverlay.classList.remove('active');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search-plus"></i> Analyze Email';
        }
    }
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        <span>${escapeHtml(message)}</span>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? 'rgba(16, 185, 129, 0.95)' : 'rgba(239, 68, 68, 0.95)'};
        color: white;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 500;
        z-index: 1001;
        animation: slideIn 0.3s ease;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
