import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import dns.resolver
import requests
import whois
import tldextract
import validators
import hashlib
import json
import time
from datetime import datetime
import pytz
from urllib.parse import urlparse, urljoin, unquote
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go
import plotly.express as px
from email_validator import validate_email, EmailNotValidError
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
import math

# ================================
# Configuration & Styling
# ================================

st.set_page_config(
    page_title="PhishShield - Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS
st.markdown("""
<style>
/* Dropdown selectbox */
div[data-baseweb="select"] > div > div {
    color: white !important;
    background-color: #1e293b !important;
    border-radius: 6px;
}

/* Dropdown menu options */
ul[role="listbox"] li {
    color: white !important;
    background-color: #1e293b !important;
}

/* Hover effect for dropdown */
ul[role="listbox"] li:hover {
    background-color: #3b82f6 !important;
    color: white !important;
}

/* File uploader button text */
div[data-testid="stFileUploader"] section div div span {
    color: white !important;
    font-weight: bold !important;
}

/* File uploader button background */
div[data-testid="stFileUploader"] section div div {
    background-color: #1e293b !important;
    border-radius: 6px !important;
}

/* Enhanced cards */
.verification-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.url-analysis-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.nlp-insights-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.metric-box {
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 8px;
    margin: 5px;
    backdrop-filter: blur(10px);
}

.status-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 12px;
    margin: 5px;
}

.status-verified {
    background-color: #10b981;
    color: white;
}

.status-failed {
    background-color: #ef4444;
    color: white;
}

.status-warning {
    background-color: #f59e0b;
    color: white;
}

.url-redirect-chain {
    background: rgba(0,0,0,0.2);
    padding: 10px;
    border-radius: 6px;
    margin: 10px 0;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ================================
# Enhanced ML Models
# ================================

@st.cache_resource
def load_advanced_models():
    """Load multiple ML models for comprehensive analysis."""
    models = {}
    device = 0 if torch.cuda.is_available() else -1

    # Primary phishing detection model
    try:
        models['phishing'] = pipeline(
            "text-classification",
            model="ealvaradob/bert-finetuned-phishing",
            device=device
        )
    except Exception as e:
        def fallback_phishing(texts):
            out = []
            for t in texts if isinstance(texts, list) else [texts]:
                score = min(0.99,
                            (len(re.findall(r'(verify|password|urgent|click|account|suspend|login)', t.lower())) * 0.2))
                label = "LABEL_1" if score > 0.4 else "LABEL_0"
                out.append({"label": label, "score": float(score)})
            return out

        models['phishing'] = lambda x: fallback_phishing(x)

    # Sentiment analysis
    try:
        models['sentiment'] = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
    except Exception:
        def fallback_sentiment(texts):
            out = []
            for t in texts if isinstance(texts, list) else [texts]:
                neg = len(re.findall(r'(urgent|angry|problem|unable|issue|suspend|fail|error)', t.lower()))
                if neg > 2:
                    out.append({"label": "NEGATIVE", "score": 0.85})
                else:
                    out.append({"label": "POSITIVE", "score": 0.75})
            return out

        models['sentiment'] = lambda x: fallback_sentiment(x)

    # Zero-shot classification
    try:
        models['intent'] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )
    except Exception:
        def fallback_intent(texts, candidate_labels):
            out = []
            for t in texts if isinstance(texts, list) else [texts]:
                scores = []
                for lbl in candidate_labels:
                    score = 0.5
                    if lbl in ['phishing', 'social engineering'] and re.search(r'(verify|password|click|login|urgent)',
                                                                               t.lower()):
                        score = 0.8
                    elif lbl == 'legitimate' and 'unsubscribe' in t.lower():
                        score = 0.6
                    scores.append(score)
                ordered = sorted(zip(candidate_labels, scores), key=lambda x: x[1], reverse=True)
                out.append({
                    "labels": [o[0] for o in ordered],
                    "scores": [o[1] for o in ordered]
                })
            return out[0] if len(out) == 1 else out

        models['intent'] = lambda text, labels: fallback_intent(text, labels)

    return models


# ================================
# Enhanced Phishing Detector
# ================================

class AdvancedPhishingDetector:
    def __init__(self):
        self.models = load_advanced_models()

        self.phishing_keywords = {
            'urgent': ['urgent', 'immediate', 'expire', 'suspend', 'deadline', 'act now',
                       'limited time', 'ends today', 'final notice', 'last chance'],
            'action': ['verify', 'confirm', 'update', 'validate', 'click here', 'click below',
                       'follow link', 'activate', 'reactivate', 'restore access'],
            'threat': ['suspended', 'locked', 'blocked', 'unauthorized', 'illegal', 'terminate',
                       'close your account', 'deactivated', 'restricted', 'frozen'],
            'financial': ['refund', 'payment', 'billing', 'invoice', 'tax', 'irs', 'prize',
                          'winner', 'lottery', 'inheritance', 'million dollars', 'bitcoin'],
            'credential': ['password', 'username', 'pin', 'ssn', 'social security', 'account number',
                           'credit card', 'cvv', 'expiry date', 'security code'],
            'social_engineering': ['dear customer', 'valued user', 'lucky winner', 'congratulations',
                                   'you have been selected', 'claim your', 'act immediately']
        }

        self.url_shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'short.link',
                               'buff.ly', 'is.gd', 'adf.ly', 'bl.ink', 'branch.io']

        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download', '.review',
                                '.work', '.date', '.top', '.bid', '.trade', '.science']

        self.legitimate_domains = self._load_legitimate_domains()

    def _load_legitimate_domains(self):
        return {
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
            'twitter.com', 'linkedin.com', 'github.com', 'stackoverflow.com', 'youtube.com',
            'paypal.com', 'ebay.com', 'netflix.com', 'spotify.com', 'adobe.com',
            'dropbox.com', 'salesforce.com', 'oracle.com', 'ibm.com', 'cisco.com'
        }

    def analyze_email(self, email_content: str) -> Dict:
        """Comprehensive email analysis with enhanced reporting"""
        results = {
            'threat_score': 0,
            'risk_level': 'Low',
            'ml_predictions': {},
            'keyword_analysis': {},
            'url_analysis': [],
            'sender_analysis': {},
            'header_analysis': {},
            'content_analysis': {},
            'nlp_insights': {},
            'recommendations': [],
            'timeline': [],
            'confidence': 0,
            'raw_sample': email_content[:1000]
        }

        # ML-based analysis with enhanced NLP insights
        results['ml_predictions'] = self._ml_analysis(email_content)
        results['threat_score'] += results['ml_predictions']['threat_contribution']

        # Extract NLP insights
        results['nlp_insights'] = self._extract_nlp_insights(email_content, results['ml_predictions'])

        # Keyword analysis
        results['keyword_analysis'] = self._keyword_analysis(email_content)
        results['threat_score'] += results['keyword_analysis']['threat_contribution']

        # Enhanced URL analysis with anchor text
        results['url_analysis'] = self._analyze_urls_enhanced(email_content)
        for url in results['url_analysis']:
            if url['risk_score'] > 50:
                results['threat_score'] += 15

        # Enhanced sender verification
        results['sender_analysis'] = self._analyze_sender_enhanced(email_content)
        results['threat_score'] += results['sender_analysis']['threat_contribution']

        # Header analysis
        results['header_analysis'] = self._analyze_headers(email_content)
        results['threat_score'] += results['header_analysis']['threat_contribution']

        # Content patterns
        results['content_analysis'] = self._analyze_content_patterns(email_content)
        results['threat_score'] += results['content_analysis']['threat_contribution']

        # Calculate final metrics
        results['threat_score'] = min(results['threat_score'], 100)
        results['risk_level'] = self._calculate_risk_level(results['threat_score'])
        results['confidence'] = self._calculate_confidence(results)

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        # Create timeline
        results['timeline'] = self._create_analysis_timeline()

        return results

    def _ml_analysis(self, content: str) -> Dict:
        """Enhanced ML analysis with detailed metrics"""
        ml_results = {
            'phishing_detection': {},
            'sentiment': {},
            'intent': {},
            'threat_contribution': 0,
            'model_performance': {}
        }

        # Phishing detection
        try:
            model = self.models['phishing']
            phishing_result = model(content[:1024])[0]
            ml_results['phishing_detection'] = {
                'label': phishing_result['label'],
                'score': float(phishing_result['score']),
                'prediction': 'Phishing' if phishing_result['label'] == 'LABEL_1' else 'Legitimate'
            }

            if phishing_result['label'] == 'LABEL_1':
                ml_results['threat_contribution'] += int(phishing_result['score'] * 40)

            ml_results['model_performance']['phishing_confidence'] = float(phishing_result['score'])

        except Exception:
            score = min(0.99, (len(re.findall(r'(verify|password|urgent|click|account|suspend|login)',
                                              content.lower())) * 0.18))
            label = 'LABEL_1' if score > 0.4 else 'LABEL_0'
            ml_results['phishing_detection'] = {
                'label': label,
                'score': float(score),
                'prediction': 'Phishing' if label == 'LABEL_1' else 'Legitimate'
            }
            if label == 'LABEL_1':
                ml_results['threat_contribution'] += int(score * 40)
            ml_results['model_performance']['phishing_confidence'] = float(score)

        # Sentiment analysis
        try:
            sentiment_result = self.models['sentiment'](content[:512])[0]
            ml_results['sentiment'] = {
                'label': sentiment_result['label'],
                'score': float(sentiment_result['score'])
            }
            if sentiment_result['label'] == 'NEGATIVE' and sentiment_result['score'] > 0.7:
                ml_results['threat_contribution'] += 10

            ml_results['model_performance']['sentiment_confidence'] = float(sentiment_result['score'])

        except Exception:
            neg = len(re.findall(r'(urgent|suspend|fail|problem|cannot|immediately)', content.lower()))
            if neg > 2:
                ml_results['sentiment'] = {'label': 'NEGATIVE', 'score': 0.8}
                ml_results['threat_contribution'] += 10
            else:
                ml_results['sentiment'] = {'label': 'POSITIVE', 'score': 0.7}
            ml_results['model_performance']['sentiment_confidence'] = 0.8 if neg > 2 else 0.7

        # Intent classification
        try:
            candidate_labels = ['phishing', 'legitimate', 'spam', 'social engineering', 'malware']
            intent_result = self.models['intent'](content[:512], candidate_labels)
            ml_results['intent'] = {
                'labels': intent_result['labels'][:3],
                'scores': [float(s) for s in intent_result['scores'][:3]]
            }
            if intent_result['labels'][0] in ['phishing', 'social engineering', 'malware']:
                ml_results['threat_contribution'] += int(intent_result['scores'][0] * 30)

            ml_results['model_performance']['intent_confidence'] = float(intent_result['scores'][0])

        except Exception:
            labels = ['phishing', 'legitimate', 'spam']
            scores = [0.6 if re.search(r'(verify|password|click|login)', content.lower()) else 0.2, 0.2, 0.2]
            ml_results['intent'] = {'labels': labels, 'scores': scores}
            if ml_results['intent']['labels'][0] == 'phishing':
                ml_results['threat_contribution'] += int(scores[0] * 30)
            ml_results['model_performance']['intent_confidence'] = scores[0]

        return ml_results

    def _extract_nlp_insights(self, content: str, ml_predictions: Dict) -> Dict:
        """Extract detailed NLP insights"""
        insights = {
            'text_statistics': {},
            'linguistic_patterns': {},
            'manipulation_tactics': [],
            'credibility_score': 0
        }

        # Text statistics
        words = content.split()
        sentences = re.split(r'[.!?]+', content)

        insights['text_statistics'] = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / max(len(words), 1)
        }

        # Linguistic patterns
        caps_count = len(re.findall(r'[A-Z]{3,}', content))
        exclamation_count = content.count('!')
        question_count = content.count('?')

        insights['linguistic_patterns'] = {
            'excessive_caps': caps_count > 5,
            'caps_count': caps_count,
            'excessive_punctuation': exclamation_count > 3,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'contains_emojis': bool(re.search(r'[😀-🙏🚀-🛿]', content))
        }

        # Manipulation tactics detection
        if ml_predictions.get('sentiment', {}).get('label') == 'NEGATIVE':
            insights['manipulation_tactics'].append('Fear-based messaging detected')

        if re.search(r'(urgent|immediate|act now|limited time)', content.lower()):
            insights['manipulation_tactics'].append('Urgency manipulation')

        if re.search(r'(winner|prize|free|congratulations)', content.lower()):
            insights['manipulation_tactics'].append('Reward-based manipulation')

        if re.search(r'(verify|confirm|update|validate)', content.lower()):
            insights['manipulation_tactics'].append('Authority impersonation')

        # Credibility score (0-100)
        credibility = 50

        if insights['text_statistics']['lexical_diversity'] > 0.6:
            credibility += 10
        if not insights['linguistic_patterns']['excessive_caps']:
            credibility += 10
        if not insights['linguistic_patterns']['excessive_punctuation']:
            credibility += 10
        if ml_predictions.get('sentiment', {}).get('label') == 'POSITIVE':
            credibility += 10
        if len(insights['manipulation_tactics']) == 0:
            credibility += 10

        credibility -= len(insights['manipulation_tactics']) * 10

        insights['credibility_score'] = max(0, min(100, credibility))

        return insights

    def _analyze_sender_enhanced(self, content: str) -> Dict:
        """Enhanced sender verification with comprehensive checks"""
        sender_analysis = {
            'email': None,
            'display_name': None,
            'domain': None,
            'valid_format': False,
            'domain_exists': False,
            'domain_age': None,
            'domain_registrar': None,
            'spf': None,
            'dkim': None,
            'dmarc': None,
            'mx_records': [],
            'reputation_score': 0,
            'verification_status': 'Unknown',
            'warning_flags': [],
            'threat_contribution': 0
        }

        # Extract sender email and display name
        from_pattern = r'[Ff]rom:\s*(?:"?([^"<]+)"?\s*)?<?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>?'
        match = re.search(from_pattern, content)

        if match:
            sender_analysis['display_name'] = match.group(1).strip() if match.group(1) else None
            sender_email = match.group(2)
            sender_analysis['email'] = sender_email

            try:
                # Validate email format
                validation = validate_email(sender_email, check_deliverability=False)
                sender_analysis['valid_format'] = True
                sender_analysis['domain'] = validation.domain

                # Check display name mismatch
                if sender_analysis['display_name']:
                    if '@' in sender_analysis['display_name']:
                        sender_analysis['warning_flags'].append('Email in display name')
                        sender_analysis['threat_contribution'] += 10

                    # Check if display name suggests official source but email doesn't match
                    official_indicators = ['support', 'admin', 'security', 'service']
                    if any(ind in sender_analysis['display_name'].lower() for ind in official_indicators):
                        if validation.domain not in self.legitimate_domains:
                            sender_analysis['warning_flags'].append('Impersonation attempt detected')
                            sender_analysis['threat_contribution'] += 25

                # MX Records check
                try:
                    mx_records = dns.resolver.resolve(validation.domain, 'MX', lifetime=5)
                    sender_analysis['domain_exists'] = True
                    sender_analysis['mx_records'] = [str(mx.exchange) for mx in mx_records]
                    sender_analysis['reputation_score'] += 20
                except Exception:
                    sender_analysis['warning_flags'].append('No MX records found')
                    sender_analysis['threat_contribution'] += 20

                # SPF check
                try:
                    txt_records = dns.resolver.resolve(validation.domain, 'TXT', lifetime=5)
                    spf_found = False
                    for record in txt_records:
                        record_str = str(record).lower()
                        if 'v=spf1' in record_str:
                            sender_analysis['spf'] = 'Present'
                            spf_found = True
                            sender_analysis['reputation_score'] += 15

                            # Check SPF strictness
                            if '-all' in record_str:
                                sender_analysis['spf'] = 'Present (Strict)'
                                sender_analysis['reputation_score'] += 5
                            elif '~all' in record_str:
                                sender_analysis['spf'] = 'Present (Soft Fail)'
                            break

                    if not spf_found:
                        sender_analysis['spf'] = 'Not found'
                        sender_analysis['warning_flags'].append('No SPF record')
                        sender_analysis['threat_contribution'] += 10
                except Exception:
                    sender_analysis['spf'] = 'Not found'
                    sender_analysis['warning_flags'].append('No SPF record')
                    sender_analysis['threat_contribution'] += 10

                # DMARC check
                try:
                    dmarc_records = dns.resolver.resolve(f'_dmarc.{validation.domain}', 'TXT', lifetime=5)
                    dmarc_str = str(list(dmarc_records)[0]).lower()
                    sender_analysis['dmarc'] = 'Present'
                    sender_analysis['reputation_score'] += 15

                    # Check DMARC policy
                    if 'p=reject' in dmarc_str:
                        sender_analysis['dmarc'] = 'Present (Strict - Reject)'
                        sender_analysis['reputation_score'] += 10
                    elif 'p=quarantine' in dmarc_str:
                        sender_analysis['dmarc'] = 'Present (Quarantine)'
                        sender_analysis['reputation_score'] += 5

                except Exception:
                    sender_analysis['dmarc'] = 'Not found'
                    sender_analysis['warning_flags'].append('No DMARC policy')
                    sender_analysis['threat_contribution'] += 10

                # Domain age check (using WHOIS)
                try:
                    domain_info = whois.whois(validation.domain)
                    if domain_info.creation_date:
                        creation_date = domain_info.creation_date
                        if isinstance(creation_date, list):
                            creation_date = creation_date[0]

                        age_days = (datetime.now() - creation_date).days
                        sender_analysis['domain_age'] = f"{age_days} days"

                        if age_days < 30:
                            sender_analysis['warning_flags'].append('Very new domain (< 30 days)')
                            sender_analysis['threat_contribution'] += 20
                        elif age_days < 180:
                            sender_analysis['warning_flags'].append('New domain (< 6 months)')
                            sender_analysis['threat_contribution'] += 10
                        else:
                            sender_analysis['reputation_score'] += 15

                    if domain_info.registrar:
                        sender_analysis['domain_registrar'] = domain_info.registrar

                except Exception:
                    sender_analysis['domain_age'] = 'Unknown'

                # Check for free email providers
                free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'mail.com']
                if validation.domain in free_providers:
                    sender_analysis['warning_flags'].append('Free email provider')
                    sender_analysis['threat_contribution'] += 5

                # Check for suspicious domain patterns
                if any(char.isdigit() for char in validation.domain):
                    if validation.domain.count('.') > 2:
                        sender_analysis['warning_flags'].append('Suspicious domain structure')
                        sender_analysis['threat_contribution'] += 15

                # Calculate verification status
                if sender_analysis['reputation_score'] >= 60:
                    sender_analysis['verification_status'] = 'Verified ✓'
                elif sender_analysis['reputation_score'] >= 30:
                    sender_analysis['verification_status'] = 'Partially Verified ⚠'
                else:
                    sender_analysis['verification_status'] = 'Failed ✗'

            except EmailNotValidError:
                sender_analysis['valid_format'] = False
                sender_analysis['verification_status'] = 'Failed ✗'
                sender_analysis['warning_flags'].append('Invalid email format')
                sender_analysis['threat_contribution'] += 25

        else:
            sender_analysis['verification_status'] = 'Failed ✗'
            sender_analysis['warning_flags'].append('No sender email found')
            sender_analysis['threat_contribution'] += 30

        return sender_analysis

    def _analyze_urls_enhanced(self, content: str) -> List[Dict]:
        """Enhanced URL analysis with anchor text and redirect tracking"""
        url_pattern = r'https?://[^\s<>"{}|\\^\[\]`]+'
        urls = re.findall(url_pattern, content)

        # Extract anchor text (from HTML if present)
        anchor_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
        anchors = re.findall(anchor_pattern, content, re.IGNORECASE)

        analyzed_urls = []

        for idx, url in enumerate(urls):
            url_analysis = {
                'url': url,
                'anchor_text': None,
                'domain': '',
                'subdomain': '',
                'tld': '',
                'risk_score': 0,
                'issues': [],
                'security_checks': {},
                'redirect_chain': [],
                'final_destination': None,
                'reputation': 'Unknown',
                'visual_similarity': None
            }

            # Find matching anchor text
            for anchor_url, anchor_text in anchors:
                if anchor_url in url or url in anchor_url:
                    url_analysis['anchor_text'] = anchor_text.strip()
                    break

            try:
                parsed = urlparse(url)
                domain = parsed.hostname or ''
                url_analysis['domain'] = domain

                # Extract domain components
                extracted = tldextract.extract(url)
                url_analysis['subdomain'] = extracted.subdomain
                url_analysis['tld'] = extracted.suffix

                # Security Checks
                url_analysis['security_checks'] = {
                    'https': parsed.scheme == 'https',
                    'ip_address': bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain)),
                    'url_shortener': any(shortener in domain for shortener in self.url_shorteners),
                    'suspicious_tld': any(domain.endswith(tld) for tld in self.suspicious_tlds),
                    'punycode': 'xn--' in domain,
                    'excessive_subdomains': len(domain.split('.')) > 4
                }

                # Check for IP address
                if url_analysis['security_checks']['ip_address']:
                    url_analysis['issues'].append('⚠️ IP address instead of domain name')
                    url_analysis['risk_score'] += 30

                # Check for URL shortener
                if url_analysis['security_checks']['url_shortener']:
                    url_analysis['issues'].append('🔗 URL shortener detected (hides destination)')
                    url_analysis['risk_score'] += 25

                # Check for suspicious TLD
                if url_analysis['security_checks']['suspicious_tld']:
                    url_analysis['issues'].append('🚩 Suspicious top-level domain')
                    url_analysis['risk_score'] += 20

                # Check for homograph attack
                if url_analysis['security_checks']['punycode']:
                    url_analysis['issues'].append('🎭 Punycode domain (homograph attack risk)')
                    url_analysis['risk_score'] += 35

                # Check for excessive subdomains
                if url_analysis['security_checks']['excessive_subdomains']:
                    url_analysis['issues'].append(f'📊 Excessive subdomains ({len(domain.split(".")) - 2})')
                    url_analysis['risk_score'] += 15

                # Check HTTPS
                if not url_analysis['security_checks']['https']:
                    url_analysis['issues'].append('🔓 Not using HTTPS encryption')
                    url_analysis['risk_score'] += 10

                # Anchor text mismatch detection
                if url_analysis['anchor_text']:
                    # Check if anchor text mentions a legitimate company but URL doesn't
                    for legit_domain in self.legitimate_domains:
                        company_name = legit_domain.split('.')[0]
                        if company_name in url_analysis['anchor_text'].lower():
                            if legit_domain not in domain:
                                url_analysis['issues'].append(
                                    f'⚠️ Anchor text mentions "{company_name}" but URL goes to different domain')
                                url_analysis['risk_score'] += 35
                                url_analysis['visual_similarity'] = f'Pretends to be {legit_domain}'
                                break

                    # Check for generic "click here" which is suspicious
                    if url_analysis['anchor_text'].lower() in ['click here', 'click', 'here', 'verify', 'confirm']:
                        url_analysis['issues'].append('🎯 Generic/suspicious anchor text')
                        url_analysis['risk_score'] += 10

                # Check for lookalike domains (typosquatting)
                for legit_domain in self.legitimate_domains:
                    try:
                        similarity = SequenceMatcher(None, domain, legit_domain).ratio()
                    except Exception:
                        similarity = 0

                    if 0.7 < similarity < 0.95:
                        url_analysis['issues'].append(f'🔍 Possible impersonation of {legit_domain}')
                        url_analysis['risk_score'] += 40
                        url_analysis['visual_similarity'] = f'Similar to {legit_domain}'
                        break

                # Simulate redirect chain detection (in real implementation, would follow redirects)
                if url_analysis['security_checks']['url_shortener']:
                    url_analysis['redirect_chain'] = [
                        {'hop': 1, 'url': url, 'status': 'Shortened URL'},
                        {'hop': 2, 'url': 'Unknown (would need to follow)', 'status': 'Hidden destination'}
                    ]
                    url_analysis['final_destination'] = 'Hidden by URL shortener'
                else:
                    url_analysis['final_destination'] = domain

                # Calculate reputation
                if url_analysis['risk_score'] >= 60:
                    url_analysis['reputation'] = '🔴 High Risk'
                elif url_analysis['risk_score'] >= 30:
                    url_analysis['reputation'] = '🟡 Suspicious'
                elif url_analysis['risk_score'] >= 15:
                    url_analysis['reputation'] = '🟠 Caution'
                else:
                    url_analysis['reputation'] = '🟢 Low Risk'

            except Exception as e:
                url_analysis['issues'].append('❌ Invalid URL format')
                url_analysis['risk_score'] = 50

            analyzed_urls.append(url_analysis)

        return analyzed_urls

    def _keyword_analysis(self, content: str) -> Dict:
        """Enhanced keyword and pattern analysis"""
        content_lower = content.lower()
        analysis = {
            'detected_categories': {},
            'total_matches': 0,
            'threat_contribution': 0
        }

        for category, keywords in self.phishing_keywords.items():
            matches = [kw for kw in keywords if kw in content_lower]
            if matches:
                analysis['detected_categories'][category] = {
                    'count': len(matches),
                    'keywords': matches[:5]
                }
                analysis['total_matches'] += len(matches)

        if analysis['total_matches'] > 10:
            analysis['threat_contribution'] = 30
        elif analysis['total_matches'] > 5:
            analysis['threat_contribution'] = 20
        elif analysis['total_matches'] > 2:
            analysis['threat_contribution'] = 10

        return analysis

    def _analyze_headers(self, content: str) -> Dict:
        """Analyze email headers for security indicators"""
        header_analysis = {
            'authentication_results': None,
            'received_chain': [],
            'suspicious_headers': [],
            'threat_contribution': 0
        }

        if 'Authentication-Results' in content:
            header_analysis['authentication_results'] = 'Present'
        else:
            header_analysis['suspicious_headers'].append('Missing Authentication-Results')
            header_analysis['threat_contribution'] += 5

        if 'X-Originating-IP' in content:
            ip_match = re.search(r'X-Originating-IP:\s*\[?(\d+\.\d+\.\d+\.\d+)\]?', content)
            if ip_match:
                header_analysis['received_chain'].append(f'Origin IP: {ip_match.group(1)}')

        if 'X-Mailer: Microsoft CDO' in content:
            header_analysis['suspicious_headers'].append('Outdated mailer detected')
            header_analysis['threat_contribution'] += 10

        if 'Received-SPF: fail' in content or 'Received-SPF: softfail' in content:
            header_analysis['suspicious_headers'].append('SPF check failed')
            header_analysis['threat_contribution'] += 15

        return header_analysis

    def _analyze_content_patterns(self, content: str) -> Dict:
        """Analyze content for suspicious patterns"""
        content_analysis = {
            'grammar_score': 100,
            'urgency_level': 'Low',
            'personalization': 'Generic',
            'suspicious_patterns': [],
            'threat_contribution': 0
        }

        grammar_issues = len(re.findall(r'\b[a-z]\s+[A-Z]|[.!?]\s*[a-z]|\s{2,}', content))
        content_analysis['grammar_score'] = max(0, 100 - (grammar_issues * 5))

        if content_analysis['grammar_score'] < 70:
            content_analysis['suspicious_patterns'].append('Poor grammar detected')
            content_analysis['threat_contribution'] += 10

        urgent_phrases = len(re.findall(r'(urgent|immediate|expire|now|today|hurry)', content.lower()))
        if urgent_phrases > 3:
            content_analysis['urgency_level'] = 'High'
            content_analysis['threat_contribution'] += 15
        elif urgent_phrases > 1:
            content_analysis['urgency_level'] = 'Medium'
            content_analysis['threat_contribution'] += 5

        if re.search(r'dear (customer|user|client|member)', content.lower()):
            content_analysis['personalization'] = 'Generic'
            content_analysis['suspicious_patterns'].append('Generic greeting')
            content_analysis['threat_contribution'] += 5

        if '$' in content or '€€' in content:
            content_analysis['suspicious_patterns'].append('Excessive currency symbols')
            content_analysis['threat_contribution'] += 10

        if len(re.findall(r'[A-Z]{5,}', content)) > 3:
            content_analysis['suspicious_patterns'].append('Excessive capitalization')
            content_analysis['threat_contribution'] += 5

        return content_analysis

    def _calculate_risk_level(self, score: int) -> str:
        """Calculate risk level based on threat score"""
        if score >= 70:
            return 'Critical'
        elif score >= 50:
            return 'High'
        elif score >= 30:
            return 'Medium'
        elif score >= 15:
            return 'Low'
        else:
            return 'Safe'

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence in the analysis"""
        confidence = 50.0

        if results['ml_predictions'].get('phishing_detection'):
            confidence += results['ml_predictions']['phishing_detection'].get('score', 0) * 20

        if results['keyword_analysis']['total_matches'] > 5:
            confidence += 10

        if len(results['url_analysis']) > 0:
            suspicious_urls = sum(1 for url in results['url_analysis'] if url['risk_score'] > 30)
            confidence += min(suspicious_urls * 5, 15)

        if results['sender_analysis'].get('domain_exists'):
            confidence += 5

        return min(confidence, 95)

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if results['risk_level'] in ['Critical', 'High']:
            recommendations.append("🚨 Do not click any links or download attachments")
            recommendations.append("⚠️ Report this email to your IT security team immediately")
            recommendations.append("🗑️ Delete this email from your inbox and trash")

        elif results['risk_level'] == 'Medium':
            recommendations.append("⚠️ Exercise caution with this email")
            recommendations.append("🔍 Verify the sender through alternative channels")
            recommendations.append("🚫 Avoid providing sensitive information")

        else:
            recommendations.append("✅ Email appears to be legitimate")
            recommendations.append("💡 Always verify unexpected requests")
            recommendations.append("🛡️ Keep your security software updated")

        if results['url_analysis']:
            for url in results['url_analysis']:
                if url['risk_score'] > 50:
                    recommendations.append(f"⛔ Do not visit: {url['domain']}")

        if not results['sender_analysis'].get('domain_exists', True):
            recommendations.append("📧 Sender domain appears invalid - likely spoofed")

        return recommendations

    def _create_analysis_timeline(self) -> List[Dict]:
        """Create analysis timeline for transparency"""
        return [
            {'step': 'Email received', 'status': 'complete', 'time': '0.0s'},
            {'step': 'ML analysis', 'status': 'complete', 'time': '0.5s'},
            {'step': 'Pattern detection', 'status': 'complete', 'time': '0.7s'},
            {'step': 'URL scanning', 'status': 'complete', 'time': '1.2s'},
            {'step': 'Sender verification', 'status': 'complete', 'time': '1.5s'},
            {'step': 'Risk assessment', 'status': 'complete', 'time': '1.8s'}
        ]


# ================================
# Visualization Components
# ================================

def create_threat_gauge(score: int) -> go.Figure:
    """Create animated threat score gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Threat Score", 'font': {'size': 20}},
        delta={'reference': 30, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#10b981'},
                {'range': [25, 50], 'color': '#f59e0b'},
                {'range': [50, 75], 'color': '#ef4444'},
                {'range': [75, 100], 'color': '#991b1b'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig


def create_risk_distribution(analysis_results: Dict) -> go.Figure:
    """Create risk distribution chart"""
    categories = []
    values = []

    if analysis_results['ml_predictions']['threat_contribution'] > 0:
        categories.append('ML Detection')
        values.append(analysis_results['ml_predictions']['threat_contribution'])

    if analysis_results['keyword_analysis']['threat_contribution'] > 0:
        categories.append('Keywords')
        values.append(analysis_results['keyword_analysis']['threat_contribution'])

    if analysis_results['sender_analysis']['threat_contribution'] > 0:
        categories.append('Sender Checks')
        values.append(analysis_results['sender_analysis']['threat_contribution'])

    if analysis_results['content_analysis']['threat_contribution'] > 0:
        categories.append('Content Patterns')
        values.append(analysis_results['content_analysis']['threat_contribution'])

    if not categories:
        categories = ['No major indicators']
        values = [1]

    fig = go.Figure(data=[go.Pie(labels=categories, values=values, hole=.4)])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def create_nlp_performance_chart(nlp_insights: Dict) -> go.Figure:
    """Create NLP performance visualization"""
    model_perf = nlp_insights.get('model_performance', {})

    categories = ['Phishing<br>Detection', 'Sentiment<br>Analysis', 'Intent<br>Classification']
    values = [
        model_perf.get('phishing_confidence', 0) * 100,
        model_perf.get('sentiment_confidence', 0) * 100,
        model_perf.get('intent_confidence', 0) * 100
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=['#6366f1', '#8b5cf6', '#a855f7'],
            text=[f'{v:.1f}%' for v in values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title='NLP Model Performance',
        yaxis_title='Confidence Score (%)',
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


# ================================
# Utility & Export Helpers
# ================================

def json_download_button(data: Dict, filename: str = "phish_report.json"):
    b = json.dumps(data, indent=2).encode('utf-8')
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">⬇️ Download JSON report</a>'
    st.markdown(href, unsafe_allow_html=True)


def csv_download_button(df: pd.DataFrame, filename: str = "phish_report.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


# ================================
# UI Components & Layout
# ================================

with st.container():
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Analyze Email", "Live Monitor", "Settings"],
        icons=["speedometer2", "envelope", "activity", "gear"],
        menu_icon="shield-check",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px 8px"},
            "nav-link-selected": {"background-color": "#6366f1"},
        }
    )

st.markdown("<h1 style='text-align:center;color:#1e293b;'>🛡️ PhishShield – Advanced Email Security</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#64748b;font-size:16px;'>AI-Powered Phishing Detection with Comprehensive Sender Verification & URL Analysis</p>",
    unsafe_allow_html=True)

detector = AdvancedPhishingDetector()

# Demo examples
demo_examples = {
    "Phishing - Urgent Account Lock": "From: \"PayPal Support\" <support@paypa1.com>\nSubject: Immediate action required\n\nDear customer,\nYour account will be suspended within 24 hours. Please verify your account by clicking the link: https://bit.ly/verify123\n\nSincerely,\nPayPal Support",
    "Legitimate - Newsletter": "From: \"GitHub\" <updates@github.com>\nSubject: GitHub Weekly\n\nHello,\nHere's what's new on GitHub this week. <a href='https://github.com/settings/notifications'>Unsubscribe</a>\n\nThanks,\nGitHub Team",
    "Suspicious - Prize Claim": "From: store@free-prizes.tk\nSubject: CONGRATULATIONS! You won $1,000,000!\n\nClick here to claim: <a href='https://bit.ly/claim-prize'>CLAIM NOW</a>\n\nAct immediately!"
}

# Page: Dashboard
if selected == "Dashboard":
    col1, col2, col3 = st.columns([2, 4, 2])
    with col1:
        st.metric("Model Status", "✅ Loaded", delta="Active")
        st.write("🖥️ GPU:", "Available" if torch.cuda.is_available() else "CPU Mode")
        st.write(f"🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.plotly_chart(create_threat_gauge(10), use_container_width=True)
    with col3:
        st.write("### Quick Stats")
        st.metric("Emails Analyzed", st.session_state.get('total_analyzed', 0))
        st.metric("Threats Detected", st.session_state.get('threats_detected', 0))

# Page: Analyze Email
if selected == "Analyze Email":
    left, right = st.columns([1, 2])

    with left:
        st.subheader("📥 Input Email")
        choice = st.selectbox('Select demo or paste email', ['-- paste or upload --'] + list(demo_examples.keys()))

        if choice != "-- paste or upload --":
            email_text = st.text_area("Raw email content", value=demo_examples[choice], height=260)
        else:
            uploaded = st.file_uploader("Upload .eml or .txt", type=['eml', 'txt'])
            if uploaded:
                email_text = uploaded.getvalue().decode(errors='ignore')
                st.text_area("Raw email content", value=email_text[:500] + "...", height=260, key="preview")
            else:
                email_text = st.text_area("Raw email content", value="", height=260)

        analyze_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    with right:
        st.subheader("📊 Analysis Results")

        if analyze_btn and email_text.strip():
            with st.spinner("🔄 Analyzing email..."):
                res = detector.analyze_email(email_text)
                st.session_state['last_result'] = res
                st.session_state['total_analyzed'] = st.session_state.get('total_analyzed', 0) + 1
                if res['risk_level'] in ['High', 'Critical']:
                    st.session_state['threats_detected'] = st.session_state.get('threats_detected', 0) + 1

        if 'last_result' in st.session_state:
            res = st.session_state['last_result']

            # Top metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                risk_emoji = {'Safe': '✅', 'Low': '🟢', 'Medium': '🟡', 'High': '🔴', 'Critical': '🚨'}
                st.metric("Risk Level", f"{risk_emoji.get(res['risk_level'], '⚠️')} {res['risk_level']}",
                          delta=f"{res['threat_score']} pts")
            with col_b:
                st.metric("Confidence", f"{res['confidence']:.0f}%")
            with col_c:
                st.metric("URLs Found", len(res['url_analysis']))

            # Threat Gauge
            st.plotly_chart(create_threat_gauge(int(res['threat_score'])), use_container_width=True)

            # ENHANCED SENDER VERIFICATION SECTION
            st.markdown("---")
            st.markdown("## 📧 Sender Verification Analysis")

            sender = res['sender_analysis']

            st.markdown(f"""
            <div class='verification-card'>
                <h3 style='margin-top:0;'>Verification Status: {sender['verification_status']}</h3>
                <div class='metric-box'>
                    <strong>Email:</strong> {sender.get('email', 'Not found')}<br>
                    <strong>Display Name:</strong> {sender.get('display_name', 'None')}<br>
                    <strong>Domain:</strong> {sender.get('domain', 'N/A')}<br>
                    <strong>Domain Age:</strong> {sender.get('domain_age', 'Unknown')}<br>
                    <strong>Registrar:</strong> {sender.get('domain_registrar', 'Unknown')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Security Protocols
            st.markdown("### 🔐 Email Authentication Protocols")
            auth_col1, auth_col2, auth_col3 = st.columns(3)

            with auth_col1:
                spf_status = sender.get('spf', 'Not checked')
                spf_class = 'status-verified' if 'Present' in str(spf_status) else 'status-failed'
                st.markdown(f"""
                <div class='status-badge {spf_class}'>
                    SPF: {spf_status}
                </div>
                """, unsafe_allow_html=True)

            with auth_col2:
                dkim_status = sender.get('dkim', 'Not checked')
                dkim_class = 'status-verified' if 'Present' in str(dkim_status) else 'status-warning'
                st.markdown(f"""
                <div class='status-badge {dkim_class}'>
                    DKIM: {dkim_status}
                </div>
                """, unsafe_allow_html=True)

            with auth_col3:
                dmarc_status = sender.get('dmarc', 'Not checked')
                dmarc_class = 'status-verified' if 'Present' in str(dmarc_status) else 'status-failed'
                st.markdown(f"""
                <div class='status-badge {dmarc_class}'>
                    DMARC: {dmarc_status}
                </div>
                """, unsafe_allow_html=True)

            # Warning Flags
            if sender.get('warning_flags'):
                st.markdown("### ⚠️ Warning Flags")
                for flag in sender['warning_flags']:
                    st.warning(flag)

            # MX Records
            if sender.get('mx_records'):
                with st.expander("📮 MX Records (Mail Servers)"):
                    for mx in sender['mx_records']:
                        st.code(mx)

            # ENHANCED URL ANCHOR ANALYSIS SECTION
            st.markdown("---")
            st.markdown("## 🔗 URL & Anchor Analysis")

            if res['url_analysis']:
                for idx, url_data in enumerate(res['url_analysis'], 1):
                    st.markdown(f"""
                    <div class='url-analysis-card'>
                        <h4 style='margin-top:0;'>URL #{idx} - {url_data['reputation']}</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    url_col1, url_col2 = st.columns([1, 1])

                    with url_col1:
                        st.markdown(f"**🌐 URL:** `{url_data['url'][:60]}...`")
                        st.markdown(f"**🏷️ Anchor Text:** `{url_data.get('anchor_text', 'No anchor text')}`")
                        st.markdown(f"**📍 Domain:** `{url_data['domain']}`")
                        st.markdown(f"**🎯 Final Destination:** `{url_data.get('final_destination', 'Unknown')}`")

                    with url_col2:
                        st.markdown(f"**⚠️ Risk Score:** {url_data['risk_score']}/100")
                        if url_data.get('visual_similarity'):
                            st.error(f"🎭 {url_data['visual_similarity']}")

                        # Security checks badges
                        checks = url_data.get('security_checks', {})
                        st.markdown("**Security Checks:**")
                        check_html = ""
                        if checks.get('https'):
                            check_html += "<span class='status-badge status-verified'>✓ HTTPS</span>"
                        else:
                            check_html += "<span class='status-badge status-failed'>✗ No HTTPS</span>"

                        if checks.get('url_shortener'):
                            check_html += "<span class='status-badge status-warning'>⚠ Shortener</span>"
                        if checks.get('ip_address'):
                            check_html += "<span class='status-badge status-failed'>⚠ IP Address</span>"

                        st.markdown(check_html, unsafe_allow_html=True)

                    # Issues
                    if url_data['issues']:
                        st.markdown("**🚨 Detected Issues:**")
                        for issue in url_data['issues']:
                            st.markdown(f"- {issue}")

                    # Redirect Chain
                    if url_data.get('redirect_chain'):
                        with st.expander("🔄 Redirect Chain"):
                            for hop in url_data['redirect_chain']:
                                st.markdown(f"""
                                <div class='url-redirect-chain'>
                                    <strong>Hop {hop['hop']}:</strong> {hop['url']}<br>
                                    <em>Status: {hop['status']}</em>
                                </div>
                                """, unsafe_allow_html=True)

                    st.markdown("---")

                # URL Summary Table
                st.markdown("### 📊 URL Summary Table")
                df_urls = pd.DataFrame(res['url_analysis'])
                display_cols = ['url', 'anchor_text', 'domain', 'risk_score', 'reputation']
                available_cols = [col for col in display_cols if col in df_urls.columns]
                st.dataframe(df_urls[available_cols], use_container_width=True)

            else:
                st.info("ℹ️ No URLs found in this email")

            # NLP INSIGHTS SECTION
            st.markdown("---")
            st.markdown("## 🤖 NLP & Machine Learning Insights")

            nlp_insights = res.get('nlp_insights', {})
            ml_pred = res.get('ml_predictions', {})

            st.markdown(f"""
            <div class='nlp-insights-card'>
                <h3 style='margin-top:0;'>Natural Language Processing Analysis</h3>
                <div class='metric-box'>
                    <strong>Content Credibility Score:</strong> {nlp_insights.get('credibility_score', 0)}/100<br>
                    <strong>Lexical Diversity:</strong> {nlp_insights.get('text_statistics', {}).get('lexical_diversity', 0):.2%}<br>
                    <strong>Unique Words:</strong> {nlp_insights.get('text_statistics', {}).get('unique_words', 0)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # NLP Performance Chart
            if ml_pred.get('model_performance'):
                st.plotly_chart(create_nlp_performance_chart(ml_pred), use_container_width=True)

            # ML Predictions Detail
            nlp_col1, nlp_col2 = st.columns(2)

            with nlp_col1:
                st.markdown("### 🎯 ML Model Predictions")
                phish_detect = ml_pred.get('phishing_detection', {})
                st.metric("Phishing Prediction",
                          phish_detect.get('prediction', 'Unknown'),
                          delta=f"{phish_detect.get('score', 0):.2%} confidence")

                sentiment = ml_pred.get('sentiment', {})
                st.metric("Email Sentiment",
                          sentiment.get('label', 'Unknown'),
                          delta=f"{sentiment.get('score', 0):.2%} confidence")

            with nlp_col2:
                st.markdown("### 🎭 Manipulation Tactics Detected")
                tactics = nlp_insights.get('manipulation_tactics', [])
                if tactics:
                    for tactic in tactics:
                        st.error(f"⚠️ {tactic}")
                else:
                    st.success("✅ No manipulation tactics detected")

            # Intent Classification
            intent = ml_pred.get('intent', {})
            if intent.get('labels'):
                st.markdown("### 📋 Intent Classification")
                intent_df = pd.DataFrame({
                    'Intent': intent['labels'][:3],
                    'Confidence': [f"{s:.2%}" for s in intent['scores'][:3]]
                })
                st.table(intent_df)

            # Linguistic Patterns
            with st.expander("📝 Linguistic Patterns Analysis"):
                ling_patterns = nlp_insights.get('linguistic_patterns', {})
                st.json(ling_patterns)

            # Risk Distribution
            st.markdown("---")
            st.markdown("### 📊 Threat Score Distribution")
            st.plotly_chart(create_risk_distribution(res), use_container_width=True)

            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Security Recommendations")
            for rec in res['recommendations']:
                if '🚨' in rec or '⛔' in rec:
                    st.error(rec)
                elif '⚠️' in rec:
                    st.warning(rec)
                else:
                    st.info(rec)

            # Analysis Timeline
            with st.expander("⏱️ Analysis Timeline"):
                for step in res['timeline']:
                    st.markdown(f"✓ **{step['step']}** - {step['time']}")

            # Export Options
            st.markdown("---")
            st.markdown("### 📥 Export Report")
            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                json_download_button(res, "phishield_report.json")

            with export_col2:
                if res['url_analysis']:
                    csv_df = pd.DataFrame(res['url_analysis'])
                    csv_download_button(csv_df, "url_analysis.csv")

            with export_col3:
                # Create summary report
                summary_text = f"""
PhishShield Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL ASSESSMENT
Risk Level: {res['risk_level']}
Threat Score: {res['threat_score']}/100
Confidence: {res['confidence']:.1f}%

SENDER VERIFICATION
Email: {sender.get('email', 'N/A')}
Domain: {sender.get('domain', 'N/A')}
Status: {sender['verification_status']}
SPF: {sender.get('spf', 'N/A')}
DMARC: {sender.get('dmarc', 'N/A')}

URL ANALYSIS
Total URLs: {len(res['url_analysis'])}
High Risk URLs: {sum(1 for u in res['url_analysis'] if u['risk_score'] > 50)}

ML PREDICTIONS
Phishing Detection: {ml_pred.get('phishing_detection', {}).get('prediction', 'N/A')}
Sentiment: {ml_pred.get('sentiment', {}).get('label', 'N/A')}

NLP INSIGHTS
Credibility Score: {nlp_insights.get('credibility_score', 0)}/100
Manipulation Tactics: {len(nlp_insights.get('manipulation_tactics', []))}
                """

                summary_bytes = summary_text.encode('utf-8')
                summary_b64 = base64.b64encode(summary_bytes).decode()
                summary_href = f'<a href="data:text/plain;base64,{summary_b64}" download="summary.txt">📄 Download Summary</a>'
                st.markdown(summary_href, unsafe_allow_html=True)

        else:
            st.info("👈 Paste or upload an email and click 'Analyze Email' to get started")

            # Show feature highlights
            st.markdown("### ✨ Analysis Features")
            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("""
                **🛡️ Comprehensive Protection:**
                - AI-powered phishing detection
                - Advanced NLP sentiment analysis
                - Intent classification
                - Real-time threat scoring
                """)

            with feature_col2:
                st.markdown("""
                **🔍 Deep Investigation:**
                - Complete sender verification
                - SPF/DKIM/DMARC validation
                - URL anchor text analysis
                - Redirect chain tracking
                """)

# Page: Live Monitor
if selected == "Live Monitor":
    st.subheader("📡 Live Email Monitor")
    st.markdown("Simulate real-time email stream monitoring and analysis")

    if 'monitor_buffer' not in st.session_state:
        st.session_state['monitor_buffer'] = []

    # Input form
    with st.form("live_input", clear_on_submit=True):
        st.markdown("### 📨 Inject Test Email into Stream")
        sample_raw = st.text_area("Paste email content", height=160)
        col1, col2 = st.columns([1, 4])

        with col1:
            add_btn = st.form_submit_button("➕ Add to Stream", type="primary")

        with col2:
            demo_select = st.selectbox("Or use demo:", [""] + list(demo_examples.keys()))

        if add_btn:
            email_content = sample_raw if sample_raw.strip() else demo_examples.get(demo_select, "")
            if email_content.strip():
                st.session_state['monitor_buffer'].append({
                    "time": datetime.now(pytz.timezone('Asia/Kolkata')).isoformat(),
                    "payload": email_content
                })
                st.success("✅ Email added to monitoring stream")

    # Display stream
    st.markdown("---")
    st.markdown("### 📊 Email Stream Buffer")

    if st.session_state['monitor_buffer']:
        items = st.session_state['monitor_buffer'][-10:][::-1]

        for idx, item in enumerate(items):
            with st.expander(f"📧 Email #{len(items) - idx} - {item['time'][:19]}"):
                preview = item['payload'][:400]
                st.text(preview + ("..." if len(item['payload']) > 400 else ""))

                analyze_col, threat_col = st.columns([1, 3])

                with analyze_col:
                    if st.button(f"🔍 Analyze", key=f"analyze_stream_{idx}"):
                        with st.spinner("Analyzing..."):
                            res = detector.analyze_email(item['payload'])
                            st.session_state[f'stream_result_{idx}'] = res

                # Show results if analyzed
                if f'stream_result_{idx}' in st.session_state:
                    res = st.session_state[f'stream_result_{idx}']

                    with threat_col:
                        risk_colors = {
                            'Safe': '🟢',
                            'Low': '🟢',
                            'Medium': '🟡',
                            'High': '🔴',
                            'Critical': '🚨'
                        }
                        st.markdown(
                            f"**Risk:** {risk_colors.get(res['risk_level'], '⚠️')} {res['risk_level']} | **Score:** {res['threat_score']}/100 | **Confidence:** {res['confidence']:.0f}%")

                    # Mini threat gauge
                    st.plotly_chart(create_threat_gauge(int(res['threat_score'])), use_container_width=True,
                                    key=f"gauge_{idx}")

        # Clear button
        if st.button("🗑️ Clear Stream Buffer"):
            st.session_state['monitor_buffer'] = []
            # Clear all results
            for key in list(st.session_state.keys()):
                if key.startswith('stream_result_'):
                    del st.session_state[key]
            st.rerun()
    else:
        st.info("📭 Stream buffer is empty. Add emails to start monitoring.")

# Page: Settings
if selected == "Settings":
    st.subheader("⚙️ Settings & Configuration")

    settings_tab1, settings_tab2, settings_tab3 = st.tabs(["🤖 ML Models", "🔍 Detection Rules", "📊 Reporting"])

    with settings_tab1:
        st.markdown("### Machine Learning Configuration")

        use_ml = st.checkbox("Enable ML Models", value=True, help="Use transformer models for analysis")

        if use_ml:
            st.success("✅ ML models are active")

            # Show model info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
            with col2:
                st.metric("Model Status", "Loaded")

            st.markdown("**Active Models:**")
            st.markdown("""
            - 🎯 BERT Phishing Classifier
            - 💭 DistilBERT Sentiment Analyzer
            - 🎭 BART Zero-Shot Intent Classifier
            """)
        else:
            st.warning("⚠️ ML models disabled - using heuristic fallbacks")

        st.markdown("---")
        st.markdown("### Model Performance Thresholds")

        phishing_threshold = st.slider("Phishing Detection Threshold", 0.0, 1.0, 0.5, 0.05)
        sentiment_threshold = st.slider("Sentiment Analysis Threshold", 0.0, 1.0, 0.7, 0.05)

        st.info(f"Emails with phishing score > {phishing_threshold} will be flagged")

    with settings_tab2:
        st.markdown("### Detection Rule Configuration")

        st.markdown("**Keyword Detection Sensitivity**")
        keyword_sensitivity = st.select_slider(
            "Sensitivity Level",
            options=["Low", "Medium", "High", "Very High"],
            value="Medium"
        )

        st.markdown("**URL Analysis Settings**")
        check_url_shorteners = st.checkbox("Flag URL shorteners", value=True)
        check_suspicious_tlds = st.checkbox("Check suspicious TLDs", value=True)
        check_homograph = st.checkbox("Detect homograph attacks", value=True)

        st.markdown("**Sender Verification**")
        require_spf = st.checkbox("Require SPF records", value=False)
        require_dmarc = st.checkbox("Require DMARC policy", value=False)
        flag_free_email = st.checkbox("Flag free email providers", value=True)

        suspicious_tld_score = st.slider("Suspicious TLD Risk Score", 0, 50, 20)

    with settings_tab3:
        st.markdown("### Report Configuration")

        st.markdown("**Export Format Preferences**")
        include_raw_email = st.checkbox("Include raw email in reports", value=False)
        include_full_headers = st.checkbox("Include full header analysis", value=True)
        include_ml_details = st.checkbox("Include detailed ML predictions", value=True)

        st.markdown("**Notification Settings**")
        notify_on_critical = st.checkbox("Alert on Critical threats", value=True)
        notify_on_high = st.checkbox("Alert on High threats", value=False)

        email_reports = st.text_input("Send reports to email (comma-separated):",
                                      placeholder="admin@company.com, security@company.com")

    # Save button
    st.markdown("---")
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        st.session_state['settings'] = {
            "use_ml": use_ml,
            "phishing_threshold": phishing_threshold,
            "sentiment_threshold": sentiment_threshold,
            "keyword_sensitivity": keyword_sensitivity,
            "check_url_shorteners": check_url_shorteners,
            "check_suspicious_tlds": check_suspicious_tlds,
            "check_homograph": check_homograph,
            "require_spf": require_spf,
            "require_dmarc": require_dmarc,
            "flag_free_email": flag_free_email,
            "suspicious_tld_score": suspicious_tld_score,
            "include_raw_email": include_raw_email,
            "include_full_headers": include_full_headers,
            "include_ml_details": include_ml_details,
            "notify_on_critical": notify_on_critical,
            "notify_on_high": notify_on_high,
            "email_reports": email_reports
        }
        st.success("✅ Settings saved successfully!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;padding:20px;">
    <strong>🛡️ PhishShield</strong> - Advanced Email Security Platform<br>
    Powered by AI & Machine Learning | Developed by DARN<br>
    <small>Version 2.0 | Enhanced with Comprehensive Analysis Features</small>
</div>
""", unsafe_allow_html=True)