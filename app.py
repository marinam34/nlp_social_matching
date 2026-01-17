from flask import Flask, render_template_string, request, jsonify
import json
import os
from datetime import datetime
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed


from src.nlp_processor import analyze_profile
from src.vector_database import VectorDatabase, add_user_to_index, find_similar_users
from src.matching_engine import get_user_matches
from src.adaptive_question_engine import AdaptiveQuestionEngine, get_next_adaptive_question
from src.adaptive_questions_template import ADAPTIVE_QUESTIONS_TEMPLATE

app = Flask(__name__)

LANGUAGE_CODES = {
    'English': 'en',
    'Spanish': 'es',
    'Arabic': 'ar',
    'French': 'fr',
    'German': 'de',
    'Urdu': 'ur',
    'Mandarin': 'zh-CN',
    'Other': 'en'
}

TRANSLATIONS_CACHE_FILE = 'data/translations_cache.json'

def load_translations_cache():
    if os.path.exists(TRANSLATIONS_CACHE_FILE):
        try:
            with open(TRANSLATIONS_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    return {}

def save_translations_cache(cache):
    try:
        with open(TRANSLATIONS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving cache: {e}")

translations_cache = load_translations_cache()

REGISTRATION_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect - Bridging Communities</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .app-container {
            width: 100%;
            max-width: 500px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        .app-logo {
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: 2px;
        }
        .app-tagline {
            font-size: 16px;
            font-weight: 300;
            opacity: 0.95;
            letter-spacing: 0.5px;
        }
        .form-container {
            padding: 40px 30px;
        }
        .form-group {
            margin-bottom: 24px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 15px;
            transition: all 0.3s;
            font-family: 'Poppins', sans-serif;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            font-family: 'Poppins', sans-serif;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        button:active {
            transform: translateY(0);
        }
        .message {
            padding: 14px;
            margin-top: 20px;
            border-radius: 10px;
            text-align: center;
            font-weight: 500;
            animation: slideIn 0.3s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .success {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        ::placeholder {
            color: #999;
        }
        .thank-you-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .thank-you-card {
            background: white;
            padding: 60px 40px;
            border-radius: 20px;
            text-align: center;
            animation: popIn 0.5s;
        }
        @keyframes popIn {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        .thank-you-card h1 {
            color: #667eea;
            font-size: 36px;
            margin-bottom: 20px;
        }
        .thank-you-card p {
            color: #666;
            font-size: 18px;
        }
        .checkmark {
            font-size: 60px;
            color: #4CAF50;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="app-header">
            <div class="app-logo">Connect</div>
            <div class="app-tagline">Bridging Communities, Building Connections</div>
        </div>
        <div class="form-container">
            <form id="registrationForm">
                <div class="form-group">
                    <label>Full Name *</label>
                    <input type="text" name="name" required>
                </div>

                <div class="form-group">
                    <label>Email *</label>
                    <input type="email" name="email" required>
                </div>

                <div class="form-group">
                    <label>Phone Number</label>
                    <input type="tel" name="phone">
                </div>

                <div class="form-group">
                    <label>Home Country *</label>
                    <input type="text" name="country" required>
                </div>

                <div class="form-group">
                    <label>Current Location *</label>
                    <input type="text" name="location" required>
                </div>

                <div class="form-group">
                    <label>Age *</label>
                    <input type="number" name="age" min="18" max="120" required placeholder="e.g. 25">
                </div>

                <div class="form-group">
                    <label>Current Status *</label>
                    <select name="status" required>
                        <option value="">Select...</option>
                        <option value="Student">Student</option>
                        <option value="Working">Working</option>
                        <option value="Job Seeking">Job Seeking</option>
                        <option value="Self-Employed">Self-Employed</option>
                        <option value="Unemployed">Unemployed</option>
                        <option value="Retired">Retired</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Profession/Field of Study</label>
                    <input type="text" name="profession" placeholder="e.g., Engineer, Teacher, Business">
                </div>

                <div class="form-group">
                    <label>Languages Spoken (comma separated) *</label>
                    <input type="text" name="languages" placeholder="e.g., English, Spanish, Arabic" required>
                </div>

                <div class="form-group">
                    <label>Preferred Language for Communication *</label>
                    <select name="preferred_language" required>
                        <option value="">Select...</option>
                        <option value="English">English</option>
                        <option value="Spanish">Spanish</option>
                        <option value="Arabic">Arabic</option>
                        <option value="French">French</option>
                        <option value="German">German</option>
                        <option value="Urdu">Urdu</option>
                        <option value="Mandarin">Mandarin</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>What are you looking for? (Primary Goal) *</label>
                    <select name="goal" required>
                        <option value="">Select your goal...</option>
                        <option value="social_connection">Social Connection (Meet new people)</option>
                        <option value="legal_support">Legal Support (Uni, work, visa, etc.)</option>
                        <option value="provide_legal_support">Provide Legal Support (Help others)</option>
                        <option value="mental_health">Mental Health Support (Talk to someone)</option>
                        <option value="language_assistance">Language Assistance (Improve skills)</option>
                    </select>
                </div>

                <button type="submit">Register</button>
            </form>
            <div id="message"></div>
        </div>
    </div>

    <div class="thank-you-overlay" id="thankYouOverlay">
        <div class="thank-you-card">
            <div class="checkmark">✓</div>
            <h1>Thank You!</h1>
            <p>Welcome to Connect</p>
        </div>
    </div>

    <script>
        document.getElementById('registrationForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                name: formData.get('name'),
                email: formData.get('email'),
                phone: formData.get('phone'),
                country: formData.get('country'),
                location: formData.get('location'),
                age: formData.get('age'),
                status: formData.get('status'),
                profession: formData.get('profession'),
                languages: formData.get('languages').split(',').map(l => l.trim()),
                preferred_language: formData.get('preferred_language'),
                goal: formData.get('goal')
            };

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('thankYouOverlay').style.display = 'flex';
                    setTimeout(() => {
                        window.location.href = '/questions?user_id=' + result.user_id;
                    }, 2000);
                } else {
                    showMessage(result.error, 'error');
                }
            } catch (error) {
                showMessage('Registration failed. Please try again.', 'error');
            }
        });

        function showMessage(text, type) {
            const msgDiv = document.getElementById('message');
            msgDiv.textContent = text;
            msgDiv.className = 'message ' + type;
            setTimeout(() => msgDiv.textContent = '', 5000);
        }
    </script>
</body>
</html>
"""

ASSESSMENT_WELCOME = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect - Assessment</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .app-container {
            width: 100%;
            max-width: 600px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        .app-logo { font-size: 36px; font-weight: 700; margin-bottom: 10px; }
        .app-tagline { font-size: 16px; font-weight: 300; opacity: 0.95; }
        .content { padding: 40px 30px; text-align: center; }
        h2 { color: #333; margin-bottom: 20px; font-size: 24px; }
        p { color: #666; line-height: 1.6; margin-bottom: 20px; }
        .language-info {
            background: #f0f4ff;
            border: 2px solid #667eea;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .language-icon { font-size: 24px; }
        .language-text {
            color: #667eea;
            font-weight: 600;
            font-size: 16px;
        }
        .start-button {
            padding: 16px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s;
            font-family: 'Poppins', sans-serif;
        }
        .start-button:hover { transform: translateY(-2px); }
        .loading {
            display: none;
            margin-top: 20px;
            color: #667eea;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="app-header">
            <div class="app-logo">Connect</div>
            <div class="app-tagline" id="tagline">Let's Find Your Perfect Matches</div>
        </div>
        <div class="content">
            <h2 id="welcomeTitle">Welcome to Your Assessment</h2>
            <div class="language-info">
                <span class="language-icon">🌐</span>
                <span class="language-text" id="languageDisplay">Loading...</span>
            </div>
            <p id="description">Help us understand your needs better so we can connect you with the right people and resources.</p>
            <p id="duration"><strong>This will only take 2-3 minutes</strong></p>
            <button class="start-button" id="startBtn" onclick="startAssessment()">Start Assessment</button>
            <div class="loading" id="loading">Loading assessment...</div>
        </div>
    </div>
    <script>
        let userId = null;
        let userLanguage = null;

        async function loadUserLanguage() {
            const urlParams = new URLSearchParams(window.location.search);
            userId = urlParams.get('user_id');
            
            try {
                const response = await fetch('/get-user-language?user_id=' + userId);
                const data = await response.json();
                userLanguage = data.language;
                
                document.getElementById('languageDisplay').textContent = 
                    'Assessment will be in ' + userLanguage;
                
                if (userLanguage !== 'English') {
                    await translatePage();
                }
            } catch (error) {
                console.error('Error loading language:', error);
                document.getElementById('languageDisplay').textContent = 'Assessment in English';
            }
        }

        async function translatePage() {
            const elements = {
                tagline: document.getElementById('tagline').textContent,
                welcomeTitle: document.getElementById('welcomeTitle').textContent,
                description: document.getElementById('description').textContent,
                duration: document.getElementById('duration').textContent,
                startBtn: document.getElementById('startBtn').textContent
            };

            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        texts: Object.values(elements),
                        target_language: userLanguage
                    })
                });

                const translations = await response.json();
                
                document.getElementById('tagline').textContent = translations[0];
                document.getElementById('welcomeTitle').textContent = translations[1];
                document.getElementById('description').textContent = translations[2];
                document.getElementById('duration').innerHTML = '<strong>' + translations[3] + '</strong>';
                document.getElementById('startBtn').textContent = translations[4];
            } catch (error) {
                console.error('Translation error:', error);
            }
        }

        function startAssessment() {
            document.getElementById('loading').style.display = 'block';
            window.location.href = '/assessment-questions?user_id=' + userId;
        }

        loadUserLanguage();
    </script>
</body>
</html>
"""

ASSESSMENT_QUESTIONS = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect - Assessment</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
        }
        .language-badge {
            background: white;
            border-radius: 20px;
            padding: 12px 20px;
            margin-bottom: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .language-badge span {
            color: #667eea;
            font-weight: 600;
            font-size: 14px;
        }
        .progress-bar {
            background: white;
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .progress {
            height: 10px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        .progress-text {
            margin-top: 10px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        .question-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .question-number {
            color: #667eea;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .question-text {
            font-size: 22px;
            color: #333;
            margin-bottom: 30px;
            line-height: 1.4;
        }
        .option {
            background: #f8f9fa;
            border: 2px solid transparent;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 16px;
        }
        .option:hover {
            background: #e9ecef;
            border-color: #667eea;
        }
        .option.selected {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .nav-buttons {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        .btn {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            font-family: 'Poppins', sans-serif;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-back {
            background: #e0e0e0;
            color: #666;
        }
        .btn-next {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .loading-content {
            background: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="language-badge">
            <span id="languageBadge">🌐 Language: Loading...</span>
        </div>
        
        <div class="progress-bar">
            <div class="progress">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">Question 1 of 10</div>
        </div>

        <div class="question-card">
            <div class="question-number" id="questionNumber">QUESTION 1</div>
            <div class="question-text" id="questionText"></div>
            <div id="optionsContainer"></div>
            <div class="nav-buttons">
                <button class="btn btn-back" id="backBtn" onclick="goBack()">Back</button>
                <button class="btn btn-next" id="nextBtn" onclick="goNext()" disabled>Next</button>
            </div>
        </div>
    </div>

    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <p>Loading question...</p>
        </div>
    </div>

    <script>
        let decisionTree = null;
        let currentQuestionIndex = 0;
        let generalQuestions = [];
        let categoryQuestions = [];
        let answers = [];
        let scores = {
            social_connection: 0,
            legal_support: 0,
            mental_health: 0,
            language_support: 0
        };
        let selectedOption = null;
        let userId = new URLSearchParams(window.location.search).get('user_id');
        let isInCategoryPhase = false;
        let topCategory = null;
        let userLanguage = null;
        let translatedQuestions = {};
        let buttonTexts = { back: 'Back', next: 'Next', question: 'QUESTION' };

        async function loadUserLanguage() {
            try {
                const response = await fetch('/get-user-language?user_id=' + userId);
                const data = await response.json();
                userLanguage = data.language;
                
                document.getElementById('languageBadge').textContent = '🌐 ' + userLanguage;
                
                if (userLanguage !== 'English') {
                    await translateButtons();
                }
            } catch (error) {
                console.error('Error loading language:', error);
                userLanguage = 'English';
            }
        }

        async function translateButtons() {
            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        texts: ['Back', 'Next', 'QUESTION', 'Question', 'of'],
                        target_language: userLanguage
                    })
                });

                const translations = await response.json();
                buttonTexts = {
                    back: translations[0],
                    next: translations[1],
                    questionUpper: translations[2],
                    question: translations[3],
                    of: translations[4]
                };

                document.getElementById('backBtn').textContent = buttonTexts.back;
                document.getElementById('nextBtn').textContent = buttonTexts.next;
            } catch (error) {
                console.error('Translation error:', error);
            }
        }

        async function loadDecisionTree() {
            document.getElementById('loadingOverlay').style.display = 'flex';
            
            const response = await fetch('/get-decision-tree');
            decisionTree = await response.json();
            generalQuestions = decisionTree.general_questions;
            
            await loadUserLanguage();
            
            if (userLanguage !== 'English') {
                await translateAllQuestions();
            }
            
            document.getElementById('loadingOverlay').style.display = 'none';
            displayQuestion();
        }

        async function translateAllQuestions() {
            const allTexts = [];
            const questionIds = [];
            
            generalQuestions.forEach(q => {
                questionIds.push({ id: q.id, type: 'general' });
                allTexts.push(q.question);
                q.options.forEach(opt => allTexts.push(opt.text));
            });

            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        texts: allTexts,
                        target_language: userLanguage
                    })
                });

                const translations = await response.json();
                
                let index = 0;
                generalQuestions.forEach(q => {
                    translatedQuestions[q.id] = {
                        question: translations[index++],
                        options: []
                    };
                    q.options.forEach(opt => {
                        translatedQuestions[q.id].options.push({
                            text: translations[index++],
                            original: opt
                        });
                    });
                });
            } catch (error) {
                console.error('Translation error:', error);
            }
        }

        async function translateCategoryQuestions() {
            if (!categoryQuestions.length || userLanguage === 'English') return;
            
            const allTexts = [];
            categoryQuestions.forEach(q => {
                allTexts.push(q.question);
                q.options.forEach(opt => allTexts.push(opt.text));
            });

            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        texts: allTexts,
                        target_language: userLanguage
                    })
                });

                const translations = await response.json();
                
                let index = 0;
                categoryQuestions.forEach(q => {
                    translatedQuestions[q.id] = {
                        question: translations[index++],
                        options: []
                    };
                    q.options.forEach(opt => {
                        translatedQuestions[q.id].options.push({
                            text: translations[index++],
                            original: opt
                        });
                    });
                });
            } catch (error) {
                console.error('Translation error:', error);
            }
        }

        function displayQuestion() {
            const questions = isInCategoryPhase ? categoryQuestions : generalQuestions;
            const question = questions[currentQuestionIndex];
            
            const totalQuestions = isInCategoryPhase ? 
                generalQuestions.length + categoryQuestions.length : 
                generalQuestions.length;
            const currentNum = isInCategoryPhase ? 
                generalQuestions.length + currentQuestionIndex + 1 : 
                currentQuestionIndex + 1;

            document.getElementById('questionNumber').textContent = 
                `${buttonTexts.questionUpper || 'QUESTION'} ${currentNum}`;
            
            if (translatedQuestions[question.id]) {
                document.getElementById('questionText').textContent = 
                    translatedQuestions[question.id].question;
            } else {
                document.getElementById('questionText').textContent = question.question;
            }
            
            document.getElementById('progressText').textContent = 
                `${buttonTexts.question || 'Question'} ${currentNum} ${buttonTexts.of || 'of'} ${totalQuestions}`;
            document.getElementById('progressFill').style.width = 
                `${(currentNum / totalQuestions) * 100}%`;

            const container = document.getElementById('optionsContainer');
            container.innerHTML = '';
            
            question.options.forEach((option, index) => {
                const div = document.createElement('div');
                div.className = 'option';
                
                if (translatedQuestions[question.id]) {
                    div.textContent = translatedQuestions[question.id].options[index].text;
                } else {
                    div.textContent = option.text;
                }
                
                div.onclick = () => selectOption(index, option);
                container.appendChild(div);
            });

            document.getElementById('backBtn').disabled = currentQuestionIndex === 0 && !isInCategoryPhase;
            document.getElementById('nextBtn').disabled = true;
            selectedOption = null;
        }

        function selectOption(index, option) {
            document.querySelectorAll('.option').forEach(el => el.classList.remove('selected'));
            document.querySelectorAll('.option')[index].classList.add('selected');
            selectedOption = option;
            document.getElementById('nextBtn').disabled = false;
        }

        async function goNext() {
            if (!selectedOption) return;

            answers.push({
                question_id: isInCategoryPhase ? 
                    categoryQuestions[currentQuestionIndex].id : 
                    generalQuestions[currentQuestionIndex].id,
                answer: selectedOption.text,
                score: selectedOption.score
            });

            if (!isInCategoryPhase) {
                for (let category in selectedOption.score) {
                    scores[category] += selectedOption.score[category];
                }
            }

            currentQuestionIndex++;

            if (!isInCategoryPhase && currentQuestionIndex >= generalQuestions.length) {
                topCategory = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
                
                categoryQuestions = decisionTree.category_specific_questions[topCategory] || [];
                
                if (categoryQuestions.length > 0) {
                    document.getElementById('loadingOverlay').style.display = 'flex';
                    await translateCategoryQuestions();
                    document.getElementById('loadingOverlay').style.display = 'none';
                    
                    isInCategoryPhase = true;
                    currentQuestionIndex = 0;
                    displayQuestion();
                } else {
                    await submitAssessment();
                }
            } else if (isInCategoryPhase && currentQuestionIndex >= categoryQuestions.length) {
                await submitAssessment();
            } else {
                displayQuestion();
            }
        }

        function goBack() {
            if (currentQuestionIndex > 0) {
                currentQuestionIndex--;
                answers.pop();
                displayQuestion();
            } else if (isInCategoryPhase) {
                isInCategoryPhase = false;
                currentQuestionIndex = generalQuestions.length - 1;
                answers.pop();
                displayQuestion();
            }
        }

        async function submitAssessment() {
            document.getElementById('loadingOverlay').style.display = 'flex';
            
            const response = await fetch('/submit-assessment', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    user_id: userId,
                    answers: answers,
                    scores: scores,
                    top_category: topCategory
                })
            });

            if (response.ok) {
                window.location.href = '/results?user_id=' + userId;
            }
        }

        loadDecisionTree();
    </script>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect - Your Results</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .header-card {
            background: white;
            border-radius: 20px;
            padding: 50px 40px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            animation: slideDown 0.5s;
        }
        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .success-icon {
            font-size: 80px;
            color: #4CAF50;
            margin-bottom: 20px;
            animation: scaleIn 0.6s;
        }
        @keyframes scaleIn {
            from { transform: scale(0); }
            to { transform: scale(1); }
        }
        .header-card h1 {
            color: #333;
            font-size: 36px;
            margin-bottom: 15px;
        }
        .header-card .user-name {
            color: #667eea;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .header-card .subtitle {
            color: #666;
            font-size: 18px;
            line-height: 1.6;
        }
        .results-grid {
            display: grid;
            gap: 25px;
            margin-bottom: 30px;
        }
        .result-card {
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeInUp 0.6s;
            animation-fill-mode: both;
        }
        .result-card:nth-child(1) { animation-delay: 0.1s; }
        .result-card:nth-child(2) { animation-delay: 0.2s; }
        .result-card:nth-child(3) { animation-delay: 0.3s; }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }
        .card-icon {
            font-size: 40px;
            margin-right: 20px;
        }
        .card-title {
            font-size: 24px;
            color: #333;
            font-weight: 600;
        }
        .card-content {
            color: #666;
            line-height: 1.8;
            font-size: 16px;
        }
        .category-badge {
            display: inline-block;
            padding: 10px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 25px;
            font-weight: 600;
            font-size: 16px;
            margin: 15px 0;
        }
        .scores-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .score-item {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .score-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
            text-transform: capitalize;
        }
        .score-value {
            font-size: 32px;
            font-weight: 700;
            color: #667eea;
        }
        .score-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 1s ease;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .info-item {
            background: #f8f9fa;
            padding: 18px;
            border-radius: 12px;
        }
        .info-label {
            color: #999;
            font-size: 13px;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .info-value {
            color: #333;
            font-size: 16px;
            font-weight: 600;
        }
        .action-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .btn {
            padding: 18px 30px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #f8f9fa;
            color: #667eea;
            border: 2px solid #667eea;
        }
        .btn-secondary:hover {
            background: #667eea;
            color: white;
        }
        .next-steps {
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeInUp 0.6s 0.4s both;
        }
        .next-steps h3 {
            color: #333;
            font-size: 24px;
            margin-bottom: 20px;
        }
        .step-list {
            list-style: none;
        }
        .step-item {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            margin-bottom: 15px;
            display: flex;
            align-items: start;
        }
        .step-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin-right: 20px;
            flex-shrink: 0;
        }
        .step-content {
            flex: 1;
        }
        .step-title {
            color: #333;
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 5px;
        }
        .step-desc {
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }
        @media (max-width: 768px) {
            .header-card { padding: 40px 25px; }
            .header-card h1 { font-size: 28px; }
            .header-card .user-name { font-size: 22px; }
            .result-card { padding: 25px; }
            .scores-container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Success Header -->
        <div class="header-card">
            <div class="success-icon">✓</div>
            <h1>Assessment Complete!</h1>
            <div class="user-name">{{ user.name }}</div>
            <div class="subtitle">
                Thank you for completing your assessment. We've analyzed your responses 
                and identified the best ways we can support you.
            </div>
        </div>

        <div class="results-grid">
            <!-- Primary Need Card -->
            <div class="result-card">
                <div class="card-header">
                    <div class="card-icon">🎯</div>
                    <div class="card-title">Your Primary Need</div>
                </div>
                <div class="card-content">
                    {% if user.assessment_results and user.assessment_results.top_category %}
                        <div class="category-badge">
                            {{ user.assessment_results.top_category.replace('_', ' ').title() }}
                        </div>
                        <p style="margin-top: 20px;">
                            Based on your responses, we've identified this as your most pressing need. 
                            We'll prioritize connecting you with resources and community members who can 
                            provide the best support in this area.
                        </p>
                    {% else %}
                        <p>Assessment results are being processed...</p>
                    {% endif %}
                </div>
            </div>

            <!-- Scores Breakdown Card -->
            {% if user.assessment_results and user.assessment_results.scores %}
            <div class="result-card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <div class="card-title">Detailed Assessment Scores</div>
                </div>
                <div class="card-content">
                    <div class="scores-container">
                        {% for category, score in user.assessment_results.scores.items() %}
                        <div class="score-item">
                            <div class="score-label">{{ category.replace('_', ' ') }}</div>
                            <div class="score-value">{{ score }}</div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: {{ (score / 15 * 100) if score <= 15 else 100 }}%"></div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    <p style="margin-top: 25px; color: #666;">
                        These scores help us understand your needs across different support categories. 
                        Higher scores indicate areas where you may benefit from additional support and connections.
                    </p>
                </div>
            </div>
            {% endif %}

            <!-- Profile Summary Card -->
            <div class="result-card">
                <div class="card-header">
                    <div class="card-icon">👤</div>
                    <div class="card-title">Your Profile Summary</div>
                </div>
                <div class="card-content">
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">User ID</div>
                            <div class="info-value">{{ user.user_id }}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Home Country</div>
                            <div class="info-value">{{ user.country }}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Current Location</div>
                            <div class="info-value">{{ user.location }}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Migration Type</div>
                            <div class="info-value">{{ user.migration_type }}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Current Status</div>
                            <div class="info-value">{{ user.status }}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Languages</div>
                            <div class="info-value">{{ user.languages|join(', ') if user.languages is iterable and user.languages is not string else user.languages }}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Next Steps Card -->
        <div class="next-steps">
            <h3>🚀 What Happens Next?</h3>
            <ul class="step-list">
                <li class="step-item">
                    <div class="step-number">1</div>
                    <div class="step-content">
                        <div class="step-title">Matching in Progress</div>
                        <div class="step-desc">
                            Our system is now searching for community members and resources that best 
                            match your needs and can provide support in your primary category.
                        </div>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-number">2</div>
                    <div class="step-content">
                        <div class="step-title">Review Your Matches</div>
                        <div class="step-desc">
                            Soon you'll be able to view potential connections, including people who have 
                            similar experiences and organizations offering relevant services.
                        </div>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-number">3</div>
                    <div class="step-content">
                        <div class="step-title">Connect & Grow</div>
                        <div class="step-desc">
                            Start conversations, join support groups, and access resources tailored to 
                            your specific needs and background.
                        </div>
                    </div>
                </li>
                <li class="step-item">
                    <div class="step-number">4</div>
                    <div class="step-content">
                        <div class="step-title">Build Your Network</div>
                        <div class="step-desc">
                            As you engage with the community, we'll continue to suggest new connections 
                            and resources that can support your journey.
                        </div>
                    </div>
                </li>
            </ul>
        </div>

        <!-- Action Buttons -->
        <div class="action-buttons">
            <a href="/dashboard?user_id={{ user.user_id }}" class="btn btn-primary">
                View Your Dashboard
            </a>
            <a href="/matches?user_id={{ user.user_id }}" class="btn btn-primary">
                See Your Matches
            </a>
            <a href="/" class="btn btn-secondary">
                Return to Home
            </a>
        </div>
    </div>

    <script>
        window.addEventListener('load', () => {
            document.querySelectorAll('.score-fill').forEach(bar => {
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = width;
                }, 100);
            });
        });
    </script>
</body>
</html>
"""

MATCHES_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect - Your Matches</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .page-header {
            background: white;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        .page-header h1 {
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }
        .page-header p {
            color: #666;
            font-size: 16px;
        }
        .loading {
            text-align: center;
            padding: 60px;
            color: white;
            font-size: 18px;
        }
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .matches-grid {
            display: grid;
            gap: 25px;
        }
        .match-card {
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: slideIn 0.5s;
            position: relative;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .match-rank {
            position: absolute;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 24px;
        }
        .match-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }
        .match-avatar {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            color: white;
            font-weight: 700;
            margin-right: 20px;
        }
        .match-info {
            flex: 1;
        }
        .match-name {
            font-size: 24px;
            color: #333;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .match-location {
            color: #666;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .compatibility-bar {
            background: #e0e0e0;
            height: 8px;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .compatibility-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 1s ease;
        }
        .compatibility-text {
            font-size: 14px;
            color: #667eea;
            font-weight: 600;
            margin-top: 5px;
        }
        .match-summary {
            color: #333;
            line-height: 1.7;
            margin-bottom: 20px;
            font-size: 15px;
        }
        .shared-interests {
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 14px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .interest-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .interest-tag {
            background: #f0f4ff;
            color: #667eea;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
        }
        .icebreaker-box {
            background: #fff8e1;
            border-left: 4px solid #ffa726;
            padding: 18px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .icebreaker-icon {
            font-size: 20px;
            margin-right: 8px;
        }
        .icebreaker-text {
            color: #333;
            line-height: 1.6;
            font-size: 14px;
        }
        .contact-info {
            background: #f8f9fa;
            padding: 18px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .contact-btn {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Poppins', sans-serif;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .no-matches {
            background: white;
            border-radius: 20px;
            padding: 60px 40px;
            text-align: center;
        }
        .no-matches-icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        .no-matches h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .no-matches p {
            color: #666;
            line-height: 1.6;
        }
        .support-box {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
            margin-bottom: 60px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border-left: 8px solid #667eea;
            animation: slideIn 0.8s;
        }
        .support-box h3 {
            color: #333;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .support-box p {
            color: #666;
            line-height: 1.6;
            font-size: 15px;
        }
        .support-box a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        .support-box a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="page-header">
            <h1>🎯 Your Perfect Matches</h1>
            <p>Based on advanced NLP analysis, here are your top recommendations</p>
        </div>

        <div id="loadingDiv" class="loading">
            <div class="spinner"></div>
            <p>Finding your perfect matches...</p>
        </div>

        <div id="matchesContainer" class="matches-grid" style="display: none;"></div>
        
        <div id="supportBox" class="support-box" style="display: none;"></div>
        
        <div id="noMatchesDiv" class="no-matches" style="display: none;">
            <div class="no-matches-icon">🔍</div>
            <h2>No Matches Yet</h2>
            <p>We couldn't find any matches at the moment. This might be because:</p>
            <ul style="text-align: left; margin: 20px auto; max-width: 400px; color: #666;">
                <li>There aren't enough users in the system yet</li>
                <li>Your profile is being processed</li>
                <li>No compatible matches are available right now</li>
            </ul>
            <p style="margin-top: 20px;">Check back soon as new users join!</p>
        </div>
    </div>

    <script>
        const userId = new URLSearchParams(window.location.search).get('user_id');

        async function loadMatches() {
            try {
                const response = await fetch(`/api/matches/${userId}`);
                const data = await response.json();

                document.getElementById('loadingDiv').style.display = 'none';

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                if (data.matches && data.matches.length > 0) {
                    displayMatches(data.matches);
                    showSupportBox(data.goal);
                } else {
                    document.getElementById('noMatchesDiv').style.display = 'block';
                }
            } catch (error) {
                console.error('Error loading matches:', error);
                document.getElementById('loadingDiv').innerHTML = 
                    '<p style="color: white;">Error loading matches. Please try again.</p>';
            }
        }

        function displayMatches(matches) {
            const container = document.getElementById('matchesContainer');
            container.style.display = 'grid';

            matches.forEach((match, index) => {
                const card = document.createElement('div');
                card.className = 'match-card';
                
                const initial = match.name ? match.name.charAt(0).toUpperCase() : '?';
                const sharedInterests = match.shared_interests || [];
                const compatibility = match.compatibility_percentage || 0;

                card.innerHTML = `
                    <div class="match-rank">${index + 1}</div>
                    
                    <div class="match-header">
                        <div class="match-avatar">${initial}</div>
                        <div class="match-info">
                            <div class="match-name">${match.name || 'User'}</div>
                            <div class="match-location">
                                📍 ${match.profile?.location || 'Unknown'} | 
                                🌍 From ${match.profile?.country || 'Unknown'}
                            </div>
                            <div class="compatibility-bar">
                                <div class="compatibility-fill" style="width: ${compatibility}%"></div>
                            </div>
                            <div class="compatibility-text">${compatibility}% Compatible</div>
                        </div>
                    </div>

                    <div class="match-summary">
                        ${match.summary || ''}
                    </div>

                    <div class="icebreaker-box">
                        <div class="section-title">💬 How to start a conversation</div>
                        <div class="icebreaker-text">
                            ${match.icebreaker || 'Introduce yourself and talk about your interests!'}
                        </div>
                    </div>

                    <div class="contact-info">
                        <div class="section-title">📞 Contact Information</div>
                        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
                            ${match.email ? `<div style="color: #333;">✉️ Email: <strong>${match.email}</strong></div>` : ''}
                            ${match.profile?.phone ? `<div style="color: #333;">📱 Phone: <strong>${match.profile.phone}</strong></div>` : ''}
                        </div>
                    </div>
                `;

                container.appendChild(card);
            });

            setTimeout(() => {
                document.querySelectorAll('.compatibility-fill').forEach(bar => {
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => bar.style.width = width, 100);
                });
            }, 300);
        }

        function showSupportBox(goal) {
            const box = document.getElementById('supportBox');
            const info = {
                'mental_health': {
                    title: '💡 Important Information',
                    text: 'You can always call the anonymous support service at <strong>+1 234 567 89 01</strong>.'
                },
                'legal_support': {
                    title: '⚖️ Legal Assistance',
                    text: 'You can ask your questions related to documents to the lawyer on duty at <strong>+2 123 456 78 90</strong> or by email <a href="mailto:lawyer@example.com">lawyer@example.com</a>. You can also find more information on the <a href="http://cityinfo.com" target="_blank">cityinfo.com</a> website.'
                },
                'language_assistance': {
                    title: '🗣️ Language Practice',
                    text: 'Besides matching with partners here, you can also search for local language clubs and tandem meetings in the city to practice your skills!'
                },
                'provide_legal_support': {
                    title: '🌟 Peer Mentor Role',
                    text: 'Thank you for offering your help! You have been matched with people who specifically need legal and administrative guidance. Your experience can make a huge difference in their journey.'
                }
            };

            const content = info[goal];
            if (content) {
                box.innerHTML = `
                    <h3>${content.title}</h3>
                    <p>${content.text}</p>
                `;
                box.style.display = 'block';
            }
        }

        loadMatches();
    </script>
    <a href="/" class="start-over-btn">🔄 Start Over</a>

    <style>
        .start-over-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: white;
            color: #667eea;
            padding: 15px 25px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .start-over-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            background: #f8f9fa;
        }
    </style>
</body>
</html>
"""

def init_files():
    if not os.path.exists('data/users.json'):
        with open('data/users.json', 'w') as f:
            json.dump([], f)
    
    if not os.path.exists('data/decision_tree.json'):
        default_tree = {
            "version": "1.0",
            "categories": ["social_connection", "legal_support", "mental_health", "language_support"],
            "general_questions": [
                {
                    "id": "q1",
                    "question": "What is your primary goal today?",
                    "options": [
                        {"text": "Meet new people", "score": {"social_connection": 5, "mental_health": 2}},
                        {"text": "Find legal help", "score": {"legal_support": 5}},
                        {"text": "Learn a language", "score": {"language_support": 5, "social_connection": 2}},
                        {"text": "Talk to someone", "score": {"mental_health": 5, "social_connection": 3}}
                    ]
                }
            ],
            "category_specific_questions": {}
        }
        with open('data/decision_tree.json', 'w') as f:
            json.dump(default_tree, f, indent=2)

def read_users():
    if os.path.exists('data/users.json'):
        with open('data/users.json', 'r') as f:
            return json.load(f)
    return []

def write_users(users):
    with open('data/users.json', 'w') as f:
        json.dump(users, f, indent=2)

def translate_text(text, target_language):
    try:
        if target_language == 'English' or not text:
            return text
        
        lang_code = LANGUAGE_CODES.get(target_language, 'en')
        if lang_code == 'en':
            return text
            
        translated = GoogleTranslator(source='en', target=lang_code).translate(text)
        print(f"Translated '{text[:50]}...' to '{translated[:50]}...'")
        return translated
    except Exception as e:
        print(f"Translation error for '{text}': {e}")
        return text

def translate_batch(texts, target_language, max_workers=10):
    global translations_cache
    
    try:
        if target_language == 'English':
            return texts
        
        lang_code = LANGUAGE_CODES.get(target_language, 'en')
        if lang_code == 'en':
            return texts
        
        if lang_code not in translations_cache:
            translations_cache[lang_code] = {}
        
        print(f"\n=== Translation Request ===")
        print(f"Target Language: {target_language} ({lang_code})")
        print(f"Total texts: {len(texts)}")
        
        translated = []
        texts_to_translate = []
        text_indices = []
        
        cached_count = 0
        for i, text in enumerate(texts):
            if not text or text.strip() == '':
                translated.append(text)
                continue
            
            if text in translations_cache[lang_code]:
                translated.append(translations_cache[lang_code][text])
                cached_count += 1
                print(f"✓ Using cached: '{text[:30]}...'")
            else:
                translated.append(None) 
                texts_to_translate.append(text)
                text_indices.append(i)
        
        print(f"Cached: {cached_count}/{len(texts)}")
        print(f"To translate: {len(texts_to_translate)}")
        
        if texts_to_translate:
            print(f"Starting parallel translation with {max_workers} workers...")
            
            def translate_single(text, index):
                try:
                    result = GoogleTranslator(source='en', target=lang_code).translate(text)
                    print(f"[{index+1}/{len(texts_to_translate)}] '{text[:30]}...' -> '{result[:30]}...'")
                    return (index, text, result)
                except Exception as e:
                    print(f"Error translating '{text[:30]}...': {e}")
                    return (index, text, text)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(translate_single, text, i) 
                    for i, text in enumerate(texts_to_translate)
                ]
                
                for future in as_completed(futures):
                    index, original_text, translated_text = future.result()
                    actual_index = text_indices[index]
                    translated[actual_index] = translated_text
                    translations_cache[lang_code][original_text] = translated_text
            
            save_translations_cache(translations_cache)
            print(f"✓ Completed and cached {len(texts_to_translate)} translations")
        
        return translated
    except Exception as e:
        print(f"Batch translation error: {e}")
        return texts

@app.route('/')
def index():
    return render_template_string(REGISTRATION_TEMPLATE)

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        required = ['name', 'email', 'country', 'location', 'age', 'status', 'preferred_language']
        for field in required:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        users = read_users()
        if any(u['email'] == data['email'] for u in users):
            return jsonify({'error': 'Email already registered'}), 400
        
        # Robust ID generation to avoid collisions
        max_id = 0
        for u in users:
            uid = u.get('user_id', '')
            if uid.startswith('USER'):
                try:
                    num = int(uid.replace('USER', ''))
                    if num > max_id:
                        max_id = num
                except ValueError:
                    continue
        
        user_id = f"USER{max_id + 1:04d}"
        
        new_user = {
            'user_id': user_id,
            'name': data['name'],
            'email': data['email'],
            'phone': data.get('phone', ''),
            'country': data['country'],
            'location': data['location'],
            'age': data['age'],
            'status': data['status'],
            'profession': data.get('profession', ''),
            'languages': data['languages'],
            'preferred_language': data['preferred_language'],
            'goal': data.get('goal', 'social_connection'), # Added 'goal' field
            'registered_at': datetime.now().isoformat(),
            'assessment_completed': False,
            'adaptive_answers': [] 
        }
        
        users.append(new_user)
        write_users(users)
        
        return jsonify({'message': 'Registration successful', 'user_id': new_user['user_id']}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-user-language')
def get_user_language():
    user_id = request.args.get('user_id')
    users = read_users()
    user = next((u for u in users if u['user_id'] == user_id), None)
    
    if user:
        return jsonify({'language': user.get('preferred_language', 'English')})
    return jsonify({'language': 'English'})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        texts = data.get('texts', [])
        target_language = data.get('target_language', 'English')
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        translated_texts = translate_batch(texts, target_language)
        return jsonify(translated_texts)
    except Exception as e:
        print(f"Translation endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/questions')
def questions_page():
    """New adaptive questions page"""

    return render_template_string(ADAPTIVE_QUESTIONS_TEMPLATE)

@app.route('/api/get-first-question', methods=['POST'])
def get_first_question():
    try:
        data = request.json
        user_id = data.get('user_id')
        
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        engine = AdaptiveQuestionEngine()
        question = engine.get_first_question(user)
        
        return jsonify({'question': question}), 200
        
    except Exception as e:
        print(f"Error getting first question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-next-question', methods=['POST'])
def get_next_question():
    try:
        data = request.json
        user_id = data.get('user_id')
        previous_answers = data.get('previous_answers', [])
        
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        question_num = len(previous_answers) + 1
        
        if question_num == 4:
            print("\n=== After Q3: Generating all remaining questions ===")
            engine = AdaptiveQuestionEngine()
            generated_questions = engine.generate_remaining_questions(user, previous_answers)
            
            if not generated_questions:
                print("PANIC: generated_questions is None!")
                generated_questions = [] # Safety
                
            user['generated_questions'] = generated_questions
            write_users(users)
            
            if not generated_questions:
                # Fallback directly in case something went very wrong
                q = {"id": "Q4", "question": "Could you tell me more about your background?", "type": "open_text"}
                return jsonify({'question': q}), 200

            return jsonify({'question': generated_questions[0]}), 200
        
        elif question_num >= 5 and question_num <= 7:
            generated_questions = user.get('generated_questions', [])
            
            if generated_questions and len(generated_questions) >= (question_num - 3):
                question = generated_questions[question_num - 4]
                print(f"✓ Returning cached Q{question_num}: {question['question'][:50]}...")
                return jsonify({'question': question}), 200
            else:
                print(f"Warning: Cache missing for Q{question_num}, regenerating...")
                engine = AdaptiveQuestionEngine()
                generated_questions = engine.generate_remaining_questions(user, previous_answers[:3])
                user['generated_questions'] = generated_questions
                write_users(users)
                
                if not generated_questions or len(generated_questions) < (question_num - 3):
                     print(f"PANIC: Still no questions after regeneration for Q{question_num}")
                     return jsonify({'error': 'Failed to generate questions'}), 500

                if not generated_questions or len(generated_questions) < (question_num - 3):
                    return jsonify({'error': 'Question missing'}), 500
                return jsonify({'question': generated_questions[question_num - 4]}), 200
        
        else:
            engine = AdaptiveQuestionEngine()
            question = engine.get_next_question(user, previous_answers)
            
            if question:
                return jsonify({'question': question}), 200
            else:
                return jsonify({'completed': True}), 200
            
    except Exception as e:
        print(f"Error generating next question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/complete-questions', methods=['POST'])
def complete_questions():
    try:
        data = request.json
        user_id = data.get('user_id')
        answers = data.get('answers', [])
        
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        print(f"\n=== Processing {len(answers)} answers for {user_id} ===")
        
        user['adaptive_answers'] = answers
        user['assessment_completed'] = True
        
        try:
            print("1. Running NLP analysis...")
            
            nlp_profile = analyze_profile(
                user_data=user,
                assessment_answers=answers,
                detailed_answers=answers 
            )
            
            user['nlp_profile'] = nlp_profile
            print(f"   ✓ Summary: {nlp_profile.get('summary', '')[:60]}...")
            print(f"   ✓ Preferences: {nlp_profile.get('preferences', [])[:3]}")
            print(f"   ✓ Constraints: {nlp_profile.get('constraints', [])[:2]}")
            
            print("2. Generating embedding...")
            add_user_to_index(user_id, user, nlp_profile)
            print("   ✓ Profile indexed and matchable")
            
        except Exception as e:
            print(f"NLP analysis error: {e}")
            import traceback
            traceback.print_exc()
        
        write_users(users)
        print("=== Profile complete! ===")
        
        return jsonify({'message': 'Profile completed', 'success': True}), 200
        
    except Exception as e:
        print(f"Error completing questions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/submit-assessment', methods=['POST'])
def submit_assessment():
    try:
        data = request.json
        user_id = data['user_id']
        
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if user:
            user['assessment_completed'] = True
            user['assessment_results'] = {
                'answers': data['answers'],
                'scores': data['scores'],
                'top_category': data['top_category'],
                'completed_at': datetime.now().isoformat()
            }

            try:
                print(f"Running NLP analysis for {user_id}...")
                
                detailed_answers = user.get('detailed_answers', [])
                
                nlp_profile = analyze_profile(
                    user_data=user,
                    assessment_answers=data['answers'],
                    detailed_answers=detailed_answers
                )
                
                user['nlp_profile'] = nlp_profile
                
                add_user_to_index(user_id, user, nlp_profile)
                
                print(f"✓ NLP analysis complete for {user_id}")
                print(f"  Summary: {nlp_profile.get('summary', '')[:50]}...")
                
            except Exception as e:
                print(f"NLP analysis error: {e}")
            
            write_users(users)
            return jsonify({'message': 'Assessment completed'}), 200
        
        return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    user_id = request.args.get('user_id')
    users = read_users()
    user = next((u for u in users if u['user_id'] == user_id), None)
    
    if not user:
        return "User not found", 404
    
    return render_template_string(RESULTS_TEMPLATE, user=user)


@app.route('/api/adaptive-questions', methods=['POST'])
def adaptive_questions():
    try:
        data = request.json
        user_id = data.get('user_id')
        assessment_answers = data.get('assessment_answers', [])
        
        questions = get_adaptive_questions(assessment_answers, num=4)
        
        return jsonify({'questions': questions}), 200
    except Exception as e:
        print(f"Adaptive questions error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit-detailed-answers', methods=['POST'])
def submit_detailed_answers():
    try:
        data = request.json
        user_id = data.get('user_id')
        detailed_answers = data.get('detailed_answers', [])
        
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user['detailed_answers'] = detailed_answers
        
        try:
            nlp_profile = analyze_profile(
                user_data=user,
                assessment_answers=user.get('assessment_results', {}).get('answers', []),
                detailed_answers=detailed_answers
            )
            
            user['nlp_profile'] = nlp_profile
            add_user_to_index(user_id, user, nlp_profile)
            
            print(f"✓ Updated NLP profile with detailed answers for {user_id}")
        except Exception as e:
            print(f"NLP update error: {e}")
        
        write_users(users)
        return jsonify({'message': 'Detailed answers saved'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matches/<user_id>')
def get_matches(user_id):
    try:
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not user.get('assessment_completed'):
            return jsonify({'error': 'Assessment not completed'}), 400
        
        matches = get_user_matches(user_id, users, top_n=3)
        
        return jsonify({
            'user_id': user_id,
            'goal': user.get('goal', 'social_connection'),
            'matches': matches,
            'total_candidates': len([u for u in users if u.get('assessment_completed') and u['user_id'] != user_id])
        }), 200
        
    except Exception as e:
        print(f"Matching error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/user-profile/<user_id>')
def get_user_profile(user_id):
    try:
        users = read_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        profile = {
            'user_id': user['user_id'],
            'name': user.get('name'),
            'location': user.get('location'),
            'country': user.get('country'),
            'status': user.get('status'),
            'languages': user.get('languages', []),
            'nlp_profile': user.get('nlp_profile', {}),
            'assessment_completed': user.get('assessment_completed', False)
        }
        
        return jsonify(profile), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-decision-tree')
def get_decision_tree():
    """Get the decision tree configuration"""
    try:
        with open('data/decision_tree.json', 'r') as f:
            tree = json.load(f)
        return jsonify(tree)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/matches')
def matches_page():
    user_id = request.args.get('user_id')
    if not user_id:
        return "User ID required", 400
    
    return render_template_string(MATCHES_TEMPLATE, user_id=user_id)

if __name__ == '__main__':
    init_files()
    app.run(debug=True, port=5000)