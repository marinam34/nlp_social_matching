# Connect - NLP-Powered Newcomer Matching Platform

A web platform that helps newcomers in a city find friends and connections by analyzing their profiles and matching them with compatible people using **Natural Language Processing (NLP)**.

The project was completed for Technische Hochschule Würzburg-Schweinfurt.

---

## 🚀 Quick Start

### 1. Install Dependencies

Ensure you have Python 3.8+ installed.

```bash
# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure API Key

Get an API key from [OpenRouter](https://openrouter.ai/keys) and create a `.env` file in the root directory:

```bash
# Create .env file and add your key
OPENROUTER_API_KEY=sk-or-v1-your_key_here
```

### 3. Run the Application

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

---

## 📋 How It Works

1. **Registration**: User provides basic details and selects one of **7 specialized goals** (e.g., Social Connection, Legal Support Seeker/Volunteer, etc.).
2. **Adaptive Interview**: A 7-question dynamic interview:
   - **3 Fixed Questions**: Goal-specific questions to build context.
   - **4 Dynamic Questions**: LLM-generated based on previous answers.
   - **Smart Navigation**: Users can go back to previous questions; the system preserves their answers and restores question history.
3. **NLP Pipeline**: **ProfileAnalyzer** performs a consolidated LLM call to:
   - Summarize the profile and extract matching-specific insights.
   - Identify **Preferences** and **Constraints**.
   - Generate vector embeddings for similarity search.
4. **Robust Matching**: Ensures 3 quality matches via a multi-tiered system:
   - **Tier 1 (Cross-Goal)**: Matches seekers with volunteers (e.g., Legal Seeker → Legal Volunteer).
   - **Tier 2 (Peer Matching)**: Matches people with the same goal (e.g., Seeker → Seeker) if volunteers are scarce.
   - **Tier 3/4 (Loose Match)**: Bypasses strict conflicts to fulfill the match count, with a disclaimer in the icebreaker.
5. **Icebreaker Generation**: Custom messages that adapt based on whether the match is a peer or a volunteer.

---

## 🌟 Key Features

- **7 Goal Categories**:
  - `Social Connection`
  - `Legal Support` & `Legal Support Volunteers`
  - `Mental Support` & `Mental Support Volunteers`
  - `Language Support` & `Language Support Volunteers`
- **Reciprocal Matching**: Logic specifically built to pair seekers with helpful volunteers.
- **Conflict Detection**: Detects semantic clashes (e.g., "Introvert" vs "Loud Groups") to ensure compatibility.
- **MMR Diversity**: Uses Maximal Marginal Relevance to ensure match variety.

---

## 📁 Project Structure

```
nlp_social_matching/
├── app.py                    # Main Flask application and UI templates
├── src/                      # Source code package
│   ├── nlp_processor.py          # Consolidated ProfileAnalyzer logic
│   ├── vector_database.py        # Embedding generation and search
│   ├── matching_engine.py        # Tiered matching, Conflicts, Ice-breakers
│   ├── adaptive_question_engine.py # Adaptive question logic
│   └── adaptive_questions_template.py # Interview UI with history tracking
├── data/                     # Data storage
│   ├── users.json                # User profiles database
│   ├── user_embeddings.json      # Vector index
│   └── translations_cache.json   # Multi-language support cache
├── requirements.txt          # Python dependencies
└── .env                      # API Credentials
```

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Deep Learning**: `sentence-transformers` (all-MiniLM-L6-v2)
- **LLM**: `xiaomi/mimo-v2-flash:free` via OpenRouter
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (with History state management)

---

## 🔌 API Endpoints

- `POST /register`: Register user.
- `POST /api/get-next-question`: Fetch next interview question.
- `POST /api/complete-questions`: Finalize profile and extraction.
- `GET /api/matches/<user_id>`: Run tiered matching.
- `GET /matches`: View matches UI.


