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

1. **Registration**: User provides basic details (name, location, status) and selects a primary goal (Social Connection, Legal Support, Language Assistance, etc.).
2. **Adaptive Interview**: The system conducts a 7-question interview:
   - **3 Fixed Questions**: Tailored to the selected goal to establish context.
   - **4 Dynamic Questions**: Generated on-the-fly by an LLM to explore specific nuances mentioned in previous answers.
3. **NLP Pipeline**: After the interview, the **ProfileAnalyzer** performs a consolidated LLM call to:
   - Summarize the profile for human reading.
   - Generate a technical matching summary.
   - Extract structured **Preferences** and **Constraints**.
   - Generate vector embeddings for fast similarity search.
4. **Robust Matching**: The system ensures 3 high-quality matches using a multi-tiered approach:
   - **Tier 1 (Cross-Goal)**: Matches seekers with providers (e.g., Legal Seeker → Legal Mentor).
   - **Tier 2 (Peer Matching)**: Matches people with similar goals if mentors are unavailable.
   - **Tier 3/4 (Loose Match)**: Bypasses strict semantic conflicts if needed to fulfill the match count, with a disclaimer in the icebreaker.
5. **Icebreaker Generation**: Custom messages are generated for each match, adapting tone based on whether the match is a peer or a mentor.

---

## 🌟 Key Features

- **Categorized Support**: Specialized logic for Legal Support, Mental Health, and Language Assistance.
- **Provider Role**: Users can register to *provide* support (e.g., "Provide Legal Support").
- **Conflict Detection**: Prevents awkward matches by detecting semantic clashes between one user's constraints and another's preferences (e.g., "Non-smoker" vs "Smoking").
- **MMR (Maximal Marginal Relevance)**: Ensures match diversity, preventing a list of 3 identical people.

---

## 📁 Project Structure

```
nlp_social_matching/
├── app.py                    # Main Flask application and UI templates
├── src/                      # Source code package
│   ├── nlp_processor.py          # Unified ProfileAnalyzer (Summaries, Prefs, Tags)
│   ├── vector_database.py        # Embedding generation and vector search
│   ├── matching_engine.py        # Tiered matching, Conflict detection, Ice-breakers
│   ├── adaptive_question_engine.py # LLM-based follow-up question generation
│   └── adaptive_questions_template.py # HTML structure for the interview
├── data/                     # Data storage
│   ├── users.json                # Main user profile database
│   ├── user_embeddings.json      # Vector index for similarity search
│   └── translations_cache.json   # Multi-language support cache
├── requirements.txt          # Python dependencies
└── .env                      # API Credentials
```

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **NLP Models**: 
  - `sentence-transformers/all-MiniLM-L6-v2` (Fast, local embeddings)
  - `xiaomi/mimo-v2-flash:free` (Optimized LLM via OpenRouter for logic and generation)
- **Libraries**: `scikit-learn`, `numpy`, `scipy`, `openai`, `python-dotenv`
- **Frontend**: Vanilla HTML5, CSS3, and JavaScript (Responsive & Modern)

---

## 🔌 API Endpoints

- `POST /register`: Register user and start session.
- `POST /api/get-next-question`: Fetch the next interview question (static or LLM-generated).
- `POST /api/complete-questions`: Finalize the interview and run the NLP extraction pipeline.
- `GET /api/matches/<user_id>`: Run the tiered matching algorithm and return 3 candidates.
- `GET /matches`: Frontend page for viewing matches and icebreakers.


