import os
import spacy
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)


class TextSummarizer:
    def __init__(self, model="deepseek/deepseek-r1-0528:free"):
        self.model = model
    
    def summarize_profile(self, user_data: Dict[str, Any], detailed_answers: List[Dict]) -> str:
        name = user_data.get('name', 'User')
        country = user_data.get('country', 'Unknown')
        location = user_data.get('location', 'Unknown')
        age = user_data.get('age', 'Unknown')
        status = user_data.get('status', 'Unknown')
        profession = user_data.get('profession', 'Not specified')
        languages = ", ".join(user_data.get('languages', []))
        
        answers_text = "\n".join([
            f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
            for qa in detailed_answers
        ])
        
        prompt = f"""TASK: Create a brief profile summary to help find compatible people.

User Information:
- Name: {name}, Age: {age}
- From: {country}, Living in: {location}
- Status: {status}, Profession/Study: {profession}
- Languages: {languages}

Their Answers:
{answers_text}

Based on this, write a concise, engaging summary (2-3 sentences) that:
1. Highlights key interests and hobbies
2. Shows what they're looking for (friends, language exchange, activities)
3. Conveys their personality and lifestyle

Good examples:
- "Active student from India, passionate about hiking and photography. Seeking friends for language exchange and nature walks around Würzburg. Prefers small groups and meaningful conversations."
- "Working professional from Brazil who loves cooking and cultural events. Wants to meet locals to practice German and share culinary traditions."

Write ONLY the summary, no headings or extra text."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating engaging profile summaries for social platforms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
            print(f"✓ Generated summary: {summary[:60]}...")
            return summary
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback summary
            return f"{name} from {country}, currently in {location}. {status}, looking to connect with new people."


class InformationExtractor:
    def __init__(self):
        self.preference_keywords = [
            'like', 'love', 'enjoy', 'prefer', 'interested', 'passion', 'hobby',
            'want', 'looking for', 'seeking', 'hope', 'wish', 'would like'
        ]
        
        self.constraint_keywords = [
            "don't", "not", "never", "avoid", "dislike", "hate", "can't",
            "won't", "refuse", "against", "no way", "prefer not"
        ]
    

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        return {'activities': [], 'locations': [], 'organizations': [], 'interests': []}

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        return []

    def extract_preferences(self, answers: List[Dict]) -> List[str]:
        return []

    def generate_key_facts_summary(self, user_data: Dict[str, Any], detailed_answers: List[Dict]) -> str:
        age = user_data.get('age', 'Unknown')
        name = user_data.get('name', 'User')
        status = user_data.get('status', 'Unknown')
        
        qa_text = ""
        for qa in detailed_answers:
            q_text = qa.get('question', '').lower()
            a_text = qa.get('answer', '')
            
            if 'avoid' in q_text or 'dislike' in q_text:
                qa_text += f"Q (AVOID): {q_text}\nA: {a_text}\n"
            else:
                qa_text += f"Q: {q_text}\nA: {a_text}\n"
                
        prompt = f"""TASK: Analyzer user data for a friendship matching app.
        
User Profile:
- Name: {name}, Age: {age}
- Status: {status}
- Profession: {user_data.get('profession', '')}
- Languages: {", ".join(user_data.get('languages', []))}

Interview Transcript:
{qa_text}

OUTPUT INSTRUCTIONS:
Create a concise, factual summary (in English) of what this user is looking for and who they are.
CRITICAL RULES:
1. If the user says they want to AVOID a topic (e.g., religion, alcohol), do NOT list it as an interest. Explicitly state "Avoids X".
2. Include Age, Status, and Activity Level.
3. Keep it under 100 words.
4. This text will be used to find similar users, so focus on matching criteria.

Example Output:
"25-year-old active student who enjoys team sports like soccer. Extroverted, prefers large groups. explicitly avoids alcohol and religious discussions. Interested in hiking wth reliable people." 
"""
        max_retries = 2
        for i in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-0528:free",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3
                )
                summary = response.choices[0].message.content.strip()
                import re
                summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
                return summary
            except Exception as e:
                if i == max_retries - 1:
                    print(f"Error generating facts summary: {e}")
                    return f"User {age} years old. {status}. Interests based on interview."
                import time
                time.sleep(2)

        keywords = self.extract_keywords(all_text)
        preferences.extend(keywords[:5])
        
        return list(set(preferences)) 
    
    def extract_constraints(self, answers: List[Dict]) -> List[str]:
        constraints = []
        
        for qa in answers:
            answer = qa.get('answer', '').lower()
            
            for keyword in self.constraint_keywords:
                if keyword in answer:
                    sentences = answer.split('.')
                    for sentence in sentences:
                        if keyword in sentence:
                            clean_sent = sentence.strip()
                            if clean_sent and len(clean_sent) > 10:
                                constraints.append(clean_sent)
                    break
        
        return list(set(constraints))
    
    def extract_personality_traits(self, answers: List[Dict]) -> List[str]:
        trait_patterns = {
            'introverted': ['quiet', 'shy', 'alone', 'peace', 'calm', 'introvert'],
            'extroverted': ['social', 'party', 'people', 'outgoing', 'extrovert', 'group'],
            'active': ['sport', 'exercise', 'hike', 'run', 'gym', 'active', 'outdoor'],
            'creative': ['art', 'music', 'creative', 'paint', 'write', 'design'],
            'intellectual': ['read', 'learn', 'study', 'book', 'science', 'discussion']
        }
        
        all_text = " ".join([qa.get('answer', '') for qa in answers]).lower()
        
        traits = []
        for trait, keywords in trait_patterns.items():
            if any(keyword in all_text for keyword in keywords):
                traits.append(trait)
        return traits


class ProfileEmbedder:
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("sentence-transformers not installed")
            self.model = None
            
    def create_embedding(self, user_profile: Dict[str, Any]) -> List[float]:
        if not self.model:
            return [0.0] * 384
            
        key_facts = user_profile.get('key_facts', [])
        rich_summary = key_facts[0] if key_facts else ""
        
        if not rich_summary:
            rich_summary = user_profile.get('summary', '')
            
        embedding = self.model.encode(rich_summary)
        return embedding.tolist()


class ProfileAnalyzer:
    def __init__(self):
        self.summarizer = TextSummarizer()
        self.extractor = InformationExtractor()
    
    def analyze_profile(self, user_data: Dict[str, Any], detailed_answers: List[Dict]) -> Dict[str, Any]:

        summary = self.summarizer.summarize_profile(user_data, detailed_answers)

        matching_summary = self.extractor.generate_key_facts_summary(user_data, detailed_answers)
        
        return {
            'summary': summary,
            'matching_summary': matching_summary, 
            'preferences': [], # Deprecated
            'constraints': [], # Deprecated
            'extracted_interests': [], # Deprecated
            'personality_traits': [],
            'key_facts': [matching_summary] 
        }

    def analyze_user_profile(self, user_data: Dict, assessment_answers: List, detailed_answers: List = None):
        return self.analyze_profile(user_data, detailed_answers or assessment_answers)


def analyze_profile(user_data: Dict, assessment_answers: List[Dict], 
                   detailed_answers: List[Dict] = None) -> Dict:

    analyzer = ProfileAnalyzer()
    return analyzer.analyze_profile(user_data, detailed_answers or assessment_answers)


if __name__ == "__main__":
    print("NLP Processor module loaded successfully!")
    print("Available classes: TextSummarizer, InformationExtractor, ProfileAnalyzer")