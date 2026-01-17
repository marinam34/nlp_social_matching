import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)

class ProfileAnalyzer:
    def __init__(self, model="xiaomi/mimo-v2-flash:free"):
        self.model = model
    
    def analyze_profile(self, user_data: Dict[str, Any], detailed_answers: List[Dict]) -> Dict[str, Any]:
        name = user_data.get('name', 'User')
        country = user_data.get('country', 'Unknown')
        location = user_data.get('location', 'Unknown')
        age = user_data.get('age', 'Unknown')
        status = user_data.get('status', 'Unknown')
        profession = user_data.get('profession', 'Not specified')
        languages = ", ".join(user_data.get('languages', []))
        goal = user_data.get('goal', 'social_connection').replace('_', ' ')
        
        answers_text = "\n".join([
            f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
            for qa in detailed_answers
        ])

        prompt = f"""TASK: Perform a comprehensive analysis of a user profile for a social matching app.
        
User Info:
- Name: {name}, Age: {age}
- From: {country}, Living in: {location}
- Status: {status}, Profession: {profession}
- Primary Goal: {goal}
- Languages: {languages}

Interview Answers:
{answers_text}

TASK: Extract and generate the following in ONE response:
1. SUMMARY: A concise (2-3 sentences), engaging profile summary.
2. MATCHING_SUMMARY: A factual, one-sentence summary focused on matching criteria.
3. PREFERENCES: A list of 3-5 specific interests or functional preferences.
4. CONSTRAINTS: A list of things to avoid or limitations.
5. KEY_FACTS: A list of 3-5 factual tidbits about them.

Format EXACTLY like this (use dash list for items, no extra text):
SUMMARY: [text]
MATCHING_SUMMARY: [text]
PREFERENCES:
- [item]
CONSTRAINTS:
- [item]
KEY_FACTS:
- [item]
"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert NLP analyzer for social matching. Extract insights accurately and concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            if not hasattr(response, 'choices') or not response.choices:
                raise Exception("No choices in LLM response")

            result = response.choices[0].message.content
            if not result:
                raise Exception("Empty LLM response")

            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            return self._parse_comprehensive_result(result, name, country, location, status, age)

        except Exception as e:
            print(f"Comprehensive analysis error: {e}")
            return self._get_fallback_profile(name, country, location, status, age)

    def _parse_comprehensive_result(self, text: str, name, country, location, status, age) -> Dict:
        profile = {
            'summary': '',
            'matching_summary': '',
            'preferences': [],
            'constraints': [],
            'key_facts': []
        }
        
        current_section = None
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            
            if line.upper().startswith('SUMMARY:'):
                profile['summary'] = line[len('SUMMARY:'):].strip()
                current_section = 'summary'
            elif line.upper().startswith('MATCHING_SUMMARY:'):
                profile['matching_summary'] = line[len('MATCHING_SUMMARY:'):].strip()
                current_section = 'matching_summary'
            elif line.upper().startswith('PREFERENCES:'):
                current_section = 'preferences'
            elif line.upper().startswith('CONSTRAINTS:'):
                current_section = 'constraints'
            elif line.upper().startswith('KEY_FACTS:'):
                current_section = 'key_facts'
            elif line.startswith('-') and current_section in ['preferences', 'constraints', 'key_facts']:
                content = re.sub(r'^\s*\*\*.*?\*\*[:\-]?\s*', '', line[1:].strip(), flags=re.IGNORECASE).strip()
                if content:
                    profile[current_section].append(content)
        
        if not profile['summary']:
            profile['summary'] = f"{name} from {country}, currently in {location}. {status} looking for connections."
        if not profile['matching_summary']:
            profile['matching_summary'] = f"User {age} years old from {country}. {status}."
        if not profile['key_facts'] and profile['matching_summary']:
            profile['key_facts'] = [profile['matching_summary']]
            
        return profile

    def _get_fallback_profile(self, name, country, location, status, age) -> Dict:
        return {
            'summary': f"{name} from {country}, currently in {location}. {status}, looking to connect with new people.",
            'matching_summary': f"User {age} years old from {country}. {status}.",
            'preferences': [],
            'constraints': [],
            'key_facts': [f"User {age} years old from {country}. {status}."]
        }

def analyze_profile(user_data: Dict, assessment_answers: List[Dict], 
                   detailed_answers: List[Dict] = None) -> Dict:
    analyzer = ProfileAnalyzer()
    return analyzer.analyze_profile(user_data, detailed_answers or assessment_answers)

if __name__ == "__main__":
    print("NLP Processor module loaded successfully!")
    print("Optimized for single-call comprehensive analysis.")