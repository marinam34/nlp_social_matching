import json
from typing import Dict, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)


class AdaptiveQuestionEngine:
    def __init__(self, model="xiaomi/mimo-v2-flash:free"):
        self.model = model
        self.total_questions = 7 
        

        self.category_questions = {
            'social_connection': [
                {'id': 'Q1', 'question': "Tell me about your hobbies and interests. What do you enjoy doing in your free time?", 'type': 'open_text'},
                {'id': 'Q2', 'question': "Are there any activities, topics, or social situations you definitely want to avoid?", 'type': 'open_text'},
                {'id': 'Q3', 'question': "How often are you available to meet new people? What times work best for you?", 'type': 'open_text'}
            ],
            'legal_support': [
                {'id': 'Q1', 'question': "What specific area do you need legal assistance with? (e.g., University regulations, Job/Work contracts, Migration/Visa, Housing)", 'type': 'open_text'},
                {'id': 'Q2', 'question': "Are you looking for general information/advice or do you have a specific active case or dispute?", 'type': 'open_text'},
                {'id': 'Q3', 'question': "How urgent is your situation, and what specific outcome are you hoping for?", 'type': 'open_text'}
            ],
            'mental_health': [
                {'id': 'Q1', 'question': "How have you been feeling lately? Are you looking for someone to talk to, or professional resources?", 'type': 'open_text'},
                {'id': 'Q2', 'question': "What kind of support environment do you prefer? (e.g., One-on-one peer support, group sessions, or finding therapy options)", 'type': 'open_text'},
                {'id': 'Q3', 'question': "Are there specific topics or stressors you want to focus on or avoid in your conversations?", 'type': 'open_text'}
            ],
            'language_assistance': [
                {'id': 'Q1', 'question': "Which language do you want to improve, and what is your current approximate level?", 'type': 'open_text'},
                {'id': 'Q2', 'question': "What are your main goals? (e.g., passing an exam, daily conversation, professional/business communication)", 'type': 'open_text'},
                {'id': 'Q3', 'question': "Do you prefer a formal tutoring style or casual conversation exchange with native speakers?", 'type': 'open_text'}
            ],
            'provide_legal_support': [
                {'id': 'Q1', 'question': "What specific areas of law or administration can you help with? (e.g., Uni regulations, Job contracts, Visa, Housing)", 'type': 'open_text'},
                {'id': 'Q2', 'question': "What is your background or experience? (e.g., Law student, professional lawyer, or personal experience with similar cases)", 'type': 'open_text'},
                {'id': 'Q3', 'question': "What is your availability, and do you prefer one-time consultations or ongoing mentorship?", 'type': 'open_text'}
            ]
        }

    def _get_fixed_questions(self, goal: str) -> List[Dict]:
        return self.category_questions.get(goal, self.category_questions['social_connection'])
    
    def get_first_question(self, user_data: Dict) -> Dict:
        goal = user_data.get('goal', 'social_connection')
        return self._get_fixed_questions(goal)[0]
    
    def get_next_question(self, user_data: Dict, previous_answers: List[Dict]) -> Optional[Dict]:
        question_num = len(previous_answers) + 1
        goal = user_data.get('goal', 'social_connection')
        fixed_questions = self._get_fixed_questions(goal)
        
        if question_num > self.total_questions:
            return None
        
        if question_num <= 3:
            return fixed_questions[question_num - 1]

        fallbacks = self._get_fallback_questions_batch(user_data)
        if 4 <= question_num <= 7:
            return fallbacks[question_num - 4]
        return None

    def generate_remaining_questions(self, user_data: Dict, previous_answers: List[Dict]) -> List[Dict]:
        context = self._build_context(user_data, previous_answers)
        goal = user_data.get('goal', 'social_connection').replace('_', ' ')
        
        prompt = f"""{context}
- Primary Goal: {goal}

TASK: Based on the 3 answers and the user's primary goal ({goal}) above, generate exactly 4 follow-up questions (Q4, Q5, Q6, Q7).

These questions MUST:
1. Dig deeper into SPECIFIC details they mentioned in their first 3 answers.
2. Be highly relevant to their primary goal ({goal}).
3. Explore their specific needs, preferences, or constraints related to this goal.
4. Help find the best matches or resources for them.

BE SPECIFIC based on their answers. 
- If their goal is Language Assistance, ask about specific learning methods or obstacles.
- If their goal is Legal Support, ask about their specific situation (deadlines, documents, etc).
- If their goal is Mental Health, ask about their preferred type of interaction or specific stressors mentioned.
- If their goal is Social Connection, explore hobby-specific details and social style.

Format EXACTLY like this (no extra text):
Q4: [specific question]
Q5: [question about preferences]
Q6: [question about values/style]
Q7: [question about lifestyle/schedule]
"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You generate follow-up questions. Output ONLY the 4 questions in Q4:/Q5:/Q6:/Q7: format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000, 
                temperature=0.7
            )
            
            if not hasattr(response, 'choices') or not response.choices:
                print(f"Error: OpenAI response missing choices: {response}")
                return self._get_fallback_questions_batch(user_data)

            result = response.choices[0].message.content
            if not result:
                return self._get_fallback_questions_batch(user_data)
                
            result = result.strip()
            import re
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

            questions = []
            for line in result.split('\n'):
                line = line.strip()
                match = re.match(r'Q([4-7])[:.\s]+(.+)', line, re.IGNORECASE)
                if match:
                    q_id = f"Q{match.group(1)}"
                    q_text = match.group(2).strip()
                    questions.append({'id': q_id, 'question': q_text, 'type': 'open_text'})
            
            if len(questions) == 4:
                return questions
            
            print(f"Warning: Expected 4 questions, got {len(questions)} - using fallback")
            return self._get_fallback_questions_batch(user_data)
                
        except Exception as e:
            print(f"Error generating LLM questions: {e}")
            return self._get_fallback_questions_batch(user_data)

    def _get_fallback_questions_batch(self, user_data: Dict) -> List[Dict]:
        goal = user_data.get('goal', 'social_connection')
        
        fallbacks = {
            'legal_support': [
                {'id': 'Q4', 'question': 'Are there any specific documents you already have or need help preparing?', 'type': 'open_text'},
                {'id': 'Q5', 'question': 'Have you already consulted with any other offices or organizations regarding this matter?', 'type': 'open_text'},
                {'id': 'Q6', 'question': 'What is your preferred way of receiving help: a one-time consultation or ongoing support?', 'type': 'open_text'},
                {'id': 'Q7', 'question': 'Are there any specific deadlines we should be aware of for your case?', 'type': 'open_text'}
            ],
            'mental_health': [
                {'id': 'Q4', 'question': 'What specific methods do you usually find helpful for managing stress or difficult emotions?', 'type': 'open_text'},
                {'id': 'Q5', 'question': 'Are you looking for support from people who have had similar experiences (peer support)?', 'type': 'open_text'},
                {'id': 'Q6', 'question': 'How much time per week are you willing to dedicate to these supportive conversations?', 'type': 'open_text'},
                {'id': 'Q7', 'question': 'Are there any particular times of day when you feel you need more support than usual?', 'type': 'open_text'}
            ],
            'language_assistance': [
                {'id': 'Q4', 'question': 'What is your mother tongue, and do you have experience learning other languages?', 'type': 'open_text'},
                {'id': 'Q5', 'question': 'Do you prefer practicing in a group or one-on-one with a language partner?', 'type': 'open_text'},
                {'id': 'Q6', 'question': 'What is the most difficult part of language learning for you (e.g., grammar, speaking, listening)?', 'type': 'open_text'},
                {'id': 'Q7', 'question': 'How many hours a week can you realistically spend on language exchange or practice?', 'type': 'open_text'}
            ],
            'social_connection': [
                {'id': 'Q4', 'question': 'What specific activities or hobbies would you like to do with new friends?', 'type': 'open_text'},
                {'id': 'Q5', 'question': 'Do you prefer one-on-one conversations, small groups of 3-5, or larger social gatherings?', 'type': 'open_text'},
                {'id': 'Q6', 'question': 'How often do you like to meet up with friends?', 'type': 'open_text'},
                {'id': 'Q7', 'question': 'What is your preferred social environment (loud/city/active or quiet/nature/relaxed)?', 'type': 'open_text'}
            ],
            'provide_legal_support': [
                {'id': 'Q4', 'question': 'Have you ever represented anyone or provided written legal advice before?', 'type': 'open_text'},
                {'id': 'Q5', 'question': 'Are you comfortable explaining complex legal terms in simpler language?', 'type': 'open_text'},
                {'id': 'Q6', 'question': 'What languages are you most comfortable using when discussing legal matters?', 'type': 'open_text'},
                {'id': 'Q7', 'question': 'Are there any types of cases or situations you would NOT be comfortable helping with?', 'type': 'open_text'}
            ]
        }
        return fallbacks.get(goal, fallbacks['social_connection'])

    def _build_context(self, user_data: Dict, previous_answers: List[Dict]) -> str:
        context = f"""User Profile:
- Name: {user_data.get('name', 'User')}
- From: {user_data.get('country', 'Unknown')}
- Status: {user_data.get('status', 'Unknown')}
- Languages: {', '.join(user_data.get('languages', []))}

Conversation so far:"""
        for qa in previous_answers:
            context += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"
        return context

    def extract_insights_for_matching(self, all_answers: List[Dict]) -> Dict:
        conversation = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in all_answers])
        prompt = f"""Conversation:\n{conversation}\n\nTASK: Extract PREFERENCES, CONSTRAINTS, and KEY_FACTS for matching. Use dash list format."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "Extract matchmaking insights."}, {"role": "user", "content": prompt}],
                max_tokens=500
            )
            if not hasattr(response, 'choices') or not response.choices:
                return {'preferences': [], 'constraints': [], 'key_facts': []}
                
            result = response.choices[0].message.content
            if not result:
                return {'preferences': [], 'constraints': [], 'key_facts': []}
                
            return self._parse_extraction_result(result.strip())
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return {'preferences': [], 'constraints': [], 'key_facts': []}

    def _parse_extraction_result(self, result: str) -> Dict:
        import re
        res = {'preferences': [], 'constraints': [], 'key_facts': []}
        curr = None
        
        for line in result.split('\n'):
            line_str = line.strip()
            if not line_str:
                continue
            
            upper_line = line_str.upper()
            if 'PREFERENCES' in upper_line:
                curr = 'preferences'
                continue
            elif 'CONSTRAINTS' in upper_line:
                curr = 'constraints'
                continue
            elif 'KEY_FACTS' in upper_line:
                curr = 'key_facts'
                continue
            
            if line_str.startswith('-') and curr:
                content = line_str[1:].strip()
                content = re.sub(r'^\s*\*\*.*?\*\*[:\-]?\s*', '', content, flags=re.IGNORECASE)
                
                if content:
                    res[curr].append(content.strip())
        
        return res

def get_next_adaptive_question(user_data: Dict, previous_answers: List[Dict]) -> Optional[Dict]:
    engine = AdaptiveQuestionEngine()
    
    if not previous_answers:
        return engine.get_first_question(user_data)
    else:
        return engine.get_next_question(user_data, previous_answers)