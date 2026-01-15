"""
Adaptive Question Engine
Generates truly adaptive questions using LLM
"""

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
    """
    Generates truly adaptive questions that build on previous answers
    Uses LLM to analyze answers and create relevant follow-up questions
    """
    
    def __init__(self, model="deepseek/deepseek-r1-0528:free"):
        self.model = model
        self.total_questions = 7  # 3 fixed + 4 generated
        
        # Fixed questions that everyone gets (1-3)
        self.fixed_questions = [
            {
                'id': 'Q1',
                'question': "Tell me about your hobbies and interests. What do you enjoy doing in your free time?",
                'type': 'open_text'
            },
            {
                'id': 'Q2',
                'question': "Are there any activities, topics, or social situations you definitely want to avoid? This helps us find compatible matches.",
                'type': 'open_text'
            },
            {
                'id': 'Q3',
                'question': "How often are you available to meet new people? What times work best for you (weekdays/weekends, mornings/evenings)?",
                'type': 'open_text'
            }
        ]
    
    def get_first_question(self, user_data: Dict) -> Dict:
        """
        Get the first fixed question
        
        Args:
            user_data: Basic registration info (name, country, status, etc.)
            
        Returns:
            Question dict with id and text
        """
        return self.fixed_questions[0]
    
    def get_next_question(self, user_data: Dict, previous_answers: List[Dict]) -> Optional[Dict]:
        """
        Get next question - either fixed or generated
        
        Args:
            user_data: User's basic info
            previous_answers: List of {question, answer} dicts
            
        Returns:
            Next question dict or None if all questions done
        """
        question_num = len(previous_answers) + 1
        
        # Stop after 7 questions
        if question_num > self.total_questions:
            return None
        
        # Questions 1-3: Return fixed questions
        if question_num <= 3:
            return self.fixed_questions[question_num - 1]
        
        # Questions 4-7: Generate based on previous answers
        return self.generate_next_question(user_data, previous_answers)
    
    def generate_remaining_questions(self, user_data: Dict, previous_answers: List[Dict]) -> List[Dict]:
        """
        Generate ALL 4 remaining questions (Q4-Q7) in one batch after Q3
        This is much faster than generating one at a time
        
        Args:
            user_data: User's basic info
            previous_answers: Should be exactly 3 answers (Q1-Q3)
            
        Returns:
            List of 4 question dicts
        """
        # Build context from the 3 fixed questions
        context = self._build_context(user_data, previous_answers)
        
        # Generate all 4 questions at once
        prompt = f"""{context}

TASK: Based on the 3 answers above, generate exactly 4 follow-up questions (Q4, Q5, Q6, Q7).

These questions MUST:
1. Dig deeper into SPECIFIC interests they mentioned (e.g., if they said "reading", ask what genres)
2. Ask about their preferred social settings and group sizes
3. Explore what they value in friendships
4. Identify lifestyle details that affect compatibility

BE SPECIFIC based on their answers. If they mentioned reading, ask about genres. If they mentioned movies, ask about types of movies.

Format EXACTLY like this (no extra text):
Q4: [specific question about their hobby/interest]
Q5: [question about social preferences]
Q6: [question about values in friendship]
Q7: [question about lifestyle/dietary/schedule details]

Example based on "I like reading and watching movies":
Q4: What genres of books and movies do you enjoy most - fiction, documentaries, thrillers, or something else?
Q5: Do you prefer discussing books and movies one-on-one, in small groups, or larger book clubs?
Q6: What qualities do you value most in people you connect with?
Q7: Are there any dietary preferences or schedules we should know about for planning meetups?

NOW generate 4 questions for THIS person:"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You generate follow-up questions. Output ONLY the 4 questions in Q4:/Q5:/Q6:/Q7: format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000, # Increased for Thinking Models
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            print(f"\n=== LLM Raw Response ===\n{result}\n")
            
            # Remove thinking tokens <think>...</think> if present
            import re
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            
            # Parse - be flexible with format
            questions = []
            lines = result.split('\n')
            
            for line in lines:
                line = line.strip()
                # Try to match Q4:, Q5:, Q6:, Q7:
                if line.startswith('Q4:') or line.startswith('Q4.') or line.lower().startswith('q4:'):
                    q_text = line[3:].strip() if ':' in line[:4] else line[3:].strip()
                    questions.append({'id': 'Q4', 'question': q_text, 'type': 'open_text'})
                elif line.startswith('Q5:') or line.startswith('Q5.') or line.lower().startswith('q5:'):
                    q_text = line[3:].strip() if ':' in line[:4] else line[3:].strip()
                    questions.append({'id': 'Q5', 'question': q_text, 'type': 'open_text'})
                elif line.startswith('Q6:') or line.startswith('Q6.') or line.lower().startswith('q6:'):
                    q_text = line[3:].strip() if ':' in line[:4] else line[3:].strip()
                    questions.append({'id': 'Q6', 'question': q_text, 'type': 'open_text'})
                elif line.startswith('Q7:') or line.startswith('Q7.') or line.lower().startswith('q7:'):
                    q_text = line[3:].strip() if ':' in line[:4] else line[3:].strip()
                    questions.append({'id': 'Q7', 'question': q_text, 'type': 'open_text'})
            
            # Validate we got exactly 4 questions
            if len(questions) == 4:
                print(f"✓ Successfully generated 4 personalized questions")
                for q in questions:
                    print(f"  {q['id']}: {q['question'][:60]}...")
                return questions
            else:
                print(f"Warning: Expected 4 questions, got {len(questions)} - using fallback")
                return self._get_fallback_questions_batch()
                
        except Exception as e:
            print(f"Error generating batch questions: {e}")
            # Do NOT raise, just return fallback
            return self._get_fallback_questions_batch()
    
    def _get_fallback_questions_batch(self) -> List[Dict]:
        """Fallback questions if batch generation fails"""
        return [
            {'id': 'Q4', 'question': 'What specific activities or hobbies would you like to do with new friends?', 'type': 'open_text'},
            {'id': 'Q5', 'question': 'Do you prefer one-on-one conversations, small groups of 3-5, or larger social gatherings?', 'type': 'open_text'},
            {'id': 'Q6', 'question': 'What do you value most in friendships - shared interests, similar values, or complementary personalities?', 'type': 'open_text'},
            {'id': 'Q7', 'question': 'Is there anything else about your lifestyle or preferences that would help us find good matches for you?', 'type': 'open_text'}
        ]
    
    def generate_next_question(self, user_data: Dict, previous_answers: List[Dict]) -> Optional[Dict]:
        """
        DEPRECATED: Now we batch-generate Q4-Q7
        This is kept for fallback only
        """
        question_num = len(previous_answers) + 1
        
        # This shouldn't be called anymore, but just in case
        fallback_questions = self._get_fallback_questions_batch()
        if question_num >= 4 and question_num <= 7:
            return fallback_questions[question_num - 4]
        
        return None
    
    def _build_context(self, user_data: Dict, previous_answers: List[Dict]) -> str:
        """Build conversation context for LLM"""
        context = f"""User Profile:
- Name: {user_data.get('name', 'User')}
- From: {user_data.get('country', 'Unknown')}
- Status: {user_data.get('status', 'Unknown')}
- Languages: {', '.join(user_data.get('languages', []))}

Conversation so far:"""
        
        for qa in previous_answers:
            context += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"
        
        return context
    
    def _get_fallback_question(self, question_num: int, previous_answers: List[Dict]) -> Dict:
        """Fallback questions if LLM fails"""
        fallback_questions = [
            "How would you describe your social style? Do you prefer one-on-one hangouts, small groups, or larger gatherings?",
            "Are there any specific activities or topics you're NOT interested in? This helps us avoid mismatches.",
            "Tell me about your schedule and availability. How often would you like to meet up with new connections?",
            "What's most important to you in friendships? What qualities do you value in the people you spend time with?",
            "Is there anything else about your lifestyle, preferences, or interests that would help us find your best matches?"
        ]
        
        idx = min(question_num - 1, len(fallback_questions) - 1)
        
        return {
            'id': f'Q{question_num}',
            'question': fallback_questions[idx],
            'type': 'open_text'
        }
    
    def extract_insights_for_matching(self, all_answers: List[Dict]) -> Dict:
        """
        Analyze all answers to extract key insights for matching
        This is called after all questions are answered
        
        Returns:
            Dict with extracted insights (preferences, constraints, key_facts)
        """
        # Combine all answers
        conversation = "\n\n".join([
            f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            for qa in all_answers
        ])
        
        prompt = f"""Below is a dialog with a user looking for new friends and connections in the city.

{conversation}

TASK: Based on these answers, extract key information for matching compatible people.

Analyze the answers and highlight:

1. PREFERENCES (what is important, what they like, interests):
   - Hobbies and interests
   - Preferred activities
   - What they are looking for in friends/connections
   - Lifestyle

2. CONSTRAINTS (what does NOT fit, what they avoid):
   - Things they DISLIKE
   - Places/activities they avoid
   - Deal-breakers (e.g. no alcohol, dislike noisy places)

3. KEY_FACTS (important facts about personality and lifestyle):
   - Personality traits (introvert/extrovert, active/calm)
   - Schedule/availability
   - Lifestyle features

Output format (STRICTLY):
PREFERENCES:
- [preference 1]
- [preference 2]
- [preference 3]
...

CONSTRAINTS:
- [constraint 1]
- [constraint 2]
...

KEY_FACTS:
- [fact 1]
- [fact 2]
- [fact 3]
...

Write concisely and specifically. Use a dash list format."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in text analysis and extracting key information about people."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            print(f"\n=== Extracted insights ===\n{result}\n")
            return self._parse_extraction_result(result)
            
        except Exception as e:
            print(f"Error extracting insights: {e}")
            import traceback
            traceback.print_exc()
            return {
                'preferences': [],
                'constraints': [],
                'key_facts': []
            }
    
    def _parse_extraction_result(self, result: str) -> Dict:
        """Parse the LLM extraction result into structured data"""
        insights = {
            'preferences': [],
            'constraints': [],
            'key_facts': []
        }
        
        current_section = None
        
        for line in result.split('\n'):
            line = line.strip()
            
            if 'PREFERENCES:' in line:
                current_section = 'preferences'
            elif 'CONSTRAINTS:' in line:
                current_section = 'constraints'
            elif 'KEY_FACTS:' in line:
                current_section = 'key_facts'
            elif line.startswith('-') and current_section:
                # Extract the item (remove the dash and clean up)
                item = line[1:].strip()
                if item:
                    # Lowercase preferences for natural sentence usage
                    if current_section == 'preferences':
                        item = item[0].lower() + item[1:] 
                    
                    insights[current_section].append(item)
        
        return insights


# Convenience function
def get_next_adaptive_question(user_data: Dict, previous_answers: List[Dict]) -> Optional[Dict]:
    """
    Get the next adaptive question based on conversation history
    
    Args:
        user_data: User's profile data
        previous_answers: List of previous Q&A pairs
        
    Returns:
        Next question or None if conversation is complete
    """
    engine = AdaptiveQuestionEngine()
    
    if not previous_answers:
        return engine.get_first_question(user_data)
    else:
        return engine.generate_next_question(user_data, previous_answers)


if __name__ == "__main__":
    print("Adaptive Question Engine loaded successfully!")
    print("This engine generates intelligent follow-up questions based on user responses.")
