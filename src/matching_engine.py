import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from .vector_database import VectorDatabase
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)


class ConflictDetector:
    def __init__(self):
        self.db = VectorDatabase()
        
        self.conflict_pairs = [
            (['alcohol', 'drinking', 'bar', 'wine', 'beer'], 
             ['no alcohol', "don't drink", 'sober', 'no drinking']),
            (['party', 'loud', 'club', 'social events'], 
             ['quiet', 'peaceful', 'introvert', 'small groups']),
            (['meat', 'steak', 'bbq'], 
             ['vegetarian', 'vegan', 'no meat']),
            (['smoke', 'smoking', 'cigarette'], 
             ['no smoking', 'non-smoker', "don't smoke"]),
        ]
    
    def has_conflict(self, user_a_constraints: List[str], 
                    user_b_preferences: List[str],
                    threshold: float = 0.4) -> Tuple[bool, Optional[str]]:
        if not user_a_constraints or not user_b_preferences:
            return False, None
        
        for constraint in user_a_constraints:
            constraint_lower = constraint.lower()
            for preference in user_b_preferences:
                preference_lower = preference.lower()
                
                for neg_keywords, pos_keywords in self.conflict_pairs:
                    constraint_match = any(kw in constraint_lower for kw in neg_keywords)
                    preference_match = any(kw in preference_lower for kw in pos_keywords)
                    
                    if constraint_match and preference_match:
                        return True, f"Conflict: '{constraint}' vs '{preference}'"
        
        try:
            model = self.db._load_model()
            
            for constraint in user_a_constraints[:3]:  
                constraint_emb = model.encode(constraint)
                
                for preference in user_b_preferences[:5]: 
                    pref_emb = model.encode(preference)
                    
                    similarity = np.dot(constraint_emb, pref_emb) / (
                        np.linalg.norm(constraint_emb) * np.linalg.norm(pref_emb)
                    )
                    
                    if similarity > threshold:
                        return True, f"Semantic conflict: '{constraint}' vs '{preference}' (sim: {similarity:.2f})"
        
        except Exception as e:
            print(f"Semantic conflict check error: {e}")
        
        return False, None
    
    def mutual_compatibility(self, user_a_data: Dict, user_b_data: Dict) -> bool:
        a_profile = user_a_data.get('nlp_profile') or {}
        b_profile = user_b_data.get('nlp_profile') or {}
        a_constraints = a_profile.get('constraints', [])
        a_preferences = a_profile.get('preferences', [])
        b_constraints = b_profile.get('constraints', [])
        b_preferences = b_profile.get('preferences', [])
        
        conflict_ab, _ = self.has_conflict(a_constraints, b_preferences)
        if conflict_ab:
            return False
        
        conflict_ba, _ = self.has_conflict(b_constraints, a_preferences)
        if conflict_ba:
            return False
        
        return True


class MMRSelector:
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param
        self.db = VectorDatabase()
    
    def calculate_diversity(self, candidate_embedding: np.ndarray, 
                           selected_embeddings: List[np.ndarray]) -> float:

        if not selected_embeddings:
            return 1.0
        
        similarities = []
        for selected_emb in selected_embeddings:
            similarity = np.dot(candidate_embedding, selected_emb) / (
                np.linalg.norm(candidate_embedding) * np.linalg.norm(selected_emb)
            )
            similarities.append(similarity)
        
        return 1.0 - max(similarities)
    
    def select_diverse_matches(self, query_user_id: str, 
                               candidates: List[Tuple[str, float, Dict]], 
                               top_n: int = 3) -> List[Tuple[str, float, Dict]]:
        if len(candidates) <= top_n:
            return candidates
        
        query_data = self.db.embeddings_data.get(query_user_id)
        if not query_data:
            return candidates[:top_n]
        
        query_embedding = np.array(query_data['embedding'])
        
        candidate_info = []
        for uid, relevance, metadata in candidates:
            cand_data = self.db.embeddings_data.get(uid)
            if cand_data:
                candidate_info.append({
                    'user_id': uid,
                    'relevance': relevance,
                    'metadata': metadata,
                    'embedding': np.array(cand_data['embedding'])
                })
        
        selected = []
        selected_embeddings = []
        remaining = candidate_info.copy()
        
        for _ in range(min(top_n, len(remaining))):
            best_score = -1
            best_candidate = None
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                relevance = candidate['relevance']
                
                diversity = self.calculate_diversity(
                    candidate['embedding'], 
                    selected_embeddings
                )
                
                mmr_score = (self.lambda_param * relevance + 
                           (1 - self.lambda_param) * diversity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append((
                    best_candidate['user_id'],
                    best_candidate['relevance'],
                    best_candidate['metadata']
                ))
                selected_embeddings.append(best_candidate['embedding'])
                remaining.pop(best_idx)
        
        return selected


class IceBreakerGenerator:
    
    def __init__(self, model="xiaomi/mimo-v2-flash:free"):
        self.model = model
    
    def generate_icebreaker(self, user_a_data: Dict, user_b_data: Dict, match_type: str = 'primary', is_loose_match: bool = False) -> str:
        goal_b = user_b_data.get('goal', 'social_connection').replace('_', ' ').title()
        name_b = user_b_data.get('name', 'This person')
        
        goal_prefix = f"{name_b}'s primary goal is {goal_b}. "
        
        if match_type == 'peer':
            goal_prefix = f"{name_b} is also looking for {goal_b}. You are in the same boat! "
        
        loose_warning = ""
        if is_loose_match:
            loose_warning = "Even though you might have different viewpoints on some topics, "

        a_profile = user_a_data.get('nlp_profile', {})
        b_profile = user_b_data.get('nlp_profile', {})

        a_prefs = a_profile.get('preferences', [])
        b_prefs = b_profile.get('preferences', [])
        
        shared = []
        for p1 in a_prefs:
            for p2 in b_prefs:
                words1 = set(p1.lower().split())
                words2 = set(p2.lower().split())
                if words1.intersection(words2):
                    shared.append(p1) 
                    break 

        if shared:
            topics = list(set(shared))[:2]
            clean_topics = [t.lower() for t in topics]
            topic_str = " and ".join(clean_topics)
            return f"{goal_prefix}{loose_warning}You both share interest in {topic_str}. Maybe you can compare notes!"
            
        return f"{goal_prefix}{loose_warning}Try introducing yourself and sharing what you've learned about {goal_b.lower()} so far!"
            

    
    def generate_match_explanation(self, similarity_score: float, 
                                   shared_interests: List[str]) -> str:

        percentage = int(similarity_score * 100)
        
        if shared_interests:
            interests_str = ", ".join(shared_interests[:2])
            return f"{percentage}% compatible - You both share interests in {interests_str}"
        else:
            return f"{percentage}% compatible based on your profiles"


class MatchingEngine:
    def __init__(self):
        self.db = VectorDatabase()
        self.conflict_detector = ConflictDetector()
        self.mmr_selector = MMRSelector(lambda_param=0.7)
        self.icebreaker_gen = IceBreakerGenerator()
    
    def find_matches(self, user_id: str, users_data: List[Dict], 
                    top_n: int = 3) -> List[Dict]:
        candidates = self.db.search_similar_users(user_id, top_k=20)
        
        if not candidates:
            return []

        users_dict = {u['user_id']: u for u in users_data}
        query_user = users_dict.get(user_id)
        
        if not query_user:
            return []
        
        query_goal = query_user.get('goal', 'social_connection')
        
        primary_targets = [query_goal]
        if query_goal == 'legal_support':
            primary_targets = ['provide_legal_support']
        elif query_goal == 'provide_legal_support':
            primary_targets = ['legal_support']

        primary_safe = []
        peer_safe = []
        primary_loose = []
        peer_loose = []

        for uid, similarity, metadata in candidates:
            candidate_user = users_dict.get(uid)
            if not candidate_user:
                continue
            
            cand_goal = candidate_user.get('goal', 'social_connection')
            is_primary = cand_goal in primary_targets
            is_peer = (cand_goal == query_goal)
            
            compatible = self.conflict_detector.mutual_compatibility(query_user, candidate_user)
            
            match_info = (uid, similarity, metadata)
            
            if is_primary:
                if compatible:
                    primary_safe.append(match_info + ('primary', False))
                else:
                    primary_loose.append(match_info + ('primary', True))
            elif is_peer:
                if compatible:
                    peer_safe.append(match_info + ('peer', False))
                else:
                    peer_loose.append(match_info + ('peer', True))
        
        final_candidates = []
        
        combined_pool = primary_safe + peer_safe + primary_loose + peer_loose
        
       
        diverse_raw = self.mmr_selector.select_diverse_matches(
            user_id, [c[:3] for c in combined_pool], top_n=top_n
        )
        
        pool_dict = {c[0]: (c[3], c[4]) for c in combined_pool}
        
        match_cards = []
        for uid, similarity, metadata in diverse_raw:
            matched_user = users_dict.get(uid)
            if not matched_user:
                continue
            
            match_type, is_loose = pool_dict.get(uid, ('primary', False))
            
            query_interests = set(query_user.get('nlp_profile', {}).get('preferences', []))
            match_interests = set(matched_user.get('nlp_profile', {}).get('preferences', []))
            shared = list(query_interests & match_interests)
            
            icebreaker = self.icebreaker_gen.generate_icebreaker(
                query_user, matched_user, match_type=match_type, is_loose_match=is_loose
            )
            explanation = self.icebreaker_gen.generate_match_explanation(similarity, shared)
            
            match_card = {
                'user_id': uid,
                'name': matched_user.get('name', 'Unknown'),
                'email': matched_user.get('email', ''),
                'similarity_score': float(similarity),
                'compatibility_percentage': int(similarity * 100),
                'match_type': match_type,
                'is_loose_match': is_loose,
                'summary': metadata.get('summary', ''),
                'shared_interests': shared[:3],
                'icebreaker': icebreaker,
                'match_explanation': explanation,
                'profile': {
                    'country': matched_user.get('country', ''),
                    'location': matched_user.get('location', ''),
                    'status': matched_user.get('status', ''),
                    'languages': matched_user.get('languages', []),
                    'phone': matched_user.get('phone', '') 
                }
            }
            
            match_cards.append(match_card)
        
        return match_cards


def get_user_matches(user_id: str, users_data: List[Dict], top_n: int = 3) -> List[Dict]:
    engine = MatchingEngine()
    return engine.find_matches(user_id, users_data, top_n)


if __name__ == "__main__":
    print("Matching Engine module loaded successfully!")
    print("Includes: ConflictDetector, MMRSelector, IceBreakerGenerator, MatchingEngine")