import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from datetime import datetime


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorDatabase:
    def __init__(self, embeddings_file: str = "data/user_embeddings.json"):
        self.embeddings_file = embeddings_file
        self.model = None
        self.embeddings_data = self._load_embeddings()
    
    def _load_model(self):
        if self.model is None:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(MODEL_NAME)
        return self.model
    
    def _load_embeddings(self) -> Dict:
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self):
        with open(self.embeddings_file, 'w') as f:
            json.dump(self.embeddings_data, f, indent=2)
    
    def create_profile_text(self, user_data: Dict, nlp_profile: Dict) -> str:
        parts = []
        
        if nlp_profile.get('summary'):
            parts.append(nlp_profile['summary'])
        
        if nlp_profile.get('preferences'):
            prefs = " ".join(nlp_profile['preferences'][:5])
            parts.append(f"Interested in: {prefs}")
        
        if nlp_profile.get('extracted_interests'):
            interests = " ".join(nlp_profile['extracted_interests'][:5])
            parts.append(f"Topics: {interests}")
        
        if nlp_profile.get('personality_traits'):
            traits = " ".join(nlp_profile['personality_traits'])
            parts.append(f"Personality: {traits}")
        
        parts.append(f"From {user_data.get('country', '')} in {user_data.get('location', '')}")
        parts.append(f"Status: {user_data.get('status', '')}")
        
        if user_data.get('assessment_results', {}).get('top_category'):
            category = user_data['assessment_results']['top_category'].replace('_', ' ')
            parts.append(f"Main need: {category}")
        
        return ". ".join(parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def add_user_embedding(self, user_id: str, user_data: Dict, nlp_profile: Dict):

        profile_text = self.create_profile_text(user_data, nlp_profile)
        
        embedding = self.generate_embedding(profile_text)
        
        self.embeddings_data[user_id] = {
            'embedding': embedding,
            'metadata': {
                'summary': nlp_profile.get('summary', ''),
                'top_category': user_data.get('assessment_results', {}).get('top_category', ''),
                'preferences': nlp_profile.get('preferences', [])[:5],
                'constraints': nlp_profile.get('constraints', [])[:3],
                'last_updated': datetime.now().isoformat()
            }
        }
        
        self._save_embeddings()
        print(f"Added embedding for user {user_id}")
    
    def search_similar_users(self, user_id: str, top_k: int = 20, 
                            exclude_self: bool = True) -> List[Tuple[str, float, Dict]]:
        if user_id not in self.embeddings_data:
            return []
        
        query_embedding = np.array(self.embeddings_data[user_id]['embedding']).reshape(1, -1)
        
        similarities = []
        for uid, data in self.embeddings_data.items():
            if exclude_self and uid == user_id:
                continue
            
            candidate_embedding = np.array(data['embedding']).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, candidate_embedding)[0][0]
            
            similarities.append((uid, float(similarity), data['metadata']))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict]:
        if user_id in self.embeddings_data:
            return self.embeddings_data[user_id]['metadata']
        return None
    
    def rebuild_index(self, users_data: List[Dict]):
        print(f"Rebuilding index for {len(users_data)} users...")
        self.embeddings_data = {}
        
        for user in users_data:
            if 'nlp_profile' in user and user.get('assessment_completed'):
                self.add_user_embedding(
                    user['user_id'],
                    user,
                    user['nlp_profile']
                )
        
        print("Index rebuild complete!")
    
    def get_stats(self) -> Dict:
        return {
            'total_users': len(self.embeddings_data),
            'embedding_dimension': 384,
            'model': MODEL_NAME
        }


def add_user_to_index(user_id: str, user_data: Dict, nlp_profile: Dict):
    db = VectorDatabase()
    db.add_user_embedding(user_id, user_data, nlp_profile)


def find_similar_users(user_id: str, top_k: int = 20) -> List[Tuple[str, float, Dict]]:
    db = VectorDatabase()
    return db.search_similar_users(user_id, top_k)


if __name__ == "__main__":
    print("Vector Database module loaded successfully!")
    db = VectorDatabase()
    print(f"Stats: {db.get_stats()}")