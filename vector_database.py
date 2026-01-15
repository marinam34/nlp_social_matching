"""
Vector Database Module
Handles embedding generation and similarity search for user matching

NLP Technique: Semantic Embeddings & Vector Similarity Search
Uses sentence-transformers to create dense vector representations
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Initialize the embedding model
# Using all-MiniLM-L6-v2: lightweight, 384 dimensions, good for semantic similarity
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorDatabase:
    """
    Manages user embeddings and similarity search
    """
    
    def __init__(self, embeddings_file: str = "user_embeddings.json"):
        self.embeddings_file = embeddings_file
        self.model = None
        self.embeddings_data = self._load_embeddings()
    
    def _load_model(self):
        """Lazy load the embedding model (only when needed)"""
        if self.model is None:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(MODEL_NAME)
        return self.model
    
    def _load_embeddings(self) -> Dict:
        """Load embeddings from JSON file"""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self):
        """Save embeddings to JSON file"""
        with open(self.embeddings_file, 'w') as f:
            json.dump(self.embeddings_data, f, indent=2)
    
    def create_profile_text(self, user_data: Dict, nlp_profile: Dict) -> str:
        """
        Create a comprehensive text representation of the user's profile
        This text will be embedded into a vector
        
        Args:
            user_data: Basic user info
            nlp_profile: NLP-extracted profile data
            
        Returns:
            Concatenated text for embedding
        """
        parts = []
        
        # Add summary (most important)
        if nlp_profile.get('summary'):
            parts.append(nlp_profile['summary'])
        
        # Add preferences
        if nlp_profile.get('preferences'):
            prefs = " ".join(nlp_profile['preferences'][:5])
            parts.append(f"Interested in: {prefs}")
        
        # Add interests
        if nlp_profile.get('extracted_interests'):
            interests = " ".join(nlp_profile['extracted_interests'][:5])
            parts.append(f"Topics: {interests}")
        
        # Add personality traits
        if nlp_profile.get('personality_traits'):
            traits = " ".join(nlp_profile['personality_traits'])
            parts.append(f"Personality: {traits}")
        
        # Add basic info
        parts.append(f"From {user_data.get('country', '')} in {user_data.get('location', '')}")
        parts.append(f"Status: {user_data.get('status', '')}")
        
        # Add top category
        if user_data.get('assessment_results', {}).get('top_category'):
            category = user_data['assessment_results']['top_category'].replace('_', ' ')
            parts.append(f"Main need: {category}")
        
        return ". ".join(parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Input text to embed
            
        Returns:
            384-dimensional vector
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def add_user_embedding(self, user_id: str, user_data: Dict, nlp_profile: Dict):
        """
        Add or update a user's embedding in the database
        
        Args:
            user_id: User ID
            user_data: Full user data dictionary
            nlp_profile: NLP-extracted profile
        """
        # Create profile text
        profile_text = self.create_profile_text(user_data, nlp_profile)
        
        # Generate embedding
        embedding = self.generate_embedding(profile_text)
        
        # Store in database
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
        """
        Find most similar users using cosine similarity
        
        Args:
            user_id: ID of the user to find matches for
            top_k: Number of similar users to return
            exclude_self: Whether to exclude the user themselves
            
        Returns:
            List of (user_id, similarity_score, metadata) tuples
        """
        if user_id not in self.embeddings_data:
            return []
        
        query_embedding = np.array(self.embeddings_data[user_id]['embedding']).reshape(1, -1)
        
        similarities = []
        for uid, data in self.embeddings_data.items():
            # Skip self if requested
            if exclude_self and uid == user_id:
                continue
            
            # Calculate cosine similarity
            candidate_embedding = np.array(data['embedding']).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, candidate_embedding)[0][0]
            
            similarities.append((uid, float(similarity), data['metadata']))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return similarities[:top_k]
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict]:
        """Get metadata for a specific user"""
        if user_id in self.embeddings_data:
            return self.embeddings_data[user_id]['metadata']
        return None
    
    def rebuild_index(self, users_data: List[Dict]):
        """
        Rebuild the entire embedding index from scratch
        Useful when adding many users at once
        
        Args:
            users_data: List of user dictionaries with nlp_profile
        """
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
        """Get statistics about the vector database"""
        return {
            'total_users': len(self.embeddings_data),
            'embedding_dimension': 384,
            'model': MODEL_NAME
        }


# Convenience functions
def add_user_to_index(user_id: str, user_data: Dict, nlp_profile: Dict):
    """Add a user to the vector index"""
    db = VectorDatabase()
    db.add_user_embedding(user_id, user_data, nlp_profile)


def find_similar_users(user_id: str, top_k: int = 20) -> List[Tuple[str, float, Dict]]:
    """Find similar users for matching"""
    db = VectorDatabase()
    return db.search_similar_users(user_id, top_k)


if __name__ == "__main__":
    # Test the module
    print("Vector Database module loaded successfully!")
    db = VectorDatabase()
    print(f"Stats: {db.get_stats()}")
