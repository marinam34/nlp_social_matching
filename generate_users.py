
import json
import os
import sys
from datetime import datetime

# Import vector database to ensure embeddings are created
from vector_database import VectorDatabase

def generate_data():
    print("🔄 Starting data generation...")
    
    # 1. Load James
    james = None
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            users = json.load(f)
            for u in users:
                if u.get('user_id') == 'USER0016':
                    james = u
                    break
    
    if not james:
        print("❌ Error: James (USER0016) not found! Please register him first.")
        return

    print("✓ Found James data")
    
    # 2. Define Synthetic Users (Diverse profiles)
    
    # OPTION 1: David (Perfect Match)
    # Likes basketball, humor, outgoing.
    david = {
        "user_id": "USER0020",
        "name": "David",
        "email": "david@example.com",
        "phone": "+1 555 0101",
        "country": "USA",
        "location": "Wurzburg",
        "age": "27",
        "status": "Working",
        "profession": "Engineer",
        "languages": ["English", "German"],
        "preferred_language": "English",
        "registered_at": datetime.now().isoformat(),
        "assessment_completed": True,
        "adaptive_answers": [], # Skipping raw answers for synthetic users
        "generated_questions": [],
        "nlp_profile": {
            "summary": "Outgoing software engineer who loves team sports and stand-up comedy.",
            "matching_summary": "27-year-old outgoing professional. passionate about basketball and football. loves large social gatherings and comedy clubs.",
            "preferences": [
                "basketball and football",
                "comedy clubs",
                "large groups",
                "active weekends",
                "tech meetups"
            ],
            "constraints": [], # No constraints
            "key_facts": [
                "Plays basketball weekly",
                "Extroverted personality",
                "Loves humor and jokes"
            ]
        }
    }

    # OPTION 2: Sarah (Good Match - Vegan)
    # Vegan, healthy lifestyle, but maybe more introverted.
    sarah = {
        "user_id": "USER0021",
        "name": "Sarah",
        "email": "sarah@example.com",
        "phone": "+1 555 0102",
        "country": "Canada",
        "location": "Wurzburg",
        "age": "24",
        "status": "Student",
        "profession": "Biology",
        "languages": ["English", "French"],
        "preferred_language": "English",
        "registered_at": datetime.now().isoformat(),
        "assessment_completed": True,
        "adaptive_answers": [],
        "generated_questions": [],
        "nlp_profile": {
            "summary": "Biology student interested in sustainable living and hiking.",
            "matching_summary": "24-year-old student. vegan and environmentally conscious. enjoys hiking and quiet cafes. prefers small groups.",
            "preferences": [
                "vegan food",
                "hiking in nature",
                "sustainability",
                "quiet cafes",
                "morning jogs"
            ],
            "constraints": [
                "no loud parties",
                "avoids fast food"
            ],
            "key_facts": [
                "Vegan diet",
                "Introverted nature",
                "Early bird"
            ]
        }
    }

    # OPTION 3: Mike (Conflict - Alcohol)
    # Likes sports (match) but loves pubs/alcohol (conflict with James?).
    mike = {
        "user_id": "USER0022",
        "name": "Mike",
        "email": "mike@example.com",
        "phone": "+1 555 0103",
        "country": "UK",
        "location": "Wurzburg",
        "age": "26",
        "status": "Working",
        "profession": "Sales",
        "languages": ["English"],
        "preferred_language": "English",
        "registered_at": datetime.now().isoformat(),
        "assessment_completed": True,
        "adaptive_answers": [],
        "generated_questions": [],
        "nlp_profile": {
            "summary": "Social guy from UK, loves football and pub culture.",
            "matching_summary": "26-year-old sales professional. die-hard football fan. enjoys varied nightlife, especially pubs and beer tasting.",
            "preferences": [
                "football matches",
                "pub quizzes",
                "craft beer",
                "watching sports in bars",
                "nightlife"
            ],
            "constraints": [],
            "key_facts": [
                "Football fanatic",
                "Social drinker",
                "Evening person"
            ]
        }
    }

    # OPTION 4: Lisa (Conflict - Religion)
    # Religious interests (Conflict with James's 'avoid religion').
    lisa = {
        "user_id": "USER0023",
        "name": "Lisa",
        "email": "lisa@example.com",
        "phone": "+1 555 0104",
        "country": "USA",
        "location": "Wurzburg",
        "age": "23",
        "status": "Student",
        "profession": "Theology",
        "languages": ["English", "German"],
        "preferred_language": "English",
        "registered_at": datetime.now().isoformat(),
        "assessment_completed": True,
        "adaptive_answers": [],
        "generated_questions": [],
        "nlp_profile": {
            "summary": "Theology student heavily involved in church community.",
            "matching_summary": "23-year-old student. very active in local church group. enjoys choir singing and bible study. looking for spiritual connections.",
            "preferences": [
                "church events",
                "choir singing",
                "community service",
                "classical music",
                "reading groups"
            ],
            "constraints": [
                "avoids bars",
                "no late nights"
            ],
            "key_facts": [
                "Religious/Spiritual",
                "Choir member",
                "Values tradition"
            ]
        }
    }

    # OPTION 5: Alex (Neutral/Creative)
    # Photography, Art.
    alex = {
        "user_id": "USER0024",
        "name": "Alex",
        "email": "alex@example.com",
        "phone": "+1 555 0105",
        "country": "Australia",
        "location": "Wurzburg",
        "age": "25",
        "status": "Working",
        "profession": "Designer",
        "languages": ["English"],
        "preferred_language": "English",
        "registered_at": datetime.now().isoformat(),
        "assessment_completed": True,
        "adaptive_answers": [],
        "generated_questions": [],
        "nlp_profile": {
            "summary": "Graphic designer looking for inspiration and city exploration.",
            "matching_summary": "25-year-old designer. passionate about photography and street art. enjoys gallery openings and urban exploration.",
            "preferences": [
                "photography",
                "modern art",
                "coffee shops",
                "urban exploration",
                "design workshops"
            ],
            "constraints": [],
            "key_facts": [
                "Creative personality",
                "Visual thinker",
                "Weekend explorer"
            ]
        }
    }

    new_users = [james, david, sarah, mike, lisa, alex]
    
    # 3. Save to Users.json
    with open('users.json', 'w') as f:
        json.dump(new_users, f, indent=2)
    print(f"✓ Saved {len(new_users)} users to users.json")

    # 4. Rebuild Vector Index
    print("🔄 Rebuilding vector index (this creates the embeddings)...")
    db = VectorDatabase()
    # Clear existing
    db.embeddings_data = {} 
    
    for user in new_users:
        print(f"   - Processing {user['name']}...")
        db.add_user_embedding(
            user['user_id'],
            user,
            user['nlp_profile']
        )
    
    print("✅ Data generation complete! James + 5 new users ready.")

if __name__ == "__main__":
    generate_data()
