import json
import os
from pathlib import Path

# Data file path
DATA_FILE = Path("app_data.json")

def load_data():
    """Load posts and comments from JSON file"""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('posts', []), data.get('comments', [])
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []
    return [], []

def save_data(posts, comments):
    """Save posts and comments to JSON file"""
    try:
        data = {
            'posts': posts,
            'comments': comments
        }
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

