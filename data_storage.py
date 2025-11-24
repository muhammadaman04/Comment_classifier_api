import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Path to shared data file
DATA_FILE = Path("app_data.json")


def load_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load posts and comments from the shared JSON file.
    Returns empty lists if the file does not exist or is invalid.
    """
    if DATA_FILE.exists():
        try:
            with DATA_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("posts", []), data.get("comments", [])
        except Exception as exc:
            print(f"Error loading data store: {exc}")
            return [], []
    return [], []


def save_data(posts: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> bool:
    """
    Persist posts and comments to the shared JSON file.
    Returns True on success, False otherwise.
    """
    payload = {"posts": posts, "comments": comments}
    try:
        with DATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"Error saving data store: {exc}")
        return False

