"""
Utilities for player information lookup and rich display formatting.
"""

import json
import os
from typing import Dict, Any, Optional, Tuple

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "players_database.json")

# Cache database in memory
_PLAYERS_DB: Optional[Dict[str, Any]] = None


def load_database() -> Dict[str, Any]:
    global _PLAYERS_DB
    if _PLAYERS_DB is None:
        if os.path.exists(DATABASE_PATH):
            with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                _PLAYERS_DB = json.load(f)
        else:
            _PLAYERS_DB = {}
    return _PLAYERS_DB


def normalize_player_key(name: str) -> str:
    """Normalize input player name to match database keys."""
    cleaned = name.strip().replace(" ", "_").replace("-", "_")
    db = load_database()
    
    # Direct match
    if cleaned in db:
        return cleaned
    
    # Case-insensitive match
    lower_map = {k.lower(): k for k in db.keys()}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]
    
    # Substring match
    for key in db.keys():
        if key.lower() in cleaned.lower() or cleaned.lower() in key.lower():
            return key
            
    # Fuzzy match on full names
    for key, info in db.items():
        if "full_name" in info and cleaned.lower() in info["full_name"].lower():
            return key
        if "name" in info and cleaned.lower() in info["name"].lower():
            return key
            
    return cleaned


def get_player_info(name: str) -> Dict[str, Any]:
    """Retrieve full player information by name or key."""
    db = load_database()
    key = normalize_player_key(name)
    if key in db:
        return db[key]
    
    # Fallback generic info
    return {
        "name": name,
        "full_name": name,
        "nationality": "Không xác định",
        "flag": "⚽",
        "club": "Không xác định",
        "league": "N/A",
        "position": "Cầu thủ bóng đá",
        "jersey_number": 0,
        "date_of_birth": "N/A",
        "age": "N/A",
        "height": "N/A",
        "weight": "N/A",
        "preferred_foot": "N/A",
        "market_value": "N/A",
        "fifa_rating": 80,
        "stats": {
            "Pace / Tốc độ": 75,
            "Shooting / Sút": 75,
            "Passing / Chuyền": 75,
            "Dribbling / Rê bóng": 75,
            "Defending / Phòng ngự": 60,
            "Physical / Thể lực": 75
        },
        "general_stats": {
            "Pace / Tốc độ": 75,
            "Shooting / Sút": 75,
            "Passing / Chuyền": 75,
            "Dribbling / Rê bóng": 75,
            "Defending / Phòng ngự": 60,
            "Physical / Thể lực": 75
        },
        "trophies": ["Cầu thủ bóng đá chuyên nghiệp"],
        "bio": f"Thông tin cầu thủ {name} chưa được cập nhật đầy đủ trong hệ thống."
    }


def get_radar_stats(name: str) -> Tuple[list, list]:
    """Return categories and numeric values for a radar chart."""
    info = get_player_info(name)
    stats_dict = info.get("general_stats", info.get("stats", {}))
    categories = list(stats_dict.keys())
    values = [float(v) for v in stats_dict.values()]
    return categories, values
