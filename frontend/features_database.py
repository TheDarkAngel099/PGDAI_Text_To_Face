"""
Forensic Face Features Database
Loads facial features and demographic data from JSON
"""
import json
from pathlib import Path
from typing import Dict, List

# Load database from JSON file
DB_FILE = Path(__file__).parent / "features_database.json"

with open(DB_FILE, 'r') as f:
    _db = json.load(f)

FACE_FEATURES_DB: Dict[str, Dict[str, List[str]]] = _db["face_features"]
DEMOGRAPHICS: Dict[str, List[str]] = _db["demographics"]
