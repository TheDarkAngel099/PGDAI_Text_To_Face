import json
import os
import config
from data.schema import MANDATORY_FIELDS  # Import the order list

def load_cache():
    """Loads the JSON database from the configured path."""
    if os.path.exists(config.CACHE_FILE_PATH):
        try:
            with open(config.CACHE_FILE_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def get_lookup_key(data):
    """Generates a deterministic key based on MANDATORY_FIELDS order."""
    # 1. Extract only valid data
    clean_data = {
        k: v for k, v in data.items() 
        if v is not None 
        and v != "Other (Unspecified)" 
        and v != "None / Unspecified"
    }

    # 2. Create an ordered dictionary based on MANDATORY_FIELDS
    ordered_data = {}
    
    # First, add mandatory fields in the correct order
    for field in MANDATORY_FIELDS:
        if field in clean_data:
            ordered_data[field] = clean_data[field]
            
    # Then add any remaining fields (alphabetically or as they appear)
    for k, v in clean_data.items():
        if k not in ordered_data:
            ordered_data[k] = v

    # 3. Dump without sorting keys (sort_keys=False) to preserve order
    return json.dumps(ordered_data, sort_keys=False)