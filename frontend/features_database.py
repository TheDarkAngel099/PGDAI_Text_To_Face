"""
Forensic Face Features Database
Contains all facial features, attributes, and demographic data
"""
from typing import Dict, List

FACE_FEATURES_DB: Dict[str, Dict[str, List[str]]] = {
    "Face Shape": {
        "shape": ["None", "Oval", "Round", "Square", "Rectangular", "Heart", "Diamond", "Triangular", "Other"]
    },
    "Forehead": {
        "height": ["None", "Low", "Medium", "High", "Very High", "Other"],
        "width": ["None", "Narrow", "Medium", "Wide", "Very Wide", "Other"],
        "features": ["None", "Wrinkles", "Scars", "Birthmarks", "Tattoo", "Other"]
    },
    "Eyes": {
        "color": ["None", "Black", "Dark Brown", "Brown", "Hazel", "Light Brown", "Green", "Blue", "Gray", "Heterochromia", "Other"],
        "shape": ["None", "Almond", "Round", "Hooded", "Monolid", "Upturned", "Downturned", "Asymmetrical", "Other"],
        "size": ["None", "Very Small", "Small", "Medium", "Large", "Very Large", "Other"],
        "distance": ["None", "Close Set", "Normal", "Wide Set", "Very Wide", "Other"],
        "eyebrows": ["None", "Thin", "Medium", "Thick", "Very Thick", "Unibrow", "Absent", "Tattooed", "Other"],
        "other_features": ["None", "Scar", "Birthmark", "Tattoo", "Glasses/Contacts", "Lazy Eye", "Other"]
    },
    "Nose": {
        "shape": ["None", "Straight", "Crooked", "Hooked", "Bulbous", "Pointed", "Flat", "Aquiline", "Other"],
        "width": ["None", "Narrow", "Medium", "Wide", "Very Wide", "Other"],
        "length": ["None", "Short", "Medium", "Long", "Very Long", "Other"],
        "bridge": ["None", "Straight", "Concave", "Convex", "Broken", "Other"],
        "tip": ["None", "Pointed", "Rounded", "Bulbous", "Split", "Other"],
        "nostrils": ["None", "Small", "Medium", "Large", "Flared", "Other"],
        "features": ["None", "Scar", "Pimple", "Birthmark", "Tattoo", "Piercing", "Other"]
    },
    "Mouth": {
        "shape": ["None", "Wide", "Normal", "Small", "Cupid Bow", "Thin Lips", "Full Lips", "Asymmetrical", "Other"],
        "upper_lip": ["None", "Thin", "Medium", "Full", "Very Full", "Protruding", "Other"],
        "lower_lip": ["None", "Thin", "Medium", "Full", "Very Full", "Protruding", "Other"],
        "color": ["None", "Pale", "Pink", "Dark Pink", "Red", "Brown", "Other"],
        "distinguishing": ["None", "Scar", "Cleft Lip", "Gap Teeth", "Tattoo", "Piercing", "Gold Teeth", "Other"]
    },
    "Cheeks": {
        "shape": ["None", "Hollow", "Normal", "Full", "Very Full", "Prominent", "Other"],
        "color": ["None", "Pale", "Normal", "Flushed", "Red", "Pigmented", "Other"],
        "features": ["None", "Acne", "Acne Scars", "Birthmark", "Tattoo", "Piercing", "Dimples", "Freckles", "Other"]
    },
    "Chin": {
        "shape": ["None", "Pointed", "Rounded", "Square", "Prominent", "Receding", "Cleft", "Other"],
        "size": ["None", "Small", "Medium", "Large", "Very Large", "Other"],
        "features": ["None", "Dimple", "Scar", "Birthmark", "Tattoo", "Beard Stubble", "Beard", "Other"],
        "beard": ["None", "Goatee", "Full Beard", "Stubble", "Van Dyke", "Soul Patch", "Other"]
    },
    "Scars & Marks": {
        "location": ["None", "Forehead", "Cheek", "Chin", "Nose", "Lips", "Eye Area", "Neck", "Multiple", "Other"],
        "type": ["None", "Scar", "Birthmark", "Tattoo", "Mole", "Wart", "Acne Scar", "Burn Mark", "Other"],
        "size": ["None", "Small (< 1 inch)", "Medium (1-2 inches)", "Large (2-3 inches)", "Very Large (> 3 inches)", "Other"],
        "appearance": ["None", "Raised", "Indented", "Flat", "Discolored", "Other"]
    },
    "Hair": {
        "color": ["None", "Black", "Dark Brown", "Brown", "Light Brown", "Blonde", "Red", "Gray", "White", "Dyed", "Other"],
        "texture": ["None", "Straight", "Wavy", "Curly", "Coily", "Kinky", "Braided", "Other"],
        "length": ["None", "Bald", "Very Short", "Short", "Medium", "Long", "Very Long", "Other"],
        "style": ["None", "Shaved", "Crew Cut", "Fade", "Afro", "Dreadlocks", "Braids", "Messy", "Combed Back", "Side Part", "Other"],
        "coverage": ["None", "Full Head", "Receding", "Widow's Peak", "Bald Spot", "Thinning", "Other"]
    }
}

DEMOGRAPHICS: Dict[str, List[str]] = {
    "Gender": ["None", "Male", "Female", "Non-Binary", "Prefer Not to Say", "Other"],
    "Race/Ethnicity": [
        "None",
        "Caucasian/White",
        "African American/Black",
        "Asian",
        "Hispanic/Latino",
        "Middle Eastern/North African",
        "Native American/Indigenous",
        "Pacific Islander",
        "South Asian",
        "Mixed Race",
        "Other"
    ],
    "Skin Tone": [
        "None",
        "Very Fair",
        "Fair",
        "Light",
        "Medium Light",
        "Medium",
        "Medium Dark",
        "Dark",
        "Very Dark",
        "Other"
    ],
    "Age Range": [
        "None",
        "Child (0-12)",
        "Teenager (13-19)",
        "Young Adult (20-30)",
        "Adult (31-45)",
        "Middle Aged (46-60)",
        "Senior (61+)",
        "Unknown"
    ],
    "Distinctive Features": [
        "Tattoos",
        "Piercings",
        "Glasses",
        "Facial Hair",
        "Scars",
        "Birthmarks",
        "Freckles",
        "Dimples",
        "Large Ears",
        "None Notable",
        "Other"
    ]
}
