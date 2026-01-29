# Fields that must be filled before generation
MANDATORY_FIELDS = [
    "Gender", "Age Category", "Race / Ethnicity", 
    "Skin Tone", "Face Shape", "Hair Color", "Eye Color"
]

# The UI Form Structure
SUSPECT_SCHEMA = {
    "Core Demographics": {
        "Gender": ["Male", "Female"],
        "Age Category": ["19-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-96"],
        "Race / Ethnicity": ["White / Caucasian", "Black / African American", "Hispanic / Latino", "Asian", "Middle Eastern", "Native American", "Mixed / Bi-racial"],
        "Skin Tone": ["Pale / Fair", "Light", "Light to Medium", "Medium (Olive/Tan)", "Medium to Dark", "Dark / Deep"]
    },
    "Hair Characteristics": {
        "Hair Style / Length": ["Bald / Shaved Head", "Buzz Cut / Very Short", "Short (Ear length)", "Medium (Chin/Shoulder length)", "Long (Past Shoulders)", "Receding Hairline"],
        "Hair Texture": ["Straight", "Wavy", "Curly", "Coily / Kinky"],
        "Hair Color": ["Black", "Dark Brown", "Light Brown", "Blonde", "Red / Auburn", "Gray / Silver", "White"],
        "Facial Hair Style": ["Clean Shaven", "Light Stubble", "Heavy Stubble", "Full Beard", "Goatee", "Mustache Only", "Mutton Chops"]
    },
    "Face Structure": {
        "Face Shape": ["Oval", "Round", "Square / Blocky", "Long / Rectangular", "Heart-shaped"],
        "Forehead Height": ["Average", "High / Broad", "Low / Narrow"],
        "Jawline Definition": ["Soft / Round", "Average", "Strong / Chiseled", "Square"],
        "Chin Shape": ["Pointed", "Rounded", "Cleft Chin", "Receding", "Double Chin"]
    },
    "Eyes & Brows": {
        "Eye Shape": ["Round", "Almond", "Narrow / Squinting", "Hooded", "Deep-set"],
        "Eye Color": ["Dark Brown", "Light Brown / Hazel", "Blue", "Green", "Gray"],
        "Eyebrows Shape": ["Straight", "Arched", "Curved", "Unibrow"],
        "Eyebrows Thickness": ["Thin / Sparse", "Average", "Thick / Bushy"]
    },
    "Nose & Mouth": {
        "Nose Shape": ["Straight", "Button / Turned-up", "Hooked / Aquiline", "Bulbous", "Crooked"],
        "Nose Width": ["Narrow", "Average", "Wide / Broad"],
        "Lip Thickness": ["Thin", "Average", "Full / Thick", "Thick Lower Lip"]
    },
    "Accessories & Marks": {
        "Eyewear": ["Reading Glasses (Wireframe)", "Reading Glasses (Thick frame)", "Sunglasses"],
        "Scars / Marks": ["Freckles", "Mole (Cheek)", "Mole (Chin)", "Scar (Eyebrow)", "Scar (Cheek)", "Scar (Lip)", "Acne / Pockmarks"]
    }
}