# -*- coding: utf-8 -*-
import re
import json
from pathlib import Path

# Attributes ranked by importance for pruning when exceeding token budget
PRIORITY = [
    "Skin color and gender",
    "Overall facial appearance",
    "Hair",
    "Hairline",
    "Beard",
    "Eyes",
    "Eyebrows",
    "Nose",
    "Lips",
    "Chin and jawline",
    "Forehead",
    "Ears",
    "Scars",
    "Additional details",
]

# Minimal normalization per attribute

def normalize_value(key: str, value: str) -> str:
    if not value:
        return ""
    v = value.strip().rstrip('.')
    v = v.replace("\\u2019", "'")

    mapping = {
        "Skin color and gender": {
            "White male, fair": "white male, fair skin",
        },
        "Overall facial appearance": {
            "Young, approx 20": "young, about 20",
            "Adult, approx 30": "adult, about 30",
            "Middle-aged, approx 45": "middle-aged, about 45",
            "Senior, approx 65": "senior, about 65",
        },
        "Hair": {
            "Blonde, short, receding hairline": "short blonde hair, receding",
        },
        "Forehead": {
            "High, smooth": "high smooth forehead",
        },
        "Hairline": {
            "Receding, M-shape": "receding M-shaped hairline",
        },
        "Beard": {
            "Goatee, reddish-brown": "reddish-brown goatee",
        },
        "Eyes": {
            "Blue, oval, no glasses": "blue oval eyes, no glasses",
        },
        "Eyebrows": {
            "Brown, thick": "thick brown eyebrows",
        },
        "Nose": {
            "Straight, average width": "straight nose, average width",
        },
        "Lips": {
            "Thin, pink": "thin pink lips",
        },
        "Chin and jawline": {
            "Square jaw, dimpled chin": "square jaw, dimpled chin",
        },
        "Ears": {
            "Visible, attached lobes": "visible ears, attached lobes",
        },
        "Scars": {
            "None": "no scars",
        },
    }

    if key in mapping and v in mapping[key]:
        return mapping[key][v]

    # fallbacks: lowercase and simple clean-up
    v = re.sub(r"\s+", " ", v)
    return v.lower()


def compose_prompt(selected: dict, clip_token_limit: int = 73) -> str:
    """Compose a compact prompt aiming to fit within ~73 CLIP tokens.
    Conservative word-based truncation; we prune low-priority attributes first.
    """
    # base context to keep the model grounded
    prefix = "mugshot, centered, neutral lighting, realistic forensic face"

    parts = []
    for key in PRIORITY:
        v = selected.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(normalize_value(key, v))

    prompt = f"{prefix}, " + ", ".join([p for p in parts if p])

    # approximate tokenization: words split; CLIP tokens are subwords, so this is conservative
    def tokens_count(s: str) -> int:
        words = re.findall(r"[\w'\-]+", s)
        return len(words)

    # prune tail attributes until under budget
    if tokens_count(prompt) > clip_token_limit:
        # attempt to drop lowest priority pieces progressively
        for drop_key in reversed(PRIORITY):
            normalized = normalize_value(drop_key, selected.get(drop_key, ""))
            if normalized and normalized in prompt:
                prompt = re.sub(r"(,\s*)?" + re.escape(normalized), "", prompt, count=1)
            if tokens_count(prompt) <= clip_token_limit:
                break

    # final cleanup
    prompt = re.sub(r",\s*,", ", ", prompt)
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*$", "", prompt)
    return prompt.strip()
