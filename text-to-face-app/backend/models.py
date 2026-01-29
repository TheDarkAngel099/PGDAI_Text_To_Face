from pydantic import BaseModel
from typing import Dict, Any

class CaptionRequest(BaseModel):
    prompt_text: str 
    json_data: Dict[str, Any] = {} # Optional, but good to keep for logging if needed

class ImageRequest(BaseModel):
    sdxl_prompt: str