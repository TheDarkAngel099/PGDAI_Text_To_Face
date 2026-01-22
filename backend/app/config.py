"""Configuration settings for the application"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Model paths
    llava_model_path: str = "llava-hf/llava-1.5-7b"
    realviz_model_path: str = "path/to/realvizxl"
    lora_weights_path: str = "path/to/lora_weights.safetensors"
    
    # Device
    device: str = "cuda"
    
    # API endpoints (if using remote CDAC models)
    cdac_llava_api: Optional[str] = None
    cdac_realviz_api: Optional[str] = None
    
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Image generation defaults
    default_height: int = 512
    default_width: int = 512
    default_inference_steps: int = 50
    default_guidance_scale: float = 7.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
