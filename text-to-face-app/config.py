import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gateway Config
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 8000))
API_BASE_URL = os.getenv("GATEWAY_URL", f"http://{GATEWAY_HOST}:{GATEWAY_PORT}")

# External Model APIs
TEXT_MODEL_URL = os.getenv("CUSTOM_TEXT_API_URL")
IMAGE_MODEL_URL = os.getenv("CUSTOM_IMAGE_API_URL")

# File Paths
CACHE_FILE_PATH = os.getenv("CACHE_FILE", "assets/cache.json")