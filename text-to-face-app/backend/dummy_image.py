import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image

app = FastAPI(title="Dummy Image Model (Large)")

class RequestPayload(BaseModel):
    description: str

def create_dummy_b64_image(width=256, height=256, color="blue") -> str:
    """Helper to create a solid color image and return as base64 string."""
    # 1. Create a new RGB image using Pillow
    img = Image.new('RGB', (width, height), color=color)
    
    # 2. Save image to an in-memory bytes buffer
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    # 3. Encode the buffer to base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Pre-generate one image to re-use (faster)
DUMMY_IMAGE_B64 = create_dummy_b64_image(width=256, height=256, color="cornflowerblue")


@app.post("/generate-image")
def generate(payload: RequestPayload):
    print(f"🎨 Received description: {payload.description[:30]}...")
    print("Generation complete. Sending 256x256 image.")
    
    # Return the large dummy image
    return {"image_base64": DUMMY_IMAGE_B64}

if __name__ == "__main__":
    print("🚀 Dummy Image API (Large) running on Port 5001")
    uvicorn.run(app, host="127.0.0.1", port=5001)