import uvicorn
from fastapi import FastAPI, HTTPException
import requests

# Import localized config and models
import config
from backend.models import CaptionRequest, ImageRequest

app = FastAPI(title="Suspect AI Gateway Modular")

@app.post("/generate-caption")
async def proxy_generate_caption(payload: CaptionRequest):
    print(f"\n[Gateway] 📥 Text Request. Forwarding to {config.TEXT_MODEL_URL}")
    try:
        # Forward to Custom Text API
        response = requests.post(config.TEXT_MODEL_URL, json={"prompt": payload.prompt_text})
        response.raise_for_status()
        
        data = response.json()
        return {"status": "success", "caption": data.get("description", "")}

    except Exception as e:
        print(f"[Gateway] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def proxy_generate_image(payload: ImageRequest):
    print(f"\n[Gateway] 🎨 Image Request. Forwarding to {config.IMAGE_MODEL_URL}")
    try:
        # Forward to Custom Image API
        # Timeout set to 120s for image generation
        response = requests.post(config.IMAGE_MODEL_URL, json={"description": payload.sdxl_prompt}, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return {"status": "success", "image_base64": data.get("image_base64", "")}

    except Exception as e:
        print(f"[Gateway] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=config.GATEWAY_HOST, port=config.GATEWAY_PORT)