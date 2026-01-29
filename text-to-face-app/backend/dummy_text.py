import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Dummy Text Model")

class RequestPayload(BaseModel):
    prompt: str

@app.post("/generate-description")
def generate(payload: RequestPayload):
    print(f"📝 Received prompt: {payload.prompt}")
    
    # Simulating a model response
    dummy_response = (
        f"DUMMY OUTPUT: A suspect matching the description '{payload.prompt[:20]}...' "
        "is a 30-year-old male with short dark hair and a scar on his left cheek."
    )
    
    return {"description": dummy_response}

if __name__ == "__main__":
    print("🚀 Dummy Text API running on Port 5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)