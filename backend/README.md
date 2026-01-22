# Text-to-Face Generation API

A modular FastAPI backend for generating realistic face images from text descriptions using LLaVa and RealVisXL.

## 🏗️ Project Structure

```
backend/
├── app/
│   ├── config.py                   # Settings & environment variables
│   ├── main.py                     # FastAPI application
│   │
│   ├── models/
│   │   ├── llava_model.py          # LLaVa 1.5 loader
│   │   └── realviz_model.py        # RealVisXL + LoRA loader
│   │
│   ├── pipelines/
│   │   ├── caption_generator.py    # LLaVa inference
│   │   └── image_generator.py      # RealVisXL inference with Compel
│   │
│   ├── routes/
│   │   ├── captions.py             # POST /api/caption
│   │   └── images.py               # POST /api/generate-image
│   │
│   └── utils/
│       └── helpers.py              # Image utilities
│
├── streamlit_app.py                # Frontend UI
├── main.py                         # Run backend
├── requirements.txt
└── .env.example
```

## 🚀 Quick Start

### Installation

1. **Clone and navigate:**
```bash
cd backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your model paths and settings
```

### Running the Backend

```bash
python main.py
```

The API will be available at `http://localhost:8000`

**Interactive API docs:** http://localhost:8000/docs

### Running the Frontend (Optional)

```bash
pip install streamlit
streamlit run streamlit_app.py
```

## 📡 API Endpoints

### 1. Health Check
```
GET /health
```
Check if API is running.

### 2. Generate Caption
```
POST /api/caption
```
Convert user facial attributes to a dense text prompt.

**Request:**
```json
{
  "attributes": [
    {
      "category": "nose",
      "attribute": "color",
      "value": "brown"
    },
    {
      "category": "eyes",
      "attribute": "color",
      "value": "blue"
    }
  ],
  "description": "Young adult"
}
```

**Response:**
```json
{
  "dense_prompt": "A detailed portrait of a face with brown nose and blue eyes. High quality, professional photography...",
  "original_input": {...},
  "message": "Caption generated successfully"
}
```

### 3. Generate Image
```
POST /api/generate-image
```
Generate image from a text prompt using RealVisXL with Compel.

**Request:**
```json
{
  "prompt": "A detailed portrait of a face with brown nose and blue eyes...",
  "height": 512,
  "width": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5
}
```

**Response:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANS...",
  "prompt": "A detailed portrait...",
  "generation_time": 45.23,
  "image_path": "/outputs/generated_images/generated_20240122_120000.png",
  "message": "Image generated successfully"
}
```

## 🔌 Enabling Models

The backend currently runs in **placeholder mode** with models disabled for testing.

### To Enable LLaVa:

1. Install transformers:
```bash
pip install torch transformers pillow
```

2. Uncomment code in `app/models/llava_model.py`

3. Uncomment code in `app/pipelines/caption_generator.py`

### To Enable RealVisXL:

1. Install diffusers:
```bash
pip install diffusers compel safetensors
```

2. Update `.env` with paths to:
   - RealVisXL model
   - LoRA weights file

3. Uncomment code in `app/models/realviz_model.py`

4. Uncomment code in `app/pipelines/image_generator.py`

## ⚙️ Configuration

Edit `.env` file to configure:

```bash
# Models
LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b
REALVIZ_MODEL_PATH=/path/to/realvizxl
LORA_WEIGHTS_PATH=/path/to/lora_weights.safetensors

# Device
DEVICE=cuda  # Options: cuda, cpu, mps

# API
API_HOST=0.0.0.0
API_PORT=8000

# Remote endpoints (optional)
CDAC_LLAVA_API=https://cdac-api.com/llava
CDAC_REALVIZ_API=https://cdac-api.com/realviz
```

## 🧪 Testing

### Test with curl:

```bash
# Caption generation
curl -X POST "http://localhost:8000/api/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": [
      {"category": "nose", "attribute": "color", "value": "brown"}
    ],
    "description": "Young adult"
  }'

# Image generation
curl -X POST "http://localhost:8000/api/generate-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A detailed portrait of a person with brown nose",
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5
  }'
```

### Test with Python:

```python
import requests

# Caption
response = requests.post("http://localhost:8000/api/caption", json={
    "attributes": [
        {"category": "nose", "attribute": "color", "value": "brown"}
    ]
})
print(response.json())

# Image
response = requests.post("http://localhost:8000/api/generate-image", json={
    "prompt": "Portrait with brown nose"
})
print(response.json()["image_path"])
```

## 📦 Dependencies

**Core:**
- FastAPI 0.104.1
- Uvicorn 0.24.0
- Pydantic 2.5.0

**Optional (uncomment in requirements.txt when using models):**
- torch 2.1.0
- transformers 4.35.0
- diffusers 0.24.0
- compel 0.1.10
- Pillow 10.1.0

## 🔄 Workflow

```
Streamlit UI
    ↓
    └─→ POST /api/caption (user attributes)
        ↓
        └─→ LLaVa generates dense prompt
            ↓
            ↓← POST /api/generate-image (dense prompt)
            ↓
            └─→ RealVisXL + Compel generates image
                ↓
                └─→ Returns base64 image + metadata
                    ↓
                    └─→ Display in Streamlit
```

## 🛠️ Extending the App

### Add new routes:
Create `app/routes/new_feature.py`:
```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-endpoint")
async def new_endpoint(request):
    return {"message": "success"}
```

Include in `app/main.py`:
```python
from app.routes import new_feature
app.include_router(new_feature.router, prefix="/api", tags=["feature"])
```

### Add new models:
1. Create `app/models/new_model.py`
2. Implement loader function with caching
3. Import in relevant pipeline
4. Add configuration to `.env`

## 📝 Notes

- Models are cached globally to avoid repeated loading
- Image generation uses Compel for better prompt understanding
- All images are saved locally with timestamps
- API supports CORS for frontend integration
- Placeholder mode allows testing without GPU/models

## 🐛 Troubleshooting

**API won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000  # On Mac/Linux
netstat -ano | findstr :8000  # On Windows
```

**Model loading errors:**
- Ensure CUDA/PyTorch is properly installed
- Check model paths in `.env`
- Verify sufficient VRAM for model

**Slow image generation:**
- Reduce `num_inference_steps` (faster, lower quality)
- Use smaller resolution
- Enable CPU offloading for diffusion models

## 📄 License

[Add your license here]

## 👥 Contributing

[Add contribution guidelines here]
