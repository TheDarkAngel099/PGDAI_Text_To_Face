# Project Structure Documentation

## Directory Layout

```
PGDAI_Text_To_Face/
│
├── 📄 app.py                          ⭐ Main launcher script
├── 📄 start.bat                       🪟 Windows shortcut launcher
├── 📄 requirements.txt                📦 Main project dependencies
├── 📄 README.md                       📚 Full documentation
├── 📄 QUICKSTART.md                   🚀 Quick start guide
│
├── 📁 backend/                        🔧 FastAPI Backend
│   ├── 📄 main.py                     Entry point for backend
│   ├── 📄 requirements.txt            Backend dependencies
│   ├── 📄 .env.example                Configuration template
│   ├── 📄 README.md                   Backend documentation
│   │
│   └── 📁 app/                        Main application package
│       ├── 📄 __init__.py
│       ├── 📄 config.py               Settings from environment
│       ├── 📄 main.py                 FastAPI app factory
│       │
│       ├── 📁 models/                 Model loaders
│       │   ├── 📄 __init__.py
│       │   ├── 📄 llava_model.py      LLaVa 1.5 loader
│       │   └── 📄 realviz_model.py    RealVisXL + LoRA loader
│       │
│       ├── 📁 pipelines/              Inference pipelines
│       │   ├── 📄 __init__.py
│       │   ├── 📄 caption_generator.py LLaVa inference
│       │   └── 📄 image_generator.py   RealVisXL inference
│       │
│       ├── 📁 routes/                 API endpoints
│       │   ├── 📄 __init__.py
│       │   ├── 📄 captions.py         POST /api/caption
│       │   └── 📄 images.py           POST /api/generate-image
│       │
│       ├── 📁 schemas/                Data validation
│       │   ├── 📄 __init__.py
│       │   └── 📄 requests.py         Pydantic models
│       │
│       └── 📁 utils/                  Utilities
│           ├── 📄 __init__.py
│           └── 📄 helpers.py          Image utilities
│
├── 📁 frontend/                       🎨 Streamlit Frontend
│   ├── 📄 forensic_app.py             🔍 Main forensics application
│   ├── 📄 streamlit_app.py            🎨 Simple image generator
│   └── 📄 requirements.txt            Frontend dependencies
│
├── 📁 outputs/                        Generated files (auto-created)
│   └── 📁 generated_images/           Generated face images
│
└── 📁 tests/                          Test directory (optional)
```

---

## File Descriptions

### Root Level

| File | Purpose |
|------|---------|
| `app.py` | **Main launcher** - Starts both backend and frontend with one command |
| `start.bat` | **Windows launcher** - Double-click to start on Windows |
| `requirements.txt` | **Main dependencies** - Install first: `pip install -r requirements.txt` |
| `README.md` | **Full documentation** - Complete project guide |
| `QUICKSTART.md` | **Quick guide** - 5-minute setup instructions |

### Backend (`backend/`)

| File | Purpose |
|------|---------|
| `main.py` | Entry point - Runs FastAPI with uvicorn |
| `requirements.txt` | Backend-specific packages |
| `.env.example` | Configuration template - copy to `.env` to customize |
| `README.md` | Backend documentation |
| `app/config.py` | Loads environment variables and settings |
| `app/main.py` | FastAPI application factory |

#### Models (`backend/app/models/`)

| File | Purpose |
|------|---------|
| `llava_model.py` | Loads LLaVa 1.5 model (commented out for testing) |
| `realviz_model.py` | Loads RealVisXL + LoRA weights (commented out for testing) |

**Currently in placeholder mode** - Uncomment when models are available

#### Pipelines (`backend/app/pipelines/`)

| File | Purpose |
|------|---------|
| `caption_generator.py` | Converts facial features to dense prompts |
| `image_generator.py` | Generates images from prompts using RealVisXL |

#### Routes (`backend/app/routes/`)

| File | Purpose |
|------|---------|
| `captions.py` | `POST /api/caption` - Generate description from attributes |
| `images.py` | `POST /api/generate-image` - Generate face image from prompt |

#### Schemas (`backend/app/schemas/`)

| File | Purpose |
|------|---------|
| `requests.py` | Pydantic models for request/response validation |

#### Utils (`backend/app/utils/`)

| File | Purpose |
|------|---------|
| `helpers.py` | Image utilities (encoding, saving, validation) |

### Frontend (`frontend/`)

| File | Purpose |
|------|---------|
| `forensic_app.py` | **Main application** - 4-tab forensic face description system |
| `streamlit_app.py` | **Alternative** - Simpler image generation interface |
| `requirements.txt` | Frontend dependencies (streamlit, requests, pillow) |

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         STREAMLIT FRONTEND (Forensic App)                   │
│  - Tab 1: Demographics                                       │
│  - Tab 2: Face Features (3D hierarchical)                   │
│  - Tab 3: Image Generation                                  │
│  - Tab 4: Summary & Export                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
          HTTP POST /api/caption
          HTTP POST /api/generate-image
                     │
┌────────────────────▼────────────────────────────────────────┐
│           FASTAPI BACKEND (app.py)                          │
│                                                              │
│  Routes:                                                    │
│  ├── /health                                                │
│  ├── /api/caption                                           │
│  └── /api/generate-image                                    │
│                                                              │
│  Pipelines:                                                 │
│  ├── caption_generator.py                                   │
│  │   └── Uses LLaVa (when enabled)                         │
│  └── image_generator.py                                     │
│      └── Uses RealVisXL + Compel (when enabled)           │
└────────────────────┬────────────────────────────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
    ┌──────▼────────┐  ┌──────▼────────┐
    │ LLaVa Model   │  │ RealVisXL      │
    │ (Disabled)    │  │ + LoRA Weights │
    │               │  │ (Disabled)     │
    └───────────────┘  └────────────────┘
            │                 │
    Dense Caption     Generated Image
```

---

## Running Different Configurations

### Configuration 1: Everything in One Click
```bash
python app.py
```
- Checks dependencies
- Starts backend
- Starts frontend
- Opens browser

### Configuration 2: Separate Terminals
**Terminal 1:**
```bash
python app.py --backend-only
```

**Terminal 2:**
```bash
python app.py --frontend-only
```

### Configuration 3: Direct Commands
**Terminal 1:**
```bash
cd backend
python main.py
```

**Terminal 2:**
```bash
cd frontend
streamlit run forensic_app.py
```

### Configuration 4: Alternative Frontend
**Terminal 1:**
```bash
cd backend
python main.py
```

**Terminal 2:**
```bash
cd frontend
streamlit run streamlit_app.py
```

---

## Adding New Features

### Add a New API Endpoint

1. Create route in `backend/app/routes/new_feature.py`:
```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-endpoint")
async def new_endpoint(request: YourModel):
    # Implementation
    return {"result": "success"}
```

2. Include in `backend/app/main.py`:
```python
from app.routes import new_feature
app.include_router(new_feature.router, prefix="/api")
```

### Add a New Face Feature

Edit `frontend/forensic_app.py`:
```python
FACE_FEATURES_DB["New Part"] = {
    "attribute1": ["value1", "value2", "Other"],
    "attribute2": ["value1", "value2", "Other"]
}
```

### Add a New Model

1. Create loader in `backend/app/models/new_model.py`
2. Implement with caching for efficiency
3. Add to pipelines as needed
4. Update `.env` with model paths

---

## Environment Variables

Create `backend/.env`:

```bash
# Models
LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b
REALVIZ_MODEL_PATH=/path/to/realvizxl
LORA_WEIGHTS_PATH=/path/to/lora_weights.safetensors

# Device (cuda, cpu, mps)
DEVICE=cuda

# Server
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Image Generation
DEFAULT_HEIGHT=512
DEFAULT_WIDTH=512
DEFAULT_INFERENCE_STEPS=50
DEFAULT_GUIDANCE_SCALE=7.5
```

---

## Dependencies Overview

### Backend
- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **pydantic** - Data validation
- **python-dotenv** - Environment variables
- **torch, transformers, diffusers** - ML models (optional)

### Frontend
- **streamlit** - UI framework
- **requests** - HTTP client
- **pillow** - Image handling

---

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/api/caption` | POST | Generate caption from attributes |
| `/api/generate-image` | POST | Generate image from prompt |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

---

## Output Files

Generated files are stored in:
- `outputs/generated_images/` - Generated face images
- File naming: `generated_YYYYMMDD_HHMMSS.png` or `placeholder_YYYYMMDD_HHMMSS.png`

---

## Key Design Principles

✅ **Modular** - Each component is independent  
✅ **Scalable** - Easy to add new features  
✅ **Testable** - Separation of concerns  
✅ **Production-Ready** - Logging, error handling, validation  
✅ **Placeholder-Ready** - Can test without GPU/models  
✅ **Well-Documented** - Clear code comments and guides  

---

## Next Steps

1. **Test**: Run `python app.py` and test the workflow
2. **Customize**: Modify face features in `forensic_app.py`
3. **Enable Models**: Uncomment code in model loaders when ready
4. **Deploy**: Docker, cloud, or local server
5. **Extend**: Add new routes, features, models as needed

---

## Support

- Check **QUICKSTART.md** for common tasks
- Check **README.md** for full documentation
- Check **backend/README.md** for API details
- View interactive API docs at `http://localhost:8000/docs`
