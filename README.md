# Forensic Face Description System

A professional forensics application for documenting criminal suspect facial features using hierarchical 3D input and AI-powered image generation.

## 📁 Project Structure

```
PGDAI_Text_To_Face/
├── app.py                          # 🚀 Main launcher (starts both backend & frontend)
├── requirements.txt                # Project-wide dependencies
├── README.md                       # This file
│
├── backend/                        # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings & env variables
│   │   ├── main.py                # FastAPI app factory
│   │   │
│   │   ├── models/
│   │   │   ├── llava_model.py      # LLaVa 1.5 loader
│   │   │   └── realviz_model.py    # RealVisXL + LoRA loader
│   │   │
│   │   ├── pipelines/
│   │   │   ├── caption_generator.py # LLaVa inference
│   │   │   └── image_generator.py   # RealVisXL inference
│   │   │
│   │   ├── routes/
│   │   │   ├── captions.py         # POST /api/caption
│   │   │   └── images.py           # POST /api/generate-image
│   │   │
│   │   ├── schemas/
│   │   │   └── requests.py         # Pydantic models
│   │   │
│   │   └── utils/
│   │       └── helpers.py          # Image utilities
│   │
│   ├── main.py                    # Backend entry point
│   ├── requirements.txt           # Backend dependencies
│   ├── .env.example               # Configuration template
│   └── README.md                  # Backend documentation
│
├── frontend/                       # Streamlit Frontend
│   ├── forensic_app.py            # 🔍 Main forensics app
│   ├── streamlit_app.py           # 🎨 Simple image generator
│   ├── requirements.txt           # Frontend dependencies
│   └── config.py                  # Frontend config (optional)
│
└── outputs/                        # Generated images (created automatically)
    └── generated_images/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Launch Everything

```bash
# Start both backend and frontend
python app.py
```

This will:
- ✅ Check all dependencies
- ✅ Start FastAPI backend on port 8000
- ✅ Start Streamlit frontend on port 8501
- ✅ Open the app in your browser automatically

### Alternative Launch Options

**Backend only:**
```bash
python app.py --backend-only
```

**Frontend only (backend must be running):**
```bash
python app.py --frontend-only
```

**Custom ports:**
```bash
python app.py --backend-port 8000 --frontend-port 8501
```

**Don't open browser:**
```bash
python app.py --no-browser
```

## 📊 Application Tabs

### Tab 1: Suspect Profile (Demographics)
- Gender selection
- Race/Ethnicity classification
- Age range estimation
- Skin tone identification
- Distinctive features (tattoos, scars, piercings, etc.)
- Additional notes field

### Tab 2: Face Features
Hierarchical 3D input system with:

**Face Parts:**
- Face Shape
- Forehead (height, width, features)
- Eyes (color, shape, size, distance, eyebrows, marks)
- Nose (shape, width, length, bridge, tip, nostrils, features)
- Mouth (shape, lips, color, distinguishing marks)
- Cheeks (shape, color, features)
- Chin (shape, size, features, beard type)
- Scars & Marks (location, type, size, appearance)
- Hair (color, texture, length, style, coverage)

**Custom Input:**
- "Other" option for any dropdown to enter custom descriptions

### Tab 3: Image Generation
- Auto-generated description prompt
- LLaVa caption enhancement (optional)
- RealVisXL image generation
- Quality/guidance controls
- API health monitoring
- Image download option

### Tab 4: Summary & Export
- Complete suspect report
- Export as TXT or JSON
- Download generated images
- Clear all data option
- System information & tips

## 🔌 Backend API Endpoints

### Health Check
```
GET /health
```

### Generate Caption
```
POST /api/caption
```

**Request:**
```json
{
  "attributes": [
    {"category": "nose", "attribute": "shape", "value": "crooked"},
    {"category": "eyes", "attribute": "color", "value": "blue"}
  ],
  "description": "Young adult male"
}
```

**Response:**
```json
{
  "dense_prompt": "Criminal suspect: male, young adult; nose with shape: crooked; eyes with color: blue. High quality, professional photography...",
  "original_input": {...},
  "message": "Caption generated successfully..."
}
```

### Generate Image
```
POST /api/generate-image
```

**Request:**
```json
{
  "prompt": "Criminal suspect description...",
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
  "prompt": "...",
  "generation_time": 45.23,
  "image_path": "outputs/generated_images/generated_20240122_120000.png",
  "message": "Image generated successfully..."
}
```

## ⚙️ Configuration

### Backend Configuration

Create `.env` in `backend/` directory:

```bash
# Model Paths
LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b
REALVIZ_MODEL_PATH=/path/to/realvizxl
LORA_WEIGHTS_PATH=/path/to/lora_weights.safetensors

# Computing Device
DEVICE=cuda  # Options: cuda, cpu, mps

# Server
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Image Defaults
DEFAULT_HEIGHT=512
DEFAULT_WIDTH=512
DEFAULT_INFERENCE_STEPS=50
DEFAULT_GUIDANCE_SCALE=7.5
```

## 🤖 Model Integration

### Current Status
Models are currently in **placeholder mode** for testing without GPU/models.

### Enable LLaVa 1.5

1. Install dependencies:
```bash
pip install torch transformers
```

2. Uncomment code in `backend/app/models/llava_model.py`

3. Uncomment code in `backend/app/pipelines/caption_generator.py`

### Enable RealVisXL

1. Install dependencies:
```bash
pip install diffusers compel safetensors
```

2. Update model paths in `.env`

3. Uncomment code in `backend/app/models/realviz_model.py`

4. Uncomment code in `backend/app/pipelines/image_generator.py`

## 📚 Usage Examples

### API Testing with cURL

**Generate Caption:**
```bash
curl -X POST "http://localhost:8000/api/caption" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": [
      {"category": "nose", "attribute": "shape", "value": "hooked"}
    ],
    "description": "Adult male"
  }'
```

**Generate Image:**
```bash
curl -X POST "http://localhost:8000/api/generate-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Criminal suspect with hooked nose",
    "height": 512,
    "width": 512
  }'
```

### Python Testing

```python
import requests

# Generate caption
response = requests.post("http://localhost:8000/api/caption", json={
    "attributes": [{"category": "nose", "attribute": "shape", "value": "crooked"}]
})
print(response.json()["dense_prompt"])

# Generate image
response = requests.post("http://localhost:8000/api/generate-image", json={
    "prompt": "Criminal suspect with crooked nose"
})
image_path = response.json()["image_path"]
```

## 🎨 Theme & Styling

The forensic app uses a professional dark theme:
- Dark blue background (#1a1a2e, #16213e)
- Red accent color (#e94560)
- Professional fonts and spacing
- Color-coded sections and buttons
- Responsive layout

## 📝 Notes

- **For Law Enforcement Use Only**: This system is designed for official criminal justice purposes
- **Placeholder Mode**: Models are disabled by default for testing without GPU
- **CORS Enabled**: Backend accepts requests from any origin (update for production)
- **Local Storage**: Generated images are saved to `outputs/generated_images/`
- **Export Formats**: Supports TXT and JSON export of descriptions

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Use different ports
python app.py --backend-port 9000 --frontend-port 9501
```

### Missing Dependencies
```bash
# Install all requirements
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Backend Won't Start
- Check if models are available
- Verify CUDA is installed (if using GPU)
- Check `.env` file configuration

### Frontend Can't Connect to Backend
- Ensure backend is running on correct port
- Update API URL in frontend config
- Check firewall settings

## 📊 API Documentation

Interactive API docs available at:
```
http://localhost:8000/docs
```

Alternative (ReDoc):
```
http://localhost:8000/redoc
```

## 🚀 Deployment

### Docker (Optional)

```bash
# Build and run with Docker
docker-compose up
```

See `docker-compose.yml` for configuration.

## 📄 License

[Add your license here]

## 👥 Contributors

- Backend: [Your Name]
- Frontend: [Your Colleague]
- Project: PGDAI Text-to-Face

## 📞 Support

For issues or questions:
1. Check the system information in the app
2. Review backend logs at `http://localhost:8000/docs`
3. Verify all dependencies are installed
4. Check configuration in `.env` file
