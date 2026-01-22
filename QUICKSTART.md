# 🚀 Quick Start Guide

## Installation (One-Time Setup)

### Step 1: Install Python Dependencies

```bash
# Navigate to project root
cd PGDAI_Text_To_Face

# Install all dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Step 2: Configure Backend (Optional)

```bash
# Copy example configuration
cd backend
cp .env.example .env

# Edit .env with your settings (optional - defaults work for testing)
# nano .env    (or open with your editor)
```

---

## Running the Application

### Option 1: Run Everything with One Command (Recommended)

```bash
python app.py
```

This will:
- ✅ Check all dependencies
- ✅ Start Backend (FastAPI) on http://localhost:8000
- ✅ Start Frontend (Streamlit) on http://localhost:8501
- ✅ Open the app in your browser

### Option 2: Run Components Separately

**Terminal 1 - Backend:**
```bash
python app.py --backend-only
```
Backend will be available at: http://localhost:8000/docs

**Terminal 2 - Frontend:**
```bash
python app.py --frontend-only
```
Frontend will be available at: http://localhost:8501

### Option 3: Direct Execution

**Backend:**
```bash
cd backend
python main.py
```

**Frontend (in different terminal):**
```bash
cd frontend
streamlit run forensic_app.py
```

---

## Using the Application

### Main Features

1. **Tab 1 - Suspect Profile**
   - Enter demographics (gender, age, race, skin tone)
   - Select distinctive features

2. **Tab 2 - Face Features**
   - Click to expand each face part
   - Select attributes from dropdowns
   - Use "Other" to enter custom descriptions

3. **Tab 3 - Image Generation**
   - Auto-generated description from your selections
   - Generate enhanced caption with LLaVa
   - Generate image sketch with RealVisXL
   - Adjust quality settings

4. **Tab 4 - Summary & Export**
   - Review complete report
   - Export as TXT or JSON
   - Download generated images

### Typical Workflow

```
1. Suspect Profile (Tab 1)
   ↓
2. Fill in Face Features (Tab 2)
   ↓
3. Generate Image (Tab 3)
   ↓
4. Export Report (Tab 4)
```

---

## Common Commands

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
pip install --upgrade -r backend/requirements.txt
pip install --upgrade -r frontend/requirements.txt
```

### Test Backend API
```bash
# Check health
curl http://localhost:8000/health

# View API docs
# Open in browser: http://localhost:8000/docs
```

### Stop the Application
```bash
# Press Ctrl+C in the terminal
# All services will shutdown gracefully
```

### Clear Generated Images
```bash
# Remove all generated images
rm -rf outputs/generated_images/*
```

---

## Customizing Ports

```bash
# Run on different ports
python app.py --backend-port 9000 --frontend-port 9501
```

---

## Troubleshooting

### Error: "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Error: "Port already in use"
```bash
# Use different ports
python app.py --backend-port 8001 --frontend-port 8502
```

### Error: "Cannot connect to API"
- Ensure backend is running (check terminal)
- Verify API URL is correct: http://localhost:8000/api
- Check firewall settings

### Model Loading Errors
- Models are in placeholder mode by default
- To enable real models:
  1. Uncomment code in `backend/app/models/llava_model.py`
  2. Uncomment code in `backend/app/models/realviz_model.py`
  3. Update paths in `backend/.env`

---

## Next Steps

1. **Test Workflow**: Fill in features and generate descriptions
2. **Enable Models**: When LLaVa and RealVisXL are available, uncomment model code
3. **Customize**: Modify features database in `frontend/forensic_app.py`
4. **Deploy**: Use Docker or cloud services for production

---

## Documentation

- **Full README**: See [README.md](README.md)
- **Backend Docs**: See [backend/README.md](backend/README.md)
- **API Docs**: http://localhost:8000/docs (when running)

---

## Support

- Check system info in app (Tab 4)
- Review API documentation at /docs endpoint
- Verify all dependencies are installed
- Check `.env` configuration
