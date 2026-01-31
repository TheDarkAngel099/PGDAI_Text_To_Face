# Text-to-Face Generative AI Pipeline - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Architecture](#technical-architecture)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Module Descriptions](#module-descriptions)
6. [Workflow & Pipeline](#workflow--pipeline)
7. [Key Components](#key-components)
8. [Training](#training)
9. [Image Generation](#image-generation)
10. [Web Application](#web-application)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Deployment](#deployment)
13. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Project Overview

### Executive Summary
This project implements a **scalable generative AI pipeline** that fine-tunes advanced diffusion models to generate realistic face images from text descriptions. The system integrates Vision Language Models (VLM) for automatic image captioning, enabling efficient dataset expansion and improved model performance.

### Key Objectives
- ✅ Fine-tune diffusion models (RealVisXL-4.0, Stable Diffusion-1.5) using LoRA
- ✅ Generate high-quality image descriptions using Vision Language Models (LLaVA-1.5-13b)
- ✅ Implement structured face detection and cropping preprocessing pipeline
- ✅ Engineer controlled prompts for consistent, forensic-style image generation
- ✅ Create a web-based interface for suspect description and face generation
- ✅ Evaluate and compare base vs fine-tuned models

### Infrastructure
**Developed & trained on:** CDAC's Param Rudra Supercomputer
- High-performance GPU computing for accelerated model training
- Distributed processing capabilities for large-scale data handling
- Optimized environment for diffusion model fine-tuning

---

## Technical Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (Web UI)                       │
│              (Streamlit Frontend Application)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Gateway                            │
│         (Request routing & API orchestration)               │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │   VLM   │    │ Diffusion│    │  Cache  │
   │ LLaVA   │    │ Models   │    │ Manager │
   │Pipeline │    │ (SDXL,   │    │         │
   │         │    │  SD1.5)  │    │         │
   └─────────┘    └──────────┘    └─────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │     Generated Output             │
        │  (Descriptions + Face Images)    │
        └────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | PyTorch 2.0+ | Deep learning operations |
| **Base Models** | Transformers, Diffusers | Pre-trained model loading |
| **Fine-tuning** | PEFT (LoRA) | Parameter-efficient training |
| **VLM** | LLaVA-1.5-13b | Auto-captioning & descriptions |
| **Diffusion Models** | RealVisXL-4.0, SD 1.5 | Image generation |
| **Web Backend** | FastAPI, Uvicorn | API server |
| **Web Frontend** | Streamlit | User interface |
| **Image Processing** | OpenCV, Pillow | Face detection & cropping |
| **Data Processing** | Pandas, HuggingFace Datasets | Dataset management |
| **Acceleration** | Hugging Face Accelerate | Distributed training |

---

## Project Structure

### Directory Layout

```
PGDAI_Text_To_Face/
│
├── README.md                          # Main project README
├── PROJECT_DOCUMENTATION.md           # This file
├── requirements.txt                   # Python dependencies
│
├── Data_Preprocessing/                # 📊 Data Preparation
│   ├── Indian_dataset/
│   │   ├── dataa_preprocessing.ipynb  # Data cleaning & preprocessing
│   │   ├── extracting_image.ipynb     # Image extraction & organization
│   │   ├── indian_dataset.csv         # Processed metadata
│   │   ├── indian.csv                 # Raw metadata
│   │   └── indian_images/             # Extracted facial images
│   │
│   └── illinois_preprocessing/
│       ├── data_preprocessing.ipynb   # Illinois dataset processing
│       ├── df_final.csv               # Final processed data
│       └── output1.csv                # Intermediate processing output
│
├── VLM/                               # 🤖 Vision Language Model Pipeline
│   ├── vlm_pipeline.py                # Main LLaVA captioning pipeline
│   ├── download_llava.py              # Model download & setup utility
│   ├── run_val_test.sh                # Validation & testing script
│   └── [outputs]                      # Generated captions & descriptions
│
├── Training/                          # 🎓 Model Fine-tuning
│   ├── RealVizXL/
│   │   ├── train_text_to_image_lora_sdxl.py  # RealVisXL LoRA training script
│   │   └── train_job.sh                       # Bash script for GPU job submission
│   │
│   └── sd1.5/
│       ├── train_text_to_image_lora_sd15.py  # SD 1.5 LoRA training script
│       ├── train_job_sd15_320.sh             # Job script (standard SD 1.5 LoRA training)
│       └── train_job_sd15_320_unetonly.sh    # Job script (UNet only variant)
│
├── Generation/                        # 🎨 Image Generation Scripts
│   ├── gen_base_model.py              # Generate images from base model
│   ├── generate_from_detailed_captions.py  # Use detailed captions for generation
│   ├── generate_images_from_captions.py    # Standard caption-based generation
│   ├── generate_images_from_captions_sd15.py # SD 1.5 specific generation
│   ├── generate_single_image.py       # Generate single image from prompt
│   └── test_indian_model.py           # Test fine-tuned Indian model
│
├── Evaluation/                        # 📈 Model Evaluation
│   ├── metrics.py                     # Metric calculation utilities
│   ├── compare_base_vs_finetuned.py  # Comparative analysis script
│   └── eval_metrics.sh                # Evaluation execution script
│
├── text-to-face-app/                  # 🌐 Web Application
│   ├── config.py                      # Configuration management
│   ├── README.md                      # App-specific documentation
│   ├── requirements.txt               # App dependencies
│   ├── values.txt                     # Configuration values
│   │
│   ├── backend/                       # FastAPI Backend
│   │   ├── __init__.py
│   │   ├── main.py                    # API endpoints & routing
│   │   ├── models.py                  # Pydantic data models
│   │   ├── dummy_image.py             # Mock image generation
│   │   └── dummy_text.py              # Mock text generation
│   │
│   ├── frontend/                      # Streamlit Frontend
│   │   ├── app.py                     # Main UI application
│   │   ├── api_client.py              # Backend API client
│   │   ├── cache_manager.py           # Cache handling
│   │   └── styles.py                  # UI styling utilities
│   │
│   ├── data/                          # Data Schemas
│   │   ├── __pycache__/
│   │   └── schema.py                  # Suspect schema & structures
│   │
│   └── assets/                        # Application Assets
│       ├── cache.json                 # Cache storage
│       └── images/                    # Generated images storage
│
├── Hosting/                           # 🚀 Deployment Configurations
│   └── huggingface spaces/
│       ├── app.py                     # HuggingFace Spaces app
│       ├── pytorch_lora_weights.safetensors
│       └── requirements.txt           # Deployment dependencies
│
├── Util/                              # 🛠️ Utility Scripts
│   ├── create_metadata.py             # Metadata generation utility
│   ├── Detect_Faces_crop.ipynb        # Face detection & cropping notebook
│   ├── download_model_static.py       # Static model download
│   ├── download_sd15_model.py         # SD 1.5 model download
│   ├── download_sd15_training_script.py # Training script downloader
│   ├── sdxl_training_script_download.py # SDXL training script downloader
│   │
│   └── Demo_Param/                    # Demo & Parameter Testing
│       ├── generate_demo.py           # Demo generation script
│       ├── run_demo_gpu.sh            # GPU demo execution
│       └── demo_result/               # Demo outputs
│
├── checkpoints/                       # 📦 Model Checkpoints & Weights
│   └── outputs_indian_finetuned_ckpt2700/
│       ├── pytorch_lora_weights.safetensors  # Merged LoRA weights
│       ├── checkpoint-2700/           # Final checkpoint
│       │   └── pytorch_lora_weights.safetensors
│       ├── checkpoint-96/             # Early checkpoint
│       │   └── pytorch_lora_weights.safetensors
│       └── logs/                      # Training logs & TensorBoard events
│           └── text2image-fine-tune/
│
├── Docs/                              # 📚 Documentation & Reports
│   ├── Graphs/                        # Visualization outputs
│   └── Report/                        # Project reports
│
└── Model Test/                        # 🧪 Model Testing Notebooks
    ├── RealVizXL (2).ipynb
    ├── stable_diffusion_xl.ipynb
    └── testing_models.ipynb
```

---

## Installation & Setup

### Prerequisites
- **Python:** 3.8 or higher (3.10+ recommended)
- **CUDA:** 11.8+ (for GPU acceleration)
- **GPU Memory:** Minimum 12GB VRAM (24GB+ recommended)
- **Disk Space:** 100GB+ for models and datasets
- **OS:** Linux (Ubuntu 20.04+) or Windows with WSL2

### Step 1: Clone Repository
```bash
cd d:\CDAC\PGDAI_Text_To_Face
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Models
```bash
# Download LLaVA model for captioning
python Util/download_llava.py

# Download base diffusion models
python Util/download_model_static.py
python Util/download_sd15_model.py
```

### Step 5: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Module Descriptions

### Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         3-STAGE TRAINING PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

  STAGE 1                    STAGE 2                    STAGE 3
  DATA PROCESSING            VLM CAPTION GENERATION     REALVIZ TRAINING
  
  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │   Raw Images     │      │  LLaVA-1.5-13b   │      │ RealVisXL-4.0    │
  │  (Indian/        │──→   │  Captioning      │──→   │ + LoRA Fine-tune │
  │  Illinois)       │      │  (Auto-describe) │      │                  │
  └──────────────────┘      └──────────────────┘      └──────────────────┘
         │                          │                         │
         ├─ Face Detection          ├─ Image-Caption          ├─ Initialize LoRA
         ├─ Crop & Normalize        │  Dataset CSV            ├─ Train on Captions
         ├─ Data Validation         ├─ Generate Metadata      ├─ Save Checkpoints
         └─ Metadata CSV            └─ Review Captions        └─ LoRA Weights
              │                           │                         │
              ▼                           ▼                         ▼
         Clean Dataset            Image-Caption Pairs      Fine-tuned Model
         + Metadata                + Validation Split        + LoRA Weights
                                                           (outputs_indian_
                                                            finetuned_ckpt2700/)

  Location:                 Location:                   Location:
  Data_Preprocessing/       VLM/                        Training/RealVizXL/
  {Indian,Illinois}/        vlm_pipeline.py             train_text_to_image_
                                                        lora_sdxl.py
```

---

## Module Descriptions

### 1. Data Preprocessing (`Data_Preprocessing/`)

#### Purpose
Prepare raw facial datasets for training by cleaning, normalizing, and organizing image data.

#### Indian Dataset Processing
- **Input:** Raw Indian facial dataset images
- **Process:** 
  - Image extraction and organization
  - Face detection and bounding box extraction
  - Data validation and quality checks
  - Metadata CSV generation
- **Output:** Organized images + `indian_dataset.csv` metadata

#### Illinois Dataset Processing
- **Input:** Illinois criminal mugshot database
- **Process:** 
  - Similar preprocessing pipeline
  - Dataset merging if multiple sources
  - Standardization of image formats
- **Output:** Processed images + `df_final.csv`

#### Key Scripts
- `dataa_preprocessing.ipynb` - Data cleaning & exploration
- `extracting_image.ipynb` - Image extraction pipeline

---

### 2. Vision Language Model Pipeline (`VLM/`)

#### Purpose
Automatically generate high-quality image descriptions using LLaVA-1.5-13b, eliminating need for manual annotations.

#### Workflow
1. **Load Images:** Reads preprocessed facial images
2. **Generate Descriptions:** Uses LLaVA to create detailed, forensic-style captions
3. **Post-processing:** Filters and validates descriptions
4. **Output:** CSV file with image paths and descriptions

#### Key Scripts
- `vlm_pipeline.py` - Main captioning pipeline
- `download_llava.py` - Model download utility

#### Key Parameters
```python
MODEL_ID = "llava-hf/llava-1.5-13b-hf"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
BATCH_SIZE = 1  # GPU memory dependent
```

#### Example Output
**Input Image:** Mugshot of male, 30s, Indian ethnicity
**Generated Caption:** 
```
"A mugshot of an Indian male in his thirties. He has medium-length black hair, 
a fair complexion, and distinctive facial features. The photo is taken with 
standard forensic photography lighting against a plain background."
```

---

### 3. Training (`Training/`)

#### Purpose
Fine-tune base diffusion models using LoRA for efficient parameter-efficient training.

#### Training Strategy: LoRA (Low-Rank Adaptation)
- **Advantage:** Reduces trainable parameters by 99%+ vs full fine-tuning
- **Efficiency:** Trains faster, uses less GPU memory
- **Flexibility:** Can merge or switch LoRA weights easily

#### RealVisXL-4.0 Training

**Script:** `RealVizXL/train_text_to_image_lora_sdxl.py`

**Key Parameters:**
```bash
RESOLUTION=320                    # 320x320 (native dataset resolution, no upscaling)
BATCH_PER_GPU=4                   # Batch size per GPU
GRAD_ACCUM=2                      # Gradient accumulation steps (effective batch = 4 × 2 = 8)
LEARNING_RATE=1e-4                # Learning rate
LR_SCHEDULER="linear"             # Linear decay scheduler for refined convergence
LR_WARMUP_STEPS=100               # Warmup steps for stability
MAX_TRAIN_STEPS=2715              # Total training steps (3 epochs × 905 steps/epoch)
VALIDATION_STEPS=225              # Validation every ~1/4 epoch
CHECKPOINT_STEPS=450              # Save checkpoint every half-epoch
CHECKPOINTS_TOTAL_LIMIT=10        # Keep all checkpoints
RANK=4                            # LoRA rank (dimension of update matrices)
SEED=42                           # Random seed for reproducibility
```

**Training Configuration:**
- **Dataset:** ~7,238 images (Indian facial dataset)
- **Epochs:** 3 complete passes
- **Effective Batch Size:** 8 (4 per GPU × 2 gradient accumulation)
- **Total Steps:** 2,715
- **Training Time:** ~4-5 hours (single GPU)
- **Output Checkpoints:** `checkpoint-96/`, `checkpoint-2700/` (final)
- **Final Weights:** `pytorch_lora_weights.safetensors`

**Job Submission:**
```bash
sbatch RealVizXL/train_job.sh
```

#### Stable Diffusion 1.5 Training

**Script:** `sd1.5/train_text_to_image_lora_sd15.py`

**Key Parameters:**
```bash
RESOLUTION=320                    # 320x320 resolution (matching RealVisXL dataset)
BATCH_PER_GPU=16                  # Batch size per GPU (larger than RealVisXL)
GRAD_ACCUM=1                      # Gradient accumulation steps (no accumulation)
LEARNING_RATE=5e-5                # Lower learning rate than RealVisXL
LR_SCHEDULER="constant"           # Constant learning rate (no decay)
LR_WARMUP_STEPS=0                 # No warmup period
MAX_TRAIN_STEPS=5000              # Total training steps
CHECKPOINT_STEPS=500              # Save checkpoint every 500 steps
RANK=4                            # LoRA rank (dimension of update matrices)
SEED=42                           # Random seed for reproducibility
```

**Training Configuration:**
- **Model:** runwayml/stable-diffusion-v1-5 (cached locally)
- **Dataset:** ~7,238 images (Indian facial dataset)
- **Batch Size:** 16 per GPU (higher throughput than RealVisXL)
- **Learning Rate:** 5e-5 (more conservative than RealVisXL's 1e-4)
- **Scheduler:** Constant (no decay)
- **Total Steps:** 5,000
- **Checkpoints:** Saved every 500 steps (~10 total)
- **Expected Time:** ~2-3 hours (faster inference than SDXL)

**Comparison with RealVisXL:**
| Parameter | RealVisXL | SD 1.5 |
|-----------|-----------|--------|
| Batch Size | 4 | 16 |
| Learning Rate | 1e-4 | 5e-5 |
| Grad Accum | 2 | 1 |
| LR Scheduler | Linear | Constant |
| Total Steps | 2,715 | 5,000 |

**Job Submission:**
```bash
sbatch sd1.5/train_job_sd15_320.sh
# Optional UNet-only variant
# sbatch sd1.5/train_job_sd15_320_unetonly.sh
```

#### Training Output
```
checkpoints/outputs_indian_finetuned_ckpt2700/
├── checkpoint-96/
├── checkpoint-2700/  (Final checkpoint)
└── pytorch_lora_weights.safetensors
```

#### Monitoring
- **TensorBoard Logs:** `checkpoints/outputs_indian_finetuned_ckpt2700/logs/`
- **Command:** `tensorboard --logdir=checkpoints/outputs_indian_finetuned_ckpt2700/logs/`

---

### 4. Image Generation (`Generation/`)

#### Purpose
Generate realistic face images from text descriptions using base and fine-tuned models.

#### Script Overview

| Script | Purpose | Model | Input |
|--------|---------|-------|-------|
| `gen_base_model.py` | Generate with base model | RealVisXL | Detailed captions |
| `generate_from_detailed_captions.py` | Enhanced generation | Fine-tuned | Detailed prompts |
| `generate_images_from_captions.py` | Standard generation | Fine-tuned (SDXL) | CSV with captions |
| `generate_images_from_captions_sd15.py` | SD 1.5 generation | Fine-tuned (SD1.5) | CSV with captions |
| `generate_single_image.py` | Single image demo | Configurable | Text prompt |
| `test_indian_model.py` | Test fine-tuned model | Fine-tuned Indian | Test prompts |

#### Basic Generation Example
```python
from diffusers import StableDiffusionXLPipeline
from compel import Compel

# Load fine-tuned model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "path/to/model",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("path/to/lora/weights")

# Generate with prompt weighting
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt = "Indian male, 30s, black hair, fair skin + (forensic mugshot)1.5"
prompt_embeds = compel.build_prompt_embeds(prompt)

image = pipe(
    prompt_embeds=prompt_embeds,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

#### Common Parameters
- **num_inference_steps:** 20-50 (higher = better quality but slower)
- **guidance_scale:** 7.5-15.0 (higher = more prompt adherence)
- **height/width:** 512 (SD1.5) or 768+ (SDXL)
- **seed:** For reproducibility

---

### 5. Web Application (`text-to-face-app/`)

#### Architecture
```
Streamlit Frontend (Port 8501)
         ↓
API Client (api_client.py)
         ↓
FastAPI Backend (Port 8000)
         ↓
Cache Manager & Model APIs
```

#### Backend (`backend/`)

**Main Endpoints:**
```python
POST /generate-caption
  Input: {"prompt_text": "Indian male, 30s"}
  Output: {"status": "success", "caption": "detailed description..."}

POST /generate-image
  Input: {"caption": "detailed description"}
  Output: {"status": "success", "image_url": "..."}
```

**Models & Schemas:**
```python
class CaptionRequest(BaseModel):
    prompt_text: str

class ImageRequest(BaseModel):
    caption: str
```

**Startup:**
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (`frontend/`)

**Features:**
- **Attribute Selection:** Gender, age, ethnicity, hair color, etc.
- **Description Generation:** AI-powered text generation
- **Image Generation:** Create faces from descriptions
- **Caching:** Store and reuse previous generations
- **History:** View past generations

**Key Components:**
- `app.py` - Main Streamlit app
- `api_client.py` - REST client for backend
- `cache_manager.py` - Local caching system
- `styles.py` - UI customization

**Startup:**
```bash
streamlit run frontend/app.py
```

**Access:** http://localhost:8501

#### Configuration (`config.py`)
```python
# API Endpoints
TEXT_MODEL_URL = "http://localhost:8001/text-generation"
IMAGE_MODEL_URL = "http://localhost:8002/image-generation"
GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 8000

# Cache Settings
CACHE_DIR = "assets/cache.json"
ENABLE_CACHE = True
CACHE_TTL = 3600  # seconds
```

---

### 6. Evaluation (`Evaluation/`)

#### Purpose
Assess model quality and compare base vs fine-tuned models.

#### Metrics Calculated

| Metric | Purpose | Tool |
|--------|---------|------|
| **LPIPS** | Perceptual image similarity | LPIPS library |
| **SSIM** | Structural similarity | scikit-image |
| **CLIP Cosine** | Image semantic alignment | OpenAI CLIP |
| **Composite Score** | Combined metric | (1 - LPIPS + SSIM + CLIP) / 3 |

#### Evaluation Script
```bash
bash Evaluation/eval_metrics.sh
```

#### Comparison Report
```python
python Evaluation/compare_base_vs_finetuned.py \
  --base_model_dir path/to/base \
  --finetuned_model_dir path/to/finetuned \
  --test_prompts test_prompts.csv \
  --output_report report.json
```

---

### 7. Utilities (`Util/`)

#### Key Utilities

**Face Detection & Cropping**
```bash
jupyter notebook Util/Detect_Faces_crop.ipynb
```
- Detects faces using dlib/MTCNN
- Crops and normalizes to standard size
- Saves processed images

**Metadata Creation**
```bash
python Util/create_metadata.py --input_dir images/ --output metadata.csv
```

**Model Downloaders**
- `download_model_static.py` - Base model download
- `download_sd15_model.py` - SD 1.5 model
- `download_llava.py` - LLaVA model

**Demo Generator**
```bash
bash Util/Demo_Param/run_demo_gpu.sh
python Util/Demo_Param/generate_demo.py
```

---

## Workflow & Pipeline

### Complete End-to-End Workflow

```
1. RAW DATA ACQUISITION
   └─→ Collect facial images (Indian/Illinois datasets)

2. DATA PREPROCESSING (Data_Preprocessing/)
   └─→ Extract faces
   └─→ Normalize images
   └─→ Create metadata CSV
   └─→ Output: Clean dataset + metadata

3. AUTO-CAPTIONING (VLM/)
   └─→ Load LLaVA model
   └─→ Process each image
   └─→ Generate descriptions
   └─→ Output: Image-caption pairs CSV

4. DATA PREPARATION FOR TRAINING
   └─→ Create image-caption dataset
   └─→ Split train/validation
   └─→ Verify alignments

5. MODEL FINE-TUNING (Training/)
   └─→ Load base diffusion model
   └─→ Initialize LoRA
   └─→ Train on captioned dataset
   └─→ Save checkpoints
   └─→ Output: Fine-tuned LoRA weights

6. IMAGE GENERATION (Generation/)
   └─→ Load fine-tuned model + LoRA
   └─→ Load test prompts/captions
   └─→ Generate images
   └─→ Save outputs

7. EVALUATION (Evaluation/)
   └─→ Calculate LPIPS, SSIM, CLIP, Composite scores
   └─→ Compare base vs fine-tuned
   └─→ Generate reports

8. DEPLOYMENT (Hosting/ + text-to-face-app/)
   └─→ Package model
   └─→ Deploy backend API
   └─→ Deploy frontend UI
   └─→ Enable caching & optimization
```

---

## Key Components

### Diffusion Models

#### RealVisXL-4.0
- **Base Model:** Latest realistic generation
- **Resolution:** 768x768 (SDXL architecture) 320 x 320 (for this project)
- **Strengths:** Photorealistic outputs, high-quality details
- **Use Case:** Primary model for production
- **Fine-tuned Checkpoint:** `outputs_indian_finetuned_ckpt2700`

#### Stable Diffusion 1.5
- **Base Model:** Widely used, lighter weight
- **Resolution:** 512x512, 320 x 320 (for this project)
- **Strengths:** Faster inference, lower memory
- **Use Case:** Testing, resource-constrained environments
- **Fine-tuning:** Available in `Training/sd1.5/`

### Vision Language Model

#### LLaVA-1.5-13b
- **Architecture:** Vision transformer + LLM
- **Capabilities:** Image understanding & description generation
- **Strengths:** Detailed, contextual captions
- **Language:** English (can be adapted)
- **Inference Speed:** ~2-5 minutes per image on GPU

### LoRA (Low-Rank Adaptation)
- **Concept:** Adds trainable low-rank matrices to model weights
- **Parameters Saved:** ~1% of original model size
- **Training Speed:** 10-50x faster than full fine-tuning
- **Flexibility:** Easily mergeable with base model
- **Typical Configuration:** rank=4

---

## Training

### Pre-training Checklist
- [ ] Data preprocessed and validated
- [ ] Captions generated and reviewed
- [ ] Compute resources allocated (GPU available)
- [ ] Model checkpoints downloaded
- [ ] LoRA configuration parameters set
- [ ] Hyperparameters tuned for dataset

### Training Process

#### 1. RealVisXL Training
```bash
cd Training/RealVizXL/

# Configure job parameters
vim train_job.sh  # Edit if needed

# Submit job
sbatch train_job.sh
```

**Expected Timeline:**
- Dataset: 10k images
- Epochs: 3
- Time: 4-5 hours (on GPU)
- GPU Memory: ~20GB

#### 2. SD 1.5 Training
```bash
cd Training/sd1.5/

# Configure parameters
vim train_text_to_image_lora_sd15.py

# Run training
python train_text_to_image_lora_sd15.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="path/to/images" \
  --output_dir="./models/finetuned" \
  --instance_prompt="a photo of a person" \
  --validation_prompt="a mugshot of a person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-4 \
  --max_train_steps=2000
```

### Monitoring Training
```bash
# Watch logs in real-time
tail -f checkpoints/outputs_indian_finetuned_ckpt2700/logs/*.log

# View TensorBoard
tensorboard --logdir=checkpoints/outputs_indian_finetuned_ckpt2700/logs/
```

### Training Completion
- Saved weights: `checkpoints/outputs_indian_finetuned_ckpt2700/pytorch_lora_weights.safetensors`
- Best checkpoint: `checkpoint-2700/`
- Total training time: Logged in TensorBoard

---

## Image Generation

### Generate from Base Model
```bash
python Generation/gen_base_model.py
```
**Configuration:**
```python
MODEL_ID = "path/to/base/model"
CAPTIONS_DIR = "path/to/captions"
OUTPUT_DIR = "path/to/output"
RESOLUTION = 320
NUM_INFERENCE_STEPS = 50
```

### Generate from Fine-tuned Model
```bash
python Generation/generate_from_detailed_captions.py \
  --model_id="path/to/model" \
  --lora_weights="checkpoints/outputs_indian_finetuned_ckpt2700/pytorch_lora_weights.safetensors" \
  --captions_file="path/to/captions.csv" \
  --output_dir="path/to/output"
```

### Single Image Generation
```bash
python Generation/generate_single_image.py \
  --prompt="Indian male, 30s, black hair, fair skin" \
  --model_path="path/to/model" \
  --use_lora=True
```

### Batch Generation
```bash
python Generation/generate_images_from_captions.py \
  --csv_file="captions.csv" \
  --num_images=100 \
  --guidance_scale=7.5 \
  --output_dir="./outputs"
```

---

## Web Application

### Starting the Application

#### 1. Start Backend
```bash
cd text-to-face-app/
python -m uvicorn backend.main:app --reload --port 8000
```

#### 2. Start Frontend (New Terminal)
```bash
cd text-to-face-app/
streamlit run frontend/app.py
```

#### 3. Access Application
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Application Flow
1. User selects attributes (gender, age, ethnicity, etc.)
2. Frontend sends to backend `/generate-caption` endpoint
3. Backend forwards to text model API
4. Returns generated description
5. User reviews and can refine description
6. Frontend sends to backend `/generate-image` endpoint
7. Backend forwards to image generation model
8. Image is displayed and cached locally
9. User can save or regenerate

### Caching System
```python
# Cache structure
{
  "prompt_hash": {
    "timestamp": 1234567890,
    "description": "...",
    "image_path": "assets/images/..."
  }
}
```
- **TTL:** 1 hour (configurable)
- **Storage:** `assets/cache.json`
- **Purge:** Automatic on TTL expiration

---

## Evaluation Metrics

### Running Evaluation
```bash
bash Evaluation/eval_metrics.sh
```

### Key Metrics Explained

**LPIPS (Learned Perceptual Image Patch Similarity)**
- Range: 0-1 (lower is better)
- Measures: Perceptual difference between images using deep network features
- Benchmark: LPIPS < 0.2 is high quality
- Implementation: [metrics.py](Evaluation/metrics.py)

**SSIM (Structural Similarity Index)**
- Range: -1 to 1 (higher is better)
- Measures: Structural similarity between two images (luminance, contrast, structure)
- Benchmark: SSIM > 0.8 indicates high quality
- Implementation: [metrics.py](Evaluation/metrics.py)

**CLIP Cosine Similarity**
- Range: 0-1 (higher is better)
- Measures: Semantic alignment between images using CLIP embeddings
- Benchmark: CLIP > 0.25 is good
- Implementation: [metrics.py](Evaluation/metrics.py)

**Composite Score**
- Range: typically 0-1 (higher is better)
- Formula: `((1 - LPIPS) + SSIM + CLIP) / 3`
- Measures: Balanced combination of all three metrics
- Benchmark: Composite > 0.5 indicates good overall quality
- Implementation: [metrics.py](Evaluation/metrics.py)

### Comparison Report
```json
{
  "base_model": {
    "lpips": 0.28,
    "ssim": 0.72,
    "clip_cosine": 0.22,
    "composite_score": 0.55
  },
  "finetuned_model": {
    "lpips": 0.18,
    "ssim": 0.85,
    "clip_cosine": 0.31,
    "composite_score": 0.66
  },
  "improvements": {
    "lpips_improvement": "35% better",
    "ssim_improvement": "18% better",
    "clip_improvement": "41% better",
    "composite_improvement": "20% better"
  }
}
```

---

## Deployment

### Option 1: HuggingFace Spaces

**Files Required:**
- `Hosting/huggingface spaces/app.py`
- `Hosting/huggingface spaces/requirements.txt`
- `pytorch_lora_weights.safetensors`

**Deployment Steps:**
1. Create HuggingFace account
2. Create new Space (Gradio or Streamlit)
3. Upload files
4. Add secrets for API keys
5. Space auto-builds and deploys

**Access:** https://huggingface.co/spaces/your-username/your-space

### Option 2: Local Docker Deployment

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
docker build -t text-to-face:latest .
docker run -p 8000:8000 -p 8501:8501 --gpus all text-to-face:latest
```

### Option 3: Cloud VM Deployment (AWS/GCP)

**Setup Steps:**
1. Launch GPU-enabled instance (p3.2xlarge or similar)
2. Install CUDA and Docker
3. Clone repository
4. Set up environment variables
5. Run application in background (tmux/screen)
6. Configure reverse proxy (nginx)
7. Enable SSL/TLS (Let's Encrypt)

---

## Troubleshooting & FAQ

### Common Issues

#### Q1: CUDA Out of Memory Error
**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size
- Lower resolution (512 instead of 768)
- Use int8 quantization
- Enable gradient checkpointing
- Use SD 1.5 instead of SDXL

```python
# Reduce memory usage
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
```

#### Q2: Model Download Fails
**Symptoms:** `FileNotFoundError: Model not found`

**Solutions:**
```bash
# Set HuggingFace cache directory
export HF_HOME="/path/to/cache"

# Clear cache and re-download
rm -rf ~/.cache/huggingface
python Util/download_model_static.py
```

#### Q3: Slow Image Generation
**Symptoms:** Generation takes > 5 minutes

**Solutions:**
- Reduce `num_inference_steps` (default 50 → 20-30)
- Lower resolution if acceptable
- Use faster GPU or batch generation
- Enable TorchCompile (PyTorch 2.0+)

```python
# Enable TorchCompile for speedup
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
```

#### Q4: Poor Quality Generated Images
**Symptoms:** Blurry, distorted, or unrecognizable faces

**Solutions:**
- Verify captions quality (check VLM output)
- Increase `guidance_scale` (7.5 → 12-15)
- Ensure proper fine-tuning completion
- Check training loss decreased during training
- Use negative prompts to exclude unwanted features

```python
prompt = "Indian male, 30s, clear face, high quality"
negative_prompt = "blurry, distorted, low quality, ugly"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=12.0
).images[0]
```

#### Q5: Fine-tuning Diverges (Loss Increases)
**Symptoms:** Training loss increases instead of decreasing

**Solutions:**
- Reduce learning rate (1e-4 → 5e-5)
- Add more image variations to dataset
- Verify captions are sensible (manual spot-check)
- Use smaller LoRA rank (64 → 32)
- Add regularization or gradient clipping

### Performance Optimization

#### GPU Acceleration
```python
# Enable optimizations
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(...)
pipe = pipe.to("cuda")

# Attention optimization
pipe.enable_attention_slicing()  # Lower memory
# OR
pipe.enable_xformers_memory_efficient_attention()  # Faster

# Enable VAE tiling for large batches
pipe.vae.enable_tiling()
```

#### Batch Processing
```python
# Generate multiple images efficiently
from PIL import Image
import torch

prompts = [
    "Indian male, 30s",
    "Indian female, 25s",
    "Indian male, 40s"
]

# Batch size 3 is more efficient than 1x3
images = pipe(
    prompt=prompts,
    num_inference_steps=30,
    guidance_scale=7.5
).images

for i, img in enumerate(images):
    img.save(f"output_{i}.png")
```

### FAQ

**Q: Can I use this for production?**
A: Yes, with proper deployment (Section: Deployment). Ensure compliance with data usage policies.

**Q: How do I improve generation quality?**
A: Fine-tune on higher-quality captions, use better base models (RealVisXL-4.0), increase inference steps.

**Q: Can I merge LoRA weights with base model?**
A: Yes, for single-model deployment:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, lora_path)
merged = model.merge_and_unload()
```

**Q: How to handle non-English prompts?**
A: Fine-tune VLM and generation models on multilingual data, or use translation pipeline.

**Q: What's the typical inference time?**
A: 
- RealVisXL: 3-5 seconds (50 steps, A100 GPU)
- SD 1.5: 1-2 seconds (30 steps, A100 GPU)

**Q: How much training data is needed?**
A: Minimum 100 images, optimal 1000-10000 images for good results.

---

## Additional Resources

### Related Papers
- **LoRA:** ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- **Diffusion Models:** ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)
- **CLIP:** ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.14030)
- **Vision Language Models:** ["LLaVA: Visual Instruction Tuning"](https://arxiv.org/abs/2304.08485)

### Documentation Links
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Community
- Hugging Face Community Forum: https://discuss.huggingface.co/
- Reddit r/MachineLearning, r/StableDiffusion
- GitHub Issues in respective repositories

---

## Contact & Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review related papers and documentation
3. Post in relevant GitHub issues
4. Consult project team leads

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-31 | Initial comprehensive documentation |

---

**Last Updated:** January 31, 2026

**Project:** PGDAI Text-to-Face Generative AI Pipeline  
**Institution:** CDAC (Centre for Development of Advanced Computing)  
**Supercomputer:** Param Rudra
