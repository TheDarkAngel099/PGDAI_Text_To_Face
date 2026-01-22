"""Streamlit UI for Text-to-Face Generation"""
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Text-to-Face Generator",
    page_icon="🎨",
    layout="wide"
)

# API Configuration
API_URL = st.secrets.get("api_url", "http://localhost:8000/api") if "secrets" in dir(st) else "http://localhost:8000/api"

st.title("🎨 Text-to-Face Generation")
st.markdown("""
Generate realistic face images from detailed text descriptions.
Select facial attributes, generate a dense caption, and create your image!
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value=API_URL)
    
    st.header("📋 Generation Settings")
    height = st.slider("Image Height", 256, 1024, 512, step=64)
    width = st.slider("Image Width", 256, 1024, 512, step=64)
    steps = st.slider("Inference Steps", 1, 100, 50, step=5)
    guidance = st.slider("Guidance Scale", 0.0, 20.0, 7.5, step=0.5)


# Main layout
col1, col2 = st.columns(2)

with col1:
    st.header("Step 1️⃣: Select Facial Attributes")
    
    # Define categories and possible attributes
    categories = {
        "nose": ["color", "size", "shape"],
        "eyes": ["color", "shape", "size"],
        "mouth": ["shape", "color"],
        "face_shape": ["oval", "round", "square", "heart"],
        "skin_tone": ["fair", "medium", "dark"],
        "hair": ["color", "length", "texture"]
    }
    
    attributes_selected = []
    
    for category in categories.keys():
        st.subheader(f"👃 {category.title()}")
        col_attr, col_val = st.columns(2)
        
        with col_attr:
            attribute = st.selectbox(
                f"Choose attribute for {category}",
                options=categories[category],
                key=f"{category}_attr"
            )
        
        with col_val:
            value = st.text_input(
                f"Value for {category}",
                placeholder="e.g., brown, large, round",
                key=f"{category}_val"
            )
        
        if attribute and value:
            attributes_selected.append({
                "category": category,
                "attribute": attribute,
                "value": value
            })
    
    # Additional description
    description = st.text_area(
        "Additional Description (optional)",
        placeholder="Add any extra details about the face...",
        height=100
    )


with col2:
    st.header("Step 2️⃣: Generate Caption & Image")
    
    # Generate Caption Button
    if st.button("🚀 Generate Dense Caption", use_container_width=True):
        if not attributes_selected:
            st.warning("Please select at least one facial attribute")
        else:
            with st.spinner("Generating caption using LLaVa..."):
                try:
                    response = requests.post(
                        f"{api_url}/caption",
                        json={
                            "attributes": attributes_selected,
                            "description": description
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        caption_data = response.json()
                        st.session_state.dense_prompt = caption_data["dense_prompt"]
                        
                        st.success("Caption generated!")
                        st.info(f"**Dense Prompt:** {caption_data['dense_prompt']}")
                        st.caption(f"*Message: {caption_data.get('message', '')}*")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to API. Make sure the backend is running on " + api_url)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Generate Image Button
    if st.button("🎨 Generate Image", use_container_width=True, type="primary"):
        if "dense_prompt" not in st.session_state:
            st.warning("Please generate a caption first")
        else:
            with st.spinner("Generating image using RealVisXL..."):
                try:
                    response = requests.post(
                        f"{api_url}/generate-image",
                        json={
                            "prompt": st.session_state.dense_prompt,
                            "height": height,
                            "width": width,
                            "num_inference_steps": steps,
                            "guidance_scale": guidance
                        },
                        timeout=300  # Longer timeout for image generation
                    )
                    
                    if response.status_code == 200:
                        image_data = response.json()
                        
                        # Display image if available
                        if image_data.get("image_base64"):
                            img_bytes = base64.b64decode(image_data["image_base64"])
                            image = Image.open(BytesIO(img_bytes))
                            st.image(image, caption="Generated Face")
                        else:
                            st.info("No image data returned (check backend logs)")
                        
                        # Display metadata
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Generation Time", f"{image_data['generation_time']:.2f}s")
                        with col_b:
                            st.metric("Image Size", f"{width}x{height}")
                        
                        if image_data.get("image_path"):
                            st.caption(f"Saved to: {image_data['image_path']}")
                        
                        st.caption(f"*Message: {image_data.get('message', '')}*")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to API. Make sure the backend is running on " + api_url)
                except requests.exceptions.Timeout:
                    st.error("Request timeout. Image generation might be taking too long.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# Footer
st.divider()
st.markdown("""
### 📚 How it works:
1. **Select Attributes**: Choose facial features and their properties
2. **Generate Caption**: LLaVa converts your selections into a detailed prompt
3. **Generate Image**: RealVisXL creates the face image from the prompt

### 🔧 Tips:
- Be specific with your attribute values for better results
- The caption generation may take a few seconds
- Image generation typically takes 1-2 minutes
- Higher inference steps = better quality but slower generation

### ⚠️ Current Status:
The backend is running in **placeholder mode** with LLaVa and RealVisXL models disabled.
To use real models, enable them in the backend configuration.
""")
