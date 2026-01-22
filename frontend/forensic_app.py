"""
Forensic Face Description System
Criminal suspect facial feature capture application
Uses hierarchical categorization: Face Part → Attributes → Values
"""
import streamlit as st
from typing import Dict, List, Tuple
import requests
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION & THEME
# ============================================================================

st.set_page_config(
    page_title="Forensic Face Description System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for forensics theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
    }
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #16213e;
        border-radius: 8px 8px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e94560 !important;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d63554;
    }
    .stSelectbox, .stTextInput, .stTextArea {
        background-color: #0f3460;
    }
    .section-header {
        background: linear-gradient(90deg, #e94560 0%, #d63554 100%);
        padding: 12px;
        border-radius: 6px;
        color: white;
        font-weight: bold;
        margin-bottom: 12px;
    }
    .info-box {
        background-color: #0f3460;
        border-left: 4px solid #e94560;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FACE FEATURES DATABASE
# ============================================================================

FACE_FEATURES_DB: Dict[str, Dict[str, List[str]]] = {
    "Face Shape": {
        "shape": ["Oval", "Round", "Square", "Rectangular", "Heart", "Diamond", "Triangular", "Other"]
    },
    "Forehead": {
        "height": ["Low", "Medium", "High", "Very High", "Other"],
        "width": ["Narrow", "Medium", "Wide", "Very Wide", "Other"],
        "features": ["Wrinkles", "Scars", "Birthmarks", "Tattoo", "None", "Other"]
    },
    "Eyes": {
        "color": ["Black", "Dark Brown", "Brown", "Hazel", "Light Brown", "Green", "Blue", "Gray", "Heterochromia", "Other"],
        "shape": ["Almond", "Round", "Hooded", "Monolid", "Upturned", "Downturned", "Asymmetrical", "Other"],
        "size": ["Very Small", "Small", "Medium", "Large", "Very Large", "Other"],
        "distance": ["Close Set", "Normal", "Wide Set", "Very Wide", "Other"],
        "eyebrows": ["Thin", "Medium", "Thick", "Very Thick", "Unibrow", "Absent", "Tattooed", "Other"],
        "other_features": ["Scar", "Birthmark", "Tattoo", "Glasses/Contacts", "Lazy Eye", "None", "Other"]
    },
    "Nose": {
        "shape": ["Straight", "Crooked", "Hooked", "Bulbous", "Pointed", "Flat", "Aquiline", "Other"],
        "width": ["Narrow", "Medium", "Wide", "Very Wide", "Other"],
        "length": ["Short", "Medium", "Long", "Very Long", "Other"],
        "bridge": ["Straight", "Concave", "Convex", "Broken", "Other"],
        "tip": ["Pointed", "Rounded", "Bulbous", "Split", "Other"],
        "nostrils": ["Small", "Medium", "Large", "Flared", "Other"],
        "features": ["Scar", "Pimple", "Birthmark", "Tattoo", "Piercing", "None", "Other"]
    },
    "Mouth": {
        "shape": ["Wide", "Normal", "Small", "Cupid Bow", "Thin Lips", "Full Lips", "Asymmetrical", "Other"],
        "upper_lip": ["Thin", "Medium", "Full", "Very Full", "Protruding", "Other"],
        "lower_lip": ["Thin", "Medium", "Full", "Very Full", "Protruding", "Other"],
        "color": ["Pale", "Pink", "Dark Pink", "Red", "Brown", "Other"],
        "distinguishing": ["Scar", "Cleft Lip", "Gap Teeth", "Tattoo", "Piercing", "Gold Teeth", "None", "Other"]
    },
    "Cheeks": {
        "shape": ["Hollow", "Normal", "Full", "Very Full", "Prominent", "Other"],
        "color": ["Pale", "Normal", "Flushed", "Red", "Pigmented", "Other"],
        "features": ["Acne", "Acne Scars", "Birthmark", "Tattoo", "Piercing", "Dimples", "Freckles", "None", "Other"]
    },
    "Chin": {
        "shape": ["Pointed", "Rounded", "Square", "Prominent", "Receding", "Cleft", "Other"],
        "size": ["Small", "Medium", "Large", "Very Large", "Other"],
        "features": ["Dimple", "Scar", "Birthmark", "Tattoo", "Beard Stubble", "Beard", "None", "Other"],
        "beard": ["None", "Goatee", "Full Beard", "Stubble", "Van Dyke", "Soul Patch", "Other"]
    },
    "Scars & Marks": {
        "location": ["Forehead", "Cheek", "Chin", "Nose", "Lips", "Eye Area", "Neck", "Multiple", "Other"],
        "type": ["Scar", "Birthmark", "Tattoo", "Mole", "Wart", "Acne Scar", "Burn Mark", "Other"],
        "size": ["Small (< 1 inch)", "Medium (1-2 inches)", "Large (2-3 inches)", "Very Large (> 3 inches)", "Other"],
        "appearance": ["Raised", "Indented", "Flat", "Discolored", "Other"]
    },
    "Hair": {
        "color": ["Black", "Dark Brown", "Brown", "Light Brown", "Blonde", "Red", "Gray", "White", "Dyed", "Other"],
        "texture": ["Straight", "Wavy", "Curly", "Coily", "Kinky", "Braided", "Other"],
        "length": ["Bald", "Very Short", "Short", "Medium", "Long", "Very Long", "Other"],
        "style": ["Shaved", "Crew Cut", "Fade", "Afro", "Dreadlocks", "Braids", "Messy", "Combed Back", "Side Part", "Other"],
        "coverage": ["Full Head", "Receding", "Widow's Peak", "Bald Spot", "Thinning", "Other"]
    }
}

DEMOGRAPHICS: Dict[str, List[str]] = {
    "Gender": ["Male", "Female", "Non-Binary", "Prefer Not to Say", "Other"],
    "Race/Ethnicity": [
        "Caucasian/White",
        "African American/Black",
        "Asian",
        "Hispanic/Latino",
        "Middle Eastern/North African",
        "Native American/Indigenous",
        "Pacific Islander",
        "South Asian",
        "Mixed Race",
        "Other"
    ],
    "Skin Tone": [
        "Very Fair",
        "Fair",
        "Light",
        "Medium Light",
        "Medium",
        "Medium Dark",
        "Dark",
        "Very Dark",
        "Other"
    ],
    "Age Range": [
        "Child (0-12)",
        "Teenager (13-19)",
        "Young Adult (20-30)",
        "Adult (31-45)",
        "Middle Aged (46-60)",
        "Senior (61+)",
        "Unknown"
    ],
    "Distinctive Features": [
        "Tattoos",
        "Piercings",
        "Glasses",
        "Facial Hair",
        "Scars",
        "Birthmarks",
        "Freckles",
        "Dimples",
        "Large Ears",
        "None Notable",
        "Other"
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "feature_selections" not in st.session_state:
        st.session_state.feature_selections = {}
    
    if "demographics" not in st.session_state:
        st.session_state.demographics = {}
    
    if "generated_prompt" not in st.session_state:
        st.session_state.generated_prompt = ""
    
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None


def generate_description_prompt() -> str:
    """
    Generate a natural language description from feature selections
    """
    parts = []
    
    # Demographics
    if st.session_state.demographics:
        demo_parts = []
        if st.session_state.demographics.get("Gender"):
            demo_parts.append(st.session_state.demographics["Gender"].lower())
        if st.session_state.demographics.get("Age Range"):
            age = st.session_state.demographics["Age Range"]
            demo_parts.append(age.lower())
        if st.session_state.demographics.get("Race/Ethnicity"):
            demo_parts.append(st.session_state.demographics["Race/Ethnicity"].lower())
        if st.session_state.demographics.get("Skin Tone"):
            demo_parts.append(f"with {st.session_state.demographics['Skin Tone'].lower()} skin")
        
        if demo_parts:
            parts.append(", ".join(demo_parts))
    
    # Face features
    if st.session_state.feature_selections:
        for face_part, attributes in st.session_state.feature_selections.items():
            if attributes:
                feature_desc = f"{face_part.lower()} "
                attr_values = []
                for attr, value in attributes.items():
                    if value:
                        attr_values.append(f"{attr.lower()}: {value}")
                feature_desc += ", ".join(attr_values)
                parts.append(feature_desc)
    
    if not parts:
        return "Criminal suspect with unspecified features"
    
    prompt = "Criminal suspect: " + "; ".join(parts)
    return prompt


def export_description(description: str, format_type: str = "txt") -> str:
    """Export description in requested format"""
    if format_type == "txt":
        return description
    elif format_type == "json":
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "demographics": st.session_state.demographics,
            "features": st.session_state.feature_selections
        }
        return json.dumps(export_data, indent=2)
    return description


# ============================================================================
# MAIN APPLICATION
# ============================================================================

initialize_session_state()

# Header
st.markdown("# 🔍 Forensic Face Description System")
st.markdown("### Criminal Suspect Facial Feature Documentation")
st.markdown("---")

# Create tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["📝 Suspect Profile", "👤 Face Features", "🖼️ Image Generation", "📊 Summary & Export"])

# ============================================================================
# TAB 1: DEMOGRAPHICS
# ============================================================================

with tab1:
    st.markdown('<div class="section-header">📋 Suspect Demographics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.demographics["Gender"] = st.selectbox(
            "Gender",
            DEMOGRAPHICS["Gender"],
            index=0 if "Gender" not in st.session_state.demographics else 
                   DEMOGRAPHICS["Gender"].index(st.session_state.demographics.get("Gender", DEMOGRAPHICS["Gender"][0]))
        )
        
        st.session_state.demographics["Race/Ethnicity"] = st.selectbox(
            "Race/Ethnicity",
            DEMOGRAPHICS["Race/Ethnicity"],
            index=0 if "Race/Ethnicity" not in st.session_state.demographics else
                   DEMOGRAPHICS["Race/Ethnicity"].index(st.session_state.demographics.get("Race/Ethnicity", DEMOGRAPHICS["Race/Ethnicity"][0]))
        )
        
        st.session_state.demographics["Age Range"] = st.selectbox(
            "Age Range",
            DEMOGRAPHICS["Age Range"],
            index=0 if "Age Range" not in st.session_state.demographics else
                   DEMOGRAPHICS["Age Range"].index(st.session_state.demographics.get("Age Range", DEMOGRAPHICS["Age Range"][0]))
        )
    
    with col2:
        st.session_state.demographics["Skin Tone"] = st.selectbox(
            "Skin Tone",
            DEMOGRAPHICS["Skin Tone"],
            index=0 if "Skin Tone" not in st.session_state.demographics else
                   DEMOGRAPHICS["Skin Tone"].index(st.session_state.demographics.get("Skin Tone", DEMOGRAPHICS["Skin Tone"][0]))
        )
        
        st.session_state.demographics["Distinctive Features"] = st.multiselect(
            "Distinctive Features",
            DEMOGRAPHICS["Distinctive Features"],
            default=st.session_state.demographics.get("Distinctive Features", [])
        )
        
        additional_info = st.text_area(
            "Additional Notes",
            placeholder="Any other relevant information...",
            height=100
        )
        if additional_info:
            st.session_state.demographics["Additional Notes"] = additional_info
    
    st.markdown('<div class="info-box">✓ Demographics section helps contextualize facial features</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: FACE FEATURES
# ============================================================================

with tab2:
    st.markdown('<div class="section-header">👤 Facial Features</div>', unsafe_allow_html=True)
    
    # Create columns for face parts
    num_cols = 3
    cols = st.columns(num_cols)
    
    face_parts = list(FACE_FEATURES_DB.keys())
    
    for idx, face_part in enumerate(face_parts):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            # Expandable section for each face part
            with st.expander(f"**{face_part}**", expanded=False):
                
                if face_part not in st.session_state.feature_selections:
                    st.session_state.feature_selections[face_part] = {}
                
                attributes = FACE_FEATURES_DB[face_part]
                
                for attribute, values in attributes.items():
                    # Create unique keys for selectbox
                    current_value = st.session_state.feature_selections[face_part].get(attribute, values[0])
                    
                    selected_value = st.selectbox(
                        label=attribute.replace("_", " ").title(),
                        options=values,
                        index=values.index(current_value) if current_value in values else 0,
                        key=f"{face_part}_{attribute}"
                    )
                    
                    # Handle "Other" option
                    if selected_value == "Other":
                        custom_value = st.text_input(
                            f"Specify {attribute.lower()}",
                            placeholder="Enter custom description",
                            key=f"custom_{face_part}_{attribute}"
                        )
                        if custom_value:
                            st.session_state.feature_selections[face_part][attribute] = custom_value
                        else:
                            st.session_state.feature_selections[face_part][attribute] = "Other"
                    else:
                        st.session_state.feature_selections[face_part][attribute] = selected_value
    
    st.markdown('<div class="info-box">ℹ️ Select "Other" to provide custom descriptions for any feature</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3: IMAGE GENERATION
# ============================================================================

with tab3:
    st.markdown('<div class="section-header">🖼️ Generate Suspect Sketch</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Generated Description Prompt")
        
        # Generate and display prompt
        generated_prompt = generate_description_prompt()
        st.session_state.generated_prompt = generated_prompt
        
        st.text_area(
            "Description Prompt (auto-generated)",
            value=generated_prompt,
            height=150,
            disabled=True,
            key="prompt_display"
        )
    
    with col2:
        st.markdown("### Generation Settings")
        
        image_height = st.slider("Image Height", 256, 1024, 512, step=64)
        image_width = st.slider("Image Width", 256, 1024, 512, step=64)
        inference_steps = st.slider("Quality (Steps)", 20, 100, 50, step=5)
        guidance_scale = st.slider("Prompt Adherence", 0.0, 20.0, 7.5, step=0.5)
    
    st.divider()
    
    # API Configuration
    st.markdown("### API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input(
            "Backend API URL",
            value="http://localhost:8000/api",
            placeholder="http://localhost:8000/api"
        )
    
    with col2:
        api_status = st.empty()
    
    # Check API health
    try:
        health_response = requests.get(f"{api_url.rsplit('/api', 1)[0]}/health", timeout=2)
        if health_response.status_code == 200:
            api_status.success("✅ API Online")
        else:
            api_status.warning("⚠️ API Unreachable")
    except:
        api_status.error("❌ API Offline")
    
    st.divider()
    
    # Generation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Generate Caption (LLaVa)", use_container_width=True, type="secondary"):
            if not st.session_state.feature_selections:
                st.warning("Please select at least one facial feature first")
            else:
                with st.spinner("Generating caption..."):
                    try:
                        # Prepare attributes for API
                        attributes = []
                        for face_part, attrs in st.session_state.feature_selections.items():
                            for attr, value in attrs.items():
                                if value and value != "None":
                                    attributes.append({
                                        "category": face_part,
                                        "attribute": attr,
                                        "value": value
                                    })
                        
                        response = requests.post(
                            f"{api_url}/caption",
                            json={
                                "attributes": attributes,
                                "description": st.session_state.demographics.get("Additional Notes", "")
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            caption_data = response.json()
                            st.session_state.generated_prompt = caption_data["dense_prompt"]
                            st.success("✅ Caption generated!")
                            st.info(f"**Enhanced Prompt:** {caption_data['dense_prompt']}")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
    
    with col2:
        if st.button("🎨 Generate Image (RealVisXL)", use_container_width=True, type="primary"):
            if not st.session_state.generated_prompt:
                st.warning("Please generate a caption first or fill in features")
            else:
                with st.spinner("Generating sketch image (this may take 1-2 minutes)..."):
                    try:
                        response = requests.post(
                            f"{api_url}/generate-image",
                            json={
                                "prompt": st.session_state.generated_prompt,
                                "height": image_height,
                                "width": image_width,
                                "num_inference_steps": inference_steps,
                                "guidance_scale": guidance_scale
                            },
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            image_data = response.json()
                            if image_data.get("image_base64"):
                                img_bytes = base64.b64decode(image_data["image_base64"])
                                st.session_state.generated_image = Image.open(BytesIO(img_bytes))
                                st.success("✅ Image generated!")
                            else:
                                st.info("Placeholder image created (enable models for real generation)")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
    
    with col3:
        if st.button("🔄 Regenerate", use_container_width=True, type="secondary"):
            st.session_state.generated_image = None
            st.rerun()
    
    st.divider()
    
    # Display generated image
    if st.session_state.generated_image:
        st.markdown("### Generated Suspect Sketch")
        st.image(st.session_state.generated_image, caption="AI-Generated Facial Reconstruction", use_column_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬇️ Download Image", use_container_width=True):
                # Save and offer download
                img_bytes = BytesIO()
                st.session_state.generated_image.save(img_bytes, format="PNG")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes.getvalue(),
                    file_name=f"suspect_sketch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        
        with col2:
            if st.button("🗑️ Clear Image", use_container_width=True):
                st.session_state.generated_image = None
                st.rerun()

# ============================================================================
# TAB 4: SUMMARY & EXPORT
# ============================================================================

with tab4:
    st.markdown('<div class="section-header">📊 Suspect Description Summary</div>', unsafe_allow_html=True)
    
    # Create comprehensive summary
    summary_text = f"""
## SUSPECT DESCRIPTION REPORT
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### DEMOGRAPHICS
"""
    
    if st.session_state.demographics:
        for key, value in st.session_state.demographics.items():
            if key != "Additional Notes":
                summary_text += f"- **{key}:** {value}\n"
        if "Additional Notes" in st.session_state.demographics:
            summary_text += f"\n**Additional Notes:** {st.session_state.demographics['Additional Notes']}\n"
    
    summary_text += "\n### FACIAL FEATURES\n"
    
    if st.session_state.feature_selections:
        for face_part, attributes in st.session_state.feature_selections.items():
            if attributes:
                summary_text += f"\n**{face_part}:**\n"
                for attr, value in attributes.items():
                    summary_text += f"  - {attr.title()}: {value}\n"
    
    summary_text += f"\n### GENERATED DESCRIPTION\n{st.session_state.generated_prompt}\n"
    
    # Display summary
    st.markdown(summary_text)
    
    st.divider()
    
    # Export options
    st.markdown("### Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as Text", use_container_width=True):
            st.download_button(
                label="Download TXT",
                data=export_description(summary_text, "txt"),
                file_name=f"suspect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📋 Export as JSON", use_container_width=True):
            st.download_button(
                label="Download JSON",
                data=export_description(summary_text, "json"),
                file_name=f"suspect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.feature_selections = {}
            st.session_state.demographics = {}
            st.session_state.generated_prompt = ""
            st.session_state.generated_image = None
            st.success("All data cleared")
            st.rerun()
    
    st.divider()
    
    # Quick reference
    with st.expander("📖 System Information & Tips"):
        st.markdown("""
        ### How to Use This System:
        
        1. **Start with Demographics** (Tab 1)
           - Provide suspect's basic information
           - Select gender, race, age range, and skin tone
           - Add any distinctive features
        
        2. **Document Facial Features** (Tab 2)
           - Expand each face part category
           - Select attributes that match the suspect
           - Use "Other" option to provide custom descriptions
        
        3. **Generate Image** (Tab 3)
           - System auto-generates description from your selections
           - Use LLaVa to enhance the description if available
           - Generate AI sketch using RealVisXL
           - Adjust quality settings as needed
        
        4. **Export Report** (Tab 4)
           - Review complete summary
           - Export as TXT or JSON for records
        
        ### Tips for Best Results:
        - Be specific and accurate with feature descriptions
        - Use multiple details for each face part
        - Include distinctive scars, tattoos, or marks
        - Update demographics completely
        - Export reports for official documentation
        
        ### System Status:
        - Backend API: Check on Image Generation tab
        - Models: LLaVa & RealVisXL currently in placeholder mode
        - Enable models in backend for real AI generation
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    <p>🔍 Forensic Face Description System v1.0</p>
    <p>For law enforcement use only. All descriptions should be verified through official channels.</p>
</div>
""", unsafe_allow_html=True)
