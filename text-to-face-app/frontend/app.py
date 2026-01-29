import streamlit as st
import base64
import os
import sys

# Add project root to path so we can import from 'data' and 'config'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.schema import SUSPECT_SCHEMA, MANDATORY_FIELDS
from frontend.styles import apply_custom_styles, render_sidebar
from frontend.cache_manager import load_cache, get_lookup_key
from frontend.api_client import BackendGateway

# --- HELPER: RESET STATE CALLBACK ---
def clear_outputs():
    """Callback to clear generated results when inputs change."""
    if 'generated_prompt' in st.session_state:
        st.session_state['generated_prompt'] = None

def main():
    st.set_page_config(page_title="Text to Face", layout="wide")
    apply_custom_styles()

    # Session State Init
    if 'generated_prompt' not in st.session_state:
        st.session_state['generated_prompt'] = None
    if 'last_selections' not in st.session_state:
        st.session_state['last_selections'] = {}

    # Load Data
    cache_db = load_cache()
    use_stored = render_sidebar(len(cache_db))

    st.title("Text to Face")
    
    user_selections = {}

    # --- LAYOUT & FORM RENDER ---
    col_input, col_output = st.columns([2, 1])

    with col_input:
        for category, attributes in SUSPECT_SCHEMA.items():
            st.markdown(f"**{category}**")
            cols = st.columns(4)
            
            widget_queue = []
            for attr_name, options in attributes.items():
                widget_queue.append({"type": "dropdown", "label": attr_name, "options": options})
                if st.session_state.get(f"{attr_name}_select") == "Other":
                    widget_queue.append({"type": "text_input", "label": attr_name})

            for i, widget in enumerate(widget_queue):
                with cols[i % 4]:
                    is_mandatory = widget["label"] in MANDATORY_FIELDS
                    
                    if widget["type"] == "dropdown":
                        # 1. Build Options
                        opts = [] if is_mandatory else ["None / Unspecified"]
                        opts.extend(widget["options"] + ["Other"])
                        
                        sb_key = f"{widget['label']}_select"
                        current_val = st.session_state.get(sb_key)

                        # --- FIX: Handle Default Logic Explicitly ---
                        if current_val is None and opts:
                            current_val = opts[0]
                        # --------------------------------------------
                        
                        # --- LABEL STYLING ---
                        clean_label = widget['label']
                        if is_mandatory:
                            clean_label += " *"
                        
                        # Green color check
                        if current_val and current_val not in ["None / Unspecified", None]:
                            label_text = f":green[{clean_label}]"
                        else:
                            label_text = clean_label
                        
                        # Render Widget
                        val = st.selectbox(
                            label_text, 
                            opts, 
                            key=sb_key, 
                            on_change=clear_outputs
                        )
                        
                        if val == "Other":
                            user_selections[widget['label']] = "Other (Unspecified)"
                        elif val and val != "None / Unspecified":
                            user_selections[widget['label']] = val

                    elif widget["type"] == "text_input":
                        custom_val = st.text_input(
                            "Specify:", 
                            key=f"{widget['label']}_custom",
                            on_change=clear_outputs
                        )
                        if custom_val:
                            user_selections[widget['label']] = custom_val

    # --- ACTIONS & LOGIC ---
    with col_output:
        st.markdown("**Actions**")
        missing = [f for f in MANDATORY_FIELDS if f not in user_selections]
        
        final_data = {k: v for k, v in user_selections.items() if v and "Unspecified" not in v}

        # 2. AUTO-CLEAR LOGIC
        if st.session_state['generated_prompt'] is not None:
            if final_data != st.session_state['last_selections']:
                st.session_state['generated_prompt'] = None
                st.rerun()

        # 3. GENERATE BUTTON
        if st.button("Generate Description", type="primary", use_container_width=True):
            if missing:
                st.error(f"Missing: {', '.join(missing)}")
            else:
                st.session_state['last_selections'] = final_data.copy()
                
                lookup_key = get_lookup_key(final_data)
                
                # Pre-calculate the prompt string based on selections
                prompt_str = ", ".join([f"{k}: {v}" for k, v in final_data.items()])

                if use_stored and lookup_key in cache_db:
                    st.session_state['generated_prompt'] = cache_db[lookup_key].get("description", "")
                elif use_stored:
                    # CHANGED: Now displays the constructed prompt instead of None
                    st.warning("Not in Cache. Displaying raw prompt from selections:")
                    st.session_state['generated_prompt'] = prompt_str
                else:
                    st.session_state['generated_prompt'] = BackendGateway.generate_caption(prompt_str, final_data)

        # 4. SHOW OUTPUT
        if st.session_state['generated_prompt']:
            st.info(st.session_state['generated_prompt'])
            
            if st.button("Generate Image", type="secondary", use_container_width=True):
                selections = st.session_state.get('last_selections', final_data)
                key = get_lookup_key(selections)
                
                if use_stored and key in cache_db:
                    img_path = cache_db[key]["image_path"]
                    full_path = os.path.join(os.getcwd(), img_path)
                    if os.path.exists(full_path):
                        st.image(full_path)
                    else:
                        st.error(f"Image file missing at {full_path}")
                elif use_stored:
                     st.warning("Not in Cache.")
                else:
                    b64_img = BackendGateway.generate_image(st.session_state['generated_prompt'])
                    if b64_img:
                        st.image(base64.b64decode(b64_img))
                    else:
                        st.error("Failed to generate image.")

if __name__ == "__main__":
    main()