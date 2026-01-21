# -*- coding: utf-8 -*-
import json
from pathlib import Path
import streamlit as st
from core.prompt_builder import compose_prompt
from core.model_runner import RealVizGenerator

st.set_page_config(page_title="Forensic Text-to-Face (RealViz)", page_icon="🧑‍⚖️", layout="wide")

APP_DIR = Path(__file__).parent
SCHEMA_PATH = APP_DIR / 'schema' / 'attributes.json'
CFG_PATH = APP_DIR / 'model_config.yaml'

@st.cache_data(show_spinner=False)
def load_schema():
    return json.loads(SCHEMA_PATH.read_text())

@st.cache_resource(show_spinner=True)
def load_generator():
    return RealVizGenerator(str(CFG_PATH))

st.title("Forensic Text-to-Face (RealViz)")
colA, colB = st.columns([2,1])
with colA:
    st.caption("Edit the schema file to add/remove attributes. Click **Reload schema** to reflect changes.")
with colB:
    if st.button("Reload schema"):
        load_schema.clear()
        st.rerun()

schema = load_schema()

# Collect inputs dynamically
values = {}
with st.form("attributes_form"):
    for key, cfg in schema.items():
        t = cfg.get('type', 'dropdown')
        if t == 'dropdown':
            values[key] = st.selectbox(key, cfg.get('options', []), help=cfg.get('help'))
        elif t == 'text':
            values[key] = st.text_area(key, value="", placeholder=cfg.get('placeholder', ''))
        else:
            st.warning(f"Unsupported field type: {t} for {key}")
    c1, c2, c3 = st.columns(3)
    with c1:
        submit = st.form_submit_button("Generate Face", use_container_width=True)
    with c2:
        adv = st.toggle("Advanced settings")
    with c3:
        show_prompt = st.toggle("Show prompt")

if submit:
    # Compose prompt under ~73 tokens
    prompt = compose_prompt(values, clip_token_limit=73)
    gen = load_generator()

    if adv:
        with st.sidebar:
            st.header("Advanced")
            inf = gen.icfg
            inf.guidance_scale = st.slider("Guidance scale", 1.0, 12.0, inf.guidance_scale, 0.5)
            inf.num_inference_steps = st.slider("Denoising steps", 10, 75, inf.num_inference_steps, 1)
            inf.height = st.selectbox("Height", [512, 576, 640, 768], index=0)
            inf.width = st.selectbox("Width", [512, 576, 640, 768], index=0)
            seed_opt = st.text_input("Seed (optional)", value=str(inf.seed) if inf.seed else "")
            try:
                inf.seed = int(seed_opt) if seed_opt.strip() else None
            except ValueError:
                st.warning("Seed must be an integer.")

    with st.spinner("Generating image…"):
        img = gen.generate(prompt)
    if img is None:
        st.error("Generation failed. Check model paths in model_config.yaml")
    else:
        if show_prompt:
            st.subheader("Prompt used")
            st.code(prompt)
        st.subheader("Generated image")
        st.image(img, use_column_width=True)
