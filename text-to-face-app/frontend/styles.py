import streamlit as st

def apply_custom_styles():
    """Injects custom CSS for a compact layout and dynamic green text/borders."""
    st.markdown("""
        <style>
                /* --- LAYOUT --- */
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                h1 { font-size: 1.8rem !important; margin-bottom: 0rem; }
                p { font-size: 0.9rem; margin-bottom: 0.2rem; }
                div[data-testid="column"] { gap: 0rem; }
                
                /* --- 1. TARGET THE LABEL --- */
                /* We detect the style attribute because Streamlit renders :green[] as inline styles */
                span[data-testid="stMarkdownContainer"] span[style*="color"] {
                    text-shadow: 0 0 5px #00ff00;
                    font-weight: 600;
                }

                /* --- 2. TARGET THE DROPDOWN (Selectbox) --- */
                /* If the box has a colored label, turn the inner text GREEN */
                div[data-testid="stSelectbox"]:has(span[style*="color"]) div[data-baseweb="select"] div {
                    color: #28a745 !important;
                    -webkit-text-fill-color: #28a745 !important; /* Required for some browsers */
                    font-weight: 600 !important;
                }
                
                /* Change the Border to Green as well */
                div[data-testid="stSelectbox"]:has(span[style*="color"]) div[data-baseweb="select"] > div {
                    border-color: #28a745 !important;
                    border-width: 2px !important;
                }

                /* --- 3. TARGET THE TEXT INPUT (Other) --- */
                /* If the input has a colored label, turn the typed text GREEN */
                div[data-testid="stTextInput"]:has(span[style*="color"]) input {
                    color: #28a745 !important;
                    -webkit-text-fill-color: #28a745 !important;
                    font-weight: 600 !important;
                }

                /* Change the Border to Green */
                div[data-testid="stTextInput"]:has(span[style*="color"]) div[data-baseweb="input"] > div {
                    border-color: #28a745 !important;
                    border-width: 2px !important;
                }

        </style>
    """, unsafe_allow_html=True)

def render_sidebar(cache_size):
    """Renders the sidebar and returns the 'Use Stored Data' toggle state."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        use_stored = st.checkbox("Use Stored Data", value=True)
        if use_stored:
            st.success(f"📂 Cache: {cache_size} items")
        else:
            st.info("📡 Live API Mode")
        return use_stored