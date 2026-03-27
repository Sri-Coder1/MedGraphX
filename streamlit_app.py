import streamlit as st
import requests
import base64
import os
import io
import math
import textwrap
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit.components.v1 as components
import plotly.graph_objects as go
from urllib.parse import quote
from module6_risk_detection import get_food_nutrients, rule_engine, train_model, get_safe_foods
from module7_dashboard import show_dashboard, show_nutrient_page
from module4_nlp_preprocessing import (
    run_nlp_pipeline,
    extract_entities_spacy,
    build_graph_dynamic,
    visualize_graph_dynamic,
)

API_BASE = "http://127.0.0.1:5000"

# Load banner image as base64 for CSS embedding
def get_banner_base64():
    banner_path = os.path.join(os.path.dirname(__file__), "banner.png")
    if os.path.exists(banner_path):
        with open(banner_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

BANNER_B64 = get_banner_base64()


def get_dashboard_banner_base64():
    dashboard_banner_path = os.path.join(os.path.dirname(__file__), "BANNER_B64.png")
    if os.path.exists(dashboard_banner_path):
        with open(dashboard_banner_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


DASHBOARD_BANNER_B64 = get_dashboard_banner_base64()

OPEN_EYE_SVG = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#000000' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z'/><circle cx='12' cy='12' r='2.5'/></svg>"
CLOSED_EYE_SVG = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#000000' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 14c2-3 5-4.5 8-4.5s6 1.5 8 4.5'/><path d='M6 12l-1.5-1.5'/><path d='M9 10.5L8 8.5'/><path d='M12 10V8'/><path d='M15 10.5l1-2'/><path d='M18 12l1.5-1.5'/></svg>"
OPEN_EYE_ICON = f"![open-eye](data:image/svg+xml;utf8,{quote(OPEN_EYE_SVG)})"
CLOSED_EYE_ICON = f"![closed-eye](data:image/svg+xml;utf8,{quote(CLOSED_EYE_SVG)})"

st.set_page_config(
    page_title="MedGraphX",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# SESSION STATE
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "auth"

if "token" not in st.session_state:
    st.session_state.token = None

if "user" not in st.session_state:
    st.session_state.user = None

if "loaded_profile" not in st.session_state:
    st.session_state.loaded_profile = None

if "profile_pic" not in st.session_state:
    st.session_state.profile_pic = None

if "user_login_show_password" not in st.session_state:
    st.session_state.user_login_show_password = False

if "login_password_value" not in st.session_state:
    st.session_state.login_password_value = ""

if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"

if "show_login" not in st.session_state:
    st.session_state.show_login = False

if "show_register" not in st.session_state:
    st.session_state.show_register = False


# ----------------------------
# CSS
# ----------------------------
st.markdown(f"""
<style>
            html, body {{
    margin: 0;
    padding: 0;
}}

    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
/* 1. Hide the default Streamlit header */
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0 !important;
        min-height: 0 !important;
    }}

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    #MainMenu {{
        display: none !important;
    }}
            

    /* 2. Remove padding from the main content area */
    .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }}

    /* 3. Keep the app shell clean so the top bar reaches the very top */
    .stApp {{
        background: #ffffff;
    }}

    .block-container {{
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }}

    /* TOP NAVBAR */
    .top-bar {{
        background: linear-gradient(90deg, #0f4fa8, #0f8fd4);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 40px;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 99990;
        margin: 0 !important;
        min-height: 100px; /* Increased slightly to ensure buttons don't clip */
        width: 100vw !important;
        box-sizing: border-box;
        overflow: visible; /* Prevents internal elements from spilling out */
    }}

    .top-bar-spacer {{
        height: 100px;
        width: 100%;
        flex-shrink: 0;
    }}
    .top-title {{
        color: white;
        font-size: 24px; /* Slightly smaller to save horizontal space */
        font-weight: 800;
        letter-spacing: 0.3px;
        line-height: 44px;
        white-space: nowrap; /* Prevents the title from wrapping */
        overflow: hidden;
        text-overflow: ellipsis; /* Adds '...' if title gets too squeezed */
        flex-shrink: 1; /* Allows title to shrink before buttons do */
        margin-right: 20px;
    }}

    .top-buttons {{
        display: flex;
        gap: 10px;
        flex-shrink: 0;
    }}

    .top-buttons a {{
        background: white;
        color: #0f4fa8;
        padding: 10px 28px;
        border-radius: 20px;
        text-decoration: none;
        font-weight: 700;
        font-size: 15px;
        border: none;
        cursor: pointer;
    }}

    .top-buttons a:hover {{
        background: #e8f0fe;
    }}

    /* Pull Streamlit buttons into the top bar area */
.st-key-nav_login, .st-key-nav_register, .st-key-nav_data_source, .st-key-nav_analysis, .st-key-nav_risk_prediction {{
    position: fixed !important;
    top: 28px !important;
    z-index: 99995;
    display: flex !important;
    align-items: center !important;
    height: 44px !important;
    margin: 0 !important;
}}

.st-key-nav_login {{
    right: 180px !important;
}}

.st-key-nav_register {{
    right: 40px !important;
}}

.st-key-nav_login button, .st-key-nav_register button, .st-key-nav_data_source button, .st-key-nav_analysis button, .st-key-nav_risk_prediction button {{
   background: white !important;
    color: #0f4fa8 !important;
    border-radius: 25px !important;
    padding: 10px 16px !important;
    font-weight: 700 !important;
    font-size: 14px !important; /* Slightly smaller text helps fit the bar */
    white-space: nowrap !important; /* Prevents text from being cut off/wrapped */
    width: auto !important;
    min-width: max-content !important;
}}

.st-key-nav_login button:hover, .st-key-nav_register button:hover, .st-key-nav_data_source button:hover, .st-key-nav_analysis button:hover, .st-key-nav_risk_prediction button:hover {{
    background: #e8f0fe !important;
    color: #0f4fa8 !important;
    transform: none !important;
}}

    /* Blur overlay for auth popup */
    .auth-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 20, 60, 0.55);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        z-index: 99998;
    }}

    /* Hide the ESC close trigger button */
    .st-key-esc_close_btn {{
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
        padding: 0 !important;
        margin: -1px !important;
    }}

    /* Auth popup form container - sits above the overlay */
    .st-key-auth_popup_container {{
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 99999 !important;
        background: white !important;
        border-radius: 22px !important;
        padding: 40px 36px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
        max-width: 480px !important;
        width: 90vw !important;
    }}

    .st-key-auth_popup_container .stTextInput label,
    .st-key-auth_popup_container .stMarkdown {{
        color: #0f172a !important;
    }}

    .st-key-auth_popup_container h3 {{
        color: #0f4fa8 !important;
        font-size: 26px !important;
        font-weight: 800 !important;
    }}

    /* MAIN SPLIT LAYOUT */
    .main-container {{
        display: flex;
        align-items: stretch;
        justify-content: space-between;
        min-height: calc(100vh - 80px);
        margin-top: 6px;
        background: #ffffff;
    }}

    .left-image {{
        width: 55%;
        background: url('data:image/png;base64,{BANNER_B64}') no-repeat center center;
        background-size: cover;
        min-height: calc(100vh - 80px);
    }}


    .right-content {{
        width: 45%;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 48px 40px;
        background: #ffffff;
    }}

    .hero-text {{
        text-align: center;
        max-width: 600px;
    }}

    .hero-title {{
        font-size: 44px;
        font-weight: 900;
        color: #0f4fa8;
        margin-bottom: 20px;
        line-height: 1.2;
    }}

    .hero-sub {{
        font-size: 18px;
        color: #334155;
        line-height: 1.8;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    section[data-testid="stSidebar"] * {{
        font-size: 18px !important;
        font-weight: 600;
        color: #17343b !important;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #e0f7fa, #f1fffb);
        border-right: 1px solid rgba(15, 143, 212, 0.2);
        padding-top: 20px;
        margin-top: 0 !important;
    }}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

    .hero-box {
        background: linear-gradient(135deg, #0f4fa8 0%, #0f8fd4 45%, #11b4a6 100%);
        border-radius: 30px;
        padding: 34px 38px;
        box-shadow: 0 18px 40px rgba(15, 79, 168, 0.16);
        border: 1px solid rgba(255,255,255,0.22);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 900;
        color: #0f4fa8;
        margin-bottom: 20px;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #edfaff;
        font-size: 16px;
        line-height: 1.7;
        max-width: 920px;
    }

    .main-card {
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 26px;
        padding: 24px;
        box-shadow: 0 14px 30px rgba(21, 53, 74, 0.08);
        backdrop-filter: blur(14px);
        margin-bottom: 18px;
    }

    .portal-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(246,252,255,0.96));
        border: 1px solid rgba(15, 143, 212, 0.10);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(22, 52, 58, 0.06);
        margin-bottom: 12px;
    }

    .section-title {
        font-size: 32px;
        font-weight: 800;
        color: #113f66;
        margin-bottom: 8px;
    }

    .section-subtitle {
        font-size: 17px;
        color: #5f7680;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .portal-title {
        font-size: 22px;
        font-weight: 800;
        color: #0f518f;
        margin-bottom: 6px;
    }

    .portal-desc {
        font-size: 14px;
        color: #627b84;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 9px 15px;
        border-radius: 999px;
        background: rgba(17, 180, 166, 0.10);
        border: 1px solid rgba(17, 180, 166, 0.16);
        color: #0d6963;
        font-size: 15px;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(223, 240, 252, 0.85);
        border: 1px solid rgba(15, 143, 212, 0.08);
        border-radius: 18px;
        padding: 10px;
        margin-bottom: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        border-radius: 14px;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(15, 79, 168, 0.08);
        color: #17517a;
        font-weight: 800;
        padding: 0 18px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f4fa8, #10b981) !important;
        color: white !important;
        box-shadow: 0 10px 18px rgba(15, 79, 168, 0.14);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: rgba(255,255,255,0.98) !important;
        border: 1px solid rgba(15, 143, 212, 0.16) !important;
        border-radius: 10px !important;
        color: #17343b !important;
        min-height: 40px !important;
    }

    textarea, input {
        color: #17343b !important;
    }

    /* Number input step buttons - white bg, black icons, inside the box */
    .stNumberInput button {
        background: #ffffff !important;
        color: #1a1a2e !important;
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
        padding: 0 !important;
        min-width: 32px !important;
        width: 32px !important;
        height: 100% !important;
    }

    .stNumberInput button:hover {
        background: #f0f4ff !important;
        color: #0f4fa8 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    .stNumberInput button svg {
        fill: #1a1a2e !important;
        stroke: #1a1a2e !important;
    }

    /* Compact number inputs */
    .stNumberInput div[data-baseweb="input"] {
        min-height: 40px !important;
    }

    /* File uploader white background */
    .stFileUploader section {
        background: #ffffff !important;
        border: 1px dashed rgba(15, 143, 212, 0.3) !important;
        border-radius: 10px !important;
        color: #1b465b !important;
    }

    .stFileUploader section * {
        color: #1b465b !important;
    }

    .stFileUploader section button {
        background: #f0f4ff !important;
        color: #0f4fa8 !important;
        border: 1px solid rgba(15, 143, 212, 0.2) !important;
        box-shadow: none !important;
    }

    .stFileUploader label {
        color: #1b465b !important;
    }

    label, .stMarkdown, .stText {
        color: #1b465b !important;
        font-size: 16px !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f4fa8 0%, #0f8fd4 55%, #10b981 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.82rem 1rem;
        font-weight: 800;
        box-shadow: 0 10px 20px rgba(15, 79, 168, 0.14);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(15, 79, 168, 0.18);
    }

    /* Keep only eye-icon look for password visibility toggle */
    .st-key-user_login_toggle_eye button {
        width: 48px !important;
        min-width: 48px !important;
        height: 48px !important;
        padding: 0 !important;
        border-radius: 14px !important;
        border: 1px solid rgba(15, 143, 212, 0.18) !important;
        background: rgba(255, 255, 255, 0.98) !important;
        color: #17517a !important;
        box-shadow: none !important;
        transform: none !important;
        margin-top: 30px !important;
        font-size: 20px !important;
        line-height: 1 !important;
    }

    .st-key-user_login_toggle_eye button:hover {
        background: rgba(246, 252, 255, 1) !important;
        color: #0f518f !important;
        box-shadow: none !important;
        transform: none !important;
    }

    .st-key-user_login_toggle_eye button img {
        width: 20px !important;
        height: 20px !important;
    }

    /* Keep the right side of password input clean and thin */
    .st-key-user_login_password div[data-baseweb="input"] > div {
        background: rgba(255,255,255,0.98) !important;
        border: 1px solid rgba(15, 143, 212, 0.16) !important;
        padding-right: 8px !important;
    }

    .st-key-user_login_password input[type="password"]::-ms-reveal,
    .st-key-user_login_password input[type="password"]::-ms-clear {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }

    .st-key-user_login_password input[type="password"]::-webkit-credentials-auto-fill-button,
    .st-key-user_login_password input[type="password"]::-webkit-contacts-auto-fill-button,
    .st-key-user_login_password input[type="password"]::-webkit-textfield-decoration-container {
        visibility: hidden !important;
        display: none !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(15, 143, 212, 0.12);
        padding: 10px 14px;
        border-radius: 14px;
        box-shadow: 0 4px 10px rgba(20, 40, 46, 0.06);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    .footer-note {
        color: #5d7882;
        font-size: 14px;
        margin-top: 8px;
    }

    /* Fix metric text visibility */
    div[data-testid="stMetric"] label {
        color: #1b465b !important;
        font-weight: 600;
        font-size: 13px !important;
    }

    div[data-testid="stMetric"] div {
        color: #0f172a !important;
        font-weight: 700;
        font-size: 18px;
    }

    /* Clean profile page layout */
    .profile-pic-section {
        display: flex;
        align-items: center;
        gap: 18px;
        margin: 10px 0 16px 0;
    }

    .st-key-profile_pic_upload {
        max-width: 350px;
        margin-top: -10px !important;
    }

    .stNumberInput, .stSelectbox, .stTextArea {
        margin-bottom: -6px !important;
    }

    /* Compact text areas */
    .stTextArea textarea {
        min-height: 80px !important;
    }

    /* Profile label sizes */
    .stNumberInput label, .stSelectbox label, .stTextArea label, .stFileUploader label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #1b465b !important;
    }

    /* Input text size */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] span,
    .stTextArea textarea {
        font-size: 16px !important;
    }


</style>


""", unsafe_allow_html=True)



# ----------------------------
# HELPERS
# ----------------------------
def show_header():
    st.markdown("""
        <div class="hero-box">
            <div class="hero-title" style="color: white;">MedGraphX</div>
            <div class="hero-subtitle">
                Cross-Domain Medical Knowledge Graph for Safe Dietary Recommendations across
                pharmaceutical, medicine, and food interaction domains.
            </div>
        </div>
    """, unsafe_allow_html=True)

def hero_section():
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200&h=400&fit=crop", use_container_width=True)

    st.markdown("""
    <div style='margin-top:-140px; padding:30px; background: rgba(0,0,0,0.5); border-radius: 10px; backdrop-filter: blur(5px);'>
        <h1 style='color:white; font-size:48px; font-weight:800;'>MedGraphX</h1>
        <p style='color:#e0f7fa; font-size:18px; max-width:600px;'>
        AI-powered medical knowledge graph for safe dietary and drug interaction analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

def auth_popup():

    if st.session_state.get("show_login") or st.session_state.get("show_register"):

        # Blur overlay background (click overlay to close)
        st.markdown('<div class="auth-overlay" id="auth-overlay"></div>', unsafe_allow_html=True)

        # Popup form - rendered in a keyed container so CSS can position it
        with st.container(key="auth_popup_container"):
            # LOGIN
            if st.session_state.get("show_login"):
                st.markdown("### 🔐 Login")

                email = st.text_input("Email", key="login_email")
                pwd_col, eye_col = st.columns([6, 1], vertical_alignment="bottom")
                with pwd_col:
                    with st.container(key="user_login_password"):
                        password = st.text_input(
                            "Password",
                            type="default" if st.session_state.user_login_show_password else "password",
                            key="login_password_value",
                        )
                with eye_col:
                    with st.container(key="user_login_toggle_eye"):
                        if st.button(
                            "‿" if st.session_state.user_login_show_password else "👁",
                            key="login_toggle_eye_btn",
                            help="Show or hide password",
                        ):
                            st.session_state.user_login_show_password = not st.session_state.user_login_show_password
                            st.rerun()

                if st.button("Login Now"):
                    payload = {"email": email.strip(), "password": password.strip()}
                    res = requests.post(f"{API_BASE}/login", json=payload)
                    data = safe_json(res)

                    if res.status_code == 200:
                        st.session_state.token = data["token"]
                        st.session_state.user = data["user"]
                        # Auto-load profile after login
                        try:
                            headers = {"Authorization": f"Bearer {data['token']}"}
                            profile_res = requests.get(f"{API_BASE}/profile", headers=headers)
                            if profile_res.status_code == 200:
                                st.session_state.loaded_profile = safe_json(profile_res)
                            # Load profile picture
                            pic_res = requests.get(f"{API_BASE}/profile/pic", headers=headers)
                            if pic_res.status_code == 200:
                                st.session_state.profile_pic = pic_res.content
                        except Exception:
                            pass
                        if data["user"].get("role") == "admin":
                            st.session_state.page = "admin"
                        else:
                            st.session_state.page = "dashboard"
                        st.session_state.show_login = False
                        st.rerun()
                    else:
                        st.error("Login failed")

            # REGISTER
            if st.session_state.get("show_register"):
                st.markdown("### 📝 Register")

                name = st.text_input("Full Name", key="reg_name")
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password", type="password", key="reg_password")

                if st.button("Create Account"):
                    payload = {
                        "full_name": name.strip(),
                        "email": email.strip(),
                        "password": password
                    }
                    res = requests.post(f"{API_BASE}/register", json=payload)

                    if res.status_code == 201:
                        st.success("Registered successfully")
                        st.session_state.show_register = False
                        st.session_state.show_login = True
                        st.rerun()
                    else:
                        st.error("Registration failed")

        # Hidden close button (invisible, triggered by ESC key via JS)
        if st.button("esc_close_trigger", key="esc_close_btn"):
            st.session_state.show_login = False
            st.session_state.show_register = False
            st.rerun()

        # Inject ESC key listener that clicks the hidden button
        components.html("""
        <script>
        (function() {
            const doc = window.parent.document;
            // Remove any previous listener to avoid duplicates
            if (doc._escHandler) {
                doc.removeEventListener('keydown', doc._escHandler);
            }
            doc._escHandler = function(e) {
                if (e.key === 'Escape') {
                    const buttons = doc.querySelectorAll('button');
                    for (const btn of buttons) {
                        if (btn.innerText.includes('esc_close_trigger')) {
                            btn.click();
                            break;
                        }
                    }
                }
            };
            doc.addEventListener('keydown', doc._escHandler);
        })();
        </script>
        """, height=0)

def features_section():
    st.markdown("## 🚀 Why MedGraphX", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.7); padding: 20px; border-radius: 15px; text-align: center;'>
            <h3>🧠 Knowledge Graph</h3>
            <p style='font-size: 14px;'>Visualize drug-food relationships intelligently</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.7); padding: 20px; border-radius: 15px; text-align: center;'>
            <h3>📊 Health Dashboard</h3>
            <p style='font-size: 14px;'>Track your health metrics over time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.7); padding: 20px; border-radius: 15px; text-align: center;'>
            <h3>🥗 Smart Diet</h3>
            <p style='font-size: 14px;'>Personalized food recommendations</p>
        </div>
        """, unsafe_allow_html=True)

def top_navbar():
    col1, col2, col3 = st.columns([6, 1, 1])

    with col1:
        st.markdown("""
        <div style='font-size:22px; font-weight:800; color:#0f4fa8;'>
        MedGraphX Knowledge Mapping Tool
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.page == "auth":
        with col2:
            if st.button("Login"):
                st.session_state.show_login = True
                st.session_state.show_register = False

        with col3:
            if st.button("Register"):
                st.session_state.show_register = True
                st.session_state.show_login = False

def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def safe_json(response):
    try:
        return response.json()
    except Exception:
        return {}

def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

def default_profile():
    return {
        "age": 0,
        "gender": "Male",
        "diseases": "",
        "medications": "",
        "allergies": "",
        "lifestyle": "",
        "height_cm": 0.0,
        "weight_kg": 0.0,
        "diet_preference": "Vegetarian"
    }

def logout_to_login():
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.loaded_profile = None
    st.session_state.profile_pic = None
    st.session_state.page = "auth"
    st.rerun()


# ----------------------------
# AUTH PAGE
# ----------------------------
def auth_page():
    # Top navbar
    st.markdown("""<div class="top-bar">
<div class="top-title"> MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)

    # Navbar buttons - positioned over the blue bar via CSS
    nav_spacer, btn_col1, btn_col2 = st.columns([7, 1, 1])
    with btn_col1:
        if st.button("Login", key="nav_login"):
            st.session_state.show_login = True
            st.session_state.show_register = False
            st.rerun()
    with btn_col2:
        if st.button("Register", key="nav_register"):
            st.session_state.show_register = True
            st.session_state.show_login = False
            st.rerun()

    # Main split layout - always visible
    st.markdown("""<div class="main-container">
<div class="left-image"></div>
<div class="right-content">
<div class="hero-text">
<div class="hero-title">Smarter health starts with connected knowledge<br>MedGraphX</div>
<div class="hero-sub">Bridges the gap between diseases, nutrition, and treatment—turning complex data into clear, actionable insights.</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # Auth popup overlays on top with blur
    auth_popup()


# ----------------------------
# PROFILE PAGE
# ----------------------------
def profile_page():
    # Top blue bar matching dashboard/login
    st.markdown("""<div class="top-bar">
<div class="top-title">MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)

    nav_spacer, btn_col1, btn_col2 = st.columns([7, 1, 1])
    with btn_col1:
        if st.button("📊 Dashboard", key="nav_login"):
            st.session_state.page = "dashboard"
            st.rerun()
    with btn_col2:
        if st.button("🚪 Logout", key="nav_register"):
            logout_to_login()

    if not st.session_state.token or not st.session_state.user:
        st.warning("Please login first.")
        return

    profile = st.session_state.loaded_profile or default_profile()

    st.markdown('<div style="padding: 10px 40px 30px 40px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">User Profile Management</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Manage patient health information for medicine-aware and food-safe recommendation support.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <span class="badge">👤 {st.session_state.user['full_name']}</span>
        <span class="badge">📧 {st.session_state.user['email']}</span>
        <span class="badge">🩺 USER PORTAL</span>
        """,
        unsafe_allow_html=True
    )

    age_value = safe_int(profile.get("age"), 0)
    gender_value = profile.get("gender") if profile.get("gender") in ["Male", "Female", "Other"] else "Male"
    height_value = safe_float(profile.get("height_cm"), 0.0)
    weight_value = safe_float(profile.get("weight_kg"), 0.0)
    sleep_value = safe_float(profile.get("sleep_hours"), 6.5)
    water_value = safe_float(profile.get("water_intake"), 1.5)
    diseases_value = profile.get("diseases") or ""
    medications_value = profile.get("medications") or ""
    allergies_value = profile.get("allergies") or ""
    lifestyle_value = profile.get("lifestyle") or ""
    diet_value = profile.get("diet_preference") or "Vegetarian"

    bmi = 0
    if height_value > 0 and weight_value > 0:
        bmi = round(weight_value / ((height_value / 100) ** 2), 2)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Age", age_value)
    with m2:
        st.metric("Height", f"{height_value} cm")
    with m3:
        st.metric("BMI", bmi if bmi > 0 else "N/A")
    with m4:
        st.metric("Weight", f"{weight_value} kg")

    # Profile Picture Upload - compact inline
    pic_col, upload_col = st.columns([1, 4], gap="small")
    with pic_col:
        if st.session_state.get("profile_pic") is not None:
            st.image(st.session_state.profile_pic, width=90)
        else:
            st.markdown("<div style='width:90px;height:90px;border-radius:50%;background:#e0f0ff;display:flex;align-items:center;justify-content:center;font-size:32px;'>👤</div>", unsafe_allow_html=True)
    with upload_col:
        uploaded_pic = st.file_uploader("Upload profile picture", type=["png", "jpg", "jpeg"], key="profile_pic_upload", label_visibility="collapsed")
        if uploaded_pic is not None:
            pic_bytes = uploaded_pic.getvalue()
            st.session_state.profile_pic = pic_bytes
            # Save to backend
            try:
                files = {"file": (uploaded_pic.name, io.BytesIO(pic_bytes), uploaded_pic.type)}
                requests.post(f"{API_BASE}/profile/pic", headers=auth_headers(), files=files)
            except Exception:
                pass
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=age_value)
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(gender_value)
        )
        height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=height_value, step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=weight_value, step=0.1)
        sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, max_value=24.0, value=sleep_value, step=0.5)
        water_intake = st.number_input("Water Intake (litres)", min_value=0.0, max_value=20.0, value=water_value, step=0.1)

        diet_options = [
            "Vegetarian",
            "Non-Vegetarian",
            "Vegan",
            "Low Sodium",
            "Diabetic Diet",
            "High Protein Diet"
        ]
        diet_index = diet_options.index(diet_value) if diet_value in diet_options else 0
        diet_preference = st.selectbox("Diet Preference", diet_options, index=diet_index)

    with col2:
        diseases = st.text_area("Diseases / Conditions", value=diseases_value, height=110)
        medications = st.text_area("Current Medications", value=medications_value, height=110)
        allergies = st.text_area("Allergies", value=allergies_value, height=110)
        lifestyle = st.text_area("Lifestyle Information", value=lifestyle_value, height=110)

    a, b = st.columns(2)
    with a:
        if st.button("Save Profile"):
            payload = {
                "age": int(age),
                "gender": gender,
                "diseases": diseases,
                "medications": medications,
                "allergies": allergies,
                "lifestyle": lifestyle,
                "height_cm": float(height_cm),
                "weight_kg": float(weight_kg),
                "diet_preference": diet_preference,
                "sleep_hours": float(sleep_hours),
                "water_intake": float(water_intake)
            }

            try:
                response = requests.post(f"{API_BASE}/profile", headers=auth_headers(), json=payload)
                data = safe_json(response)
                if response.status_code == 200:
                    st.session_state.loaded_profile = payload
                    st.success("Profile saved successfully.")
                else:
                    st.error(data.get("error", "Profile save failed."))
            except Exception as e:
                st.error(f"Backend connection error: {e}")

    with b:
        if st.button("Load Profile"):
            try:
                response = requests.get(f"{API_BASE}/profile", headers=auth_headers())
                data = safe_json(response)
                if response.status_code == 200:
                    st.session_state.loaded_profile = data
                    st.success("Profile loaded successfully.")
                    st.rerun()
                else:
                    st.error(data.get("error", "Could not load profile."))
            except Exception as e:
                st.error(f"Backend connection error: {e}")

    # Medical History / Reports Upload
    st.markdown("---")
    st.markdown("### 📁 Medical History & Reports")
    st.markdown("<p style='color:#5f7680; font-size:15px;'>Upload your medical reports, prescriptions, or health records for safekeeping.</p>", unsafe_allow_html=True)

    medical_file = st.file_uploader(
        "Upload medical documents",
        type=["pdf", "xls", "xlsx", "json", "csv", "png", "jpg", "jpeg"],
        key="medical_reports_upload",
        accept_multiple_files=True
    )

    if medical_file:
        if "medical_reports" not in st.session_state:
            st.session_state.medical_reports = []
        for f in medical_file:
            file_data = {"name": f.name, "size": f.size, "type": f.type}
            if file_data not in [r for r in st.session_state.medical_reports if isinstance(r, dict) and r.get("name") == f.name]:
                st.session_state.medical_reports.append(file_data)
        st.success(f"{len(medical_file)} file(s) uploaded successfully.")

    if st.session_state.get("medical_reports"):
        st.markdown("**Uploaded files:**")
        for report in st.session_state.medical_reports:
            st.markdown(f"- 📄 {report['name']} ({round(report['size']/1024, 1)} KB)")

    st.markdown('<div class="footer-note">Fill and save patient details securely for healthcare recommendation processing.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Add inner padding for the profile page content
    st.markdown("""
    <style>
        .block-container {
            padding-left: 40px !important;
            padding-right: 40px !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------
# ADMIN PAGE
# ----------------------------
def admin_page():
    st.markdown("""<div class="top-bar">
<div class="top-title">MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)

    nav_spacer, btn_col1, btn_col2 = st.columns([7, 1, 1])
    with btn_col1:
        if st.button("📊 Dashboard", key="nav_login"):
            st.session_state.page = "dashboard"
            st.rerun()
    with btn_col2:
        if st.button("🚪 Logout", key="nav_register"):
            logout_to_login()

    if not st.session_state.token or not st.session_state.user:
        st.warning("Please login first.")
        return

    if st.session_state.user["role"] != "admin":
        st.error("Only admin users can access this page.")
        return

    st.markdown('<div style="padding: 10px 40px 30px 40px;">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <span class="badge">👤 {st.session_state.user['full_name']}</span>
        <span class="badge">📧 {st.session_state.user['email']}</span>
        <span class="badge">🛡️ ADMIN ACCESS</span>
        """,
        unsafe_allow_html=True
    )

    # --- Fetch stats on load ---
    stats = None
    try:
        stats_resp = requests.get(f"{API_BASE}/admin/stats", headers=auth_headers(), timeout=10)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
    except Exception:
        pass

    # --- Overview Metrics ---
    st.markdown("### 📊 Platform Overview")
    if stats:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("👥 Total Users", stats.get("total_users", 0))
        with m2:
            st.metric("🧑‍💼 Regular Users", stats.get("user_count", 0))
        with m3:
            st.metric("🛡️ Admins", stats.get("admin_count", 0))
        with m4:
            st.metric("🩺 With Diseases", stats.get("profiles_with_diseases", 0))
        with m5:
            st.metric("💊 On Medications", stats.get("profiles_with_medications", 0))
    else:
        st.warning("Could not load platform stats. Make sure the Flask backend is running.")

    st.markdown("---")

    # --- Tabs for different admin sections ---
    admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5 = st.tabs(["👥 User Management", "📁 Medical Reports", "📥 Dataset Import","⭐Feedback","📊System analytics"])

    # ── TAB 1: User Management ──
    with admin_tab1:
        if st.button("🔄 Load All Users", key="admin_load_users"):
            try:
                response = requests.get(f"{API_BASE}/admin/user_details", headers=auth_headers(), timeout=10)
                data = safe_json(response)
                if response.status_code == 200:
                    st.session_state["admin_users_detail"] = data
                else:
                    st.error(data.get("error", "Failed to fetch users."))
            except Exception as e:
                st.error(f"Backend connection error: {e}")

            if st.session_state.get("admin_users_detail"):
                users_data = st.session_state["admin_users_detail"]

                df = pd.DataFrame(users_data)
                display_cols = ["id", "full_name", "email", "role", "age", "gender",
                                "diseases", "medications", "allergies", "diet_preference"]
                available_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[available_cols], use_container_width=True, height=400)

                # --- Per-user detail expander ---
                st.markdown("#### 🔍 User Details")
                for user in users_data:
                    if user.get("role") == "admin":
                        continue
                    with st.expander(f"👤 {user.get('full_name', 'Unknown')} — {user.get('email', '')}"):
                        uc1, uc2, uc3, uc4 = st.columns(4)
                        with uc1:
                            st.metric("Age", user.get("age", 0))
                        with uc2:
                            st.metric("Gender", user.get("gender", "—"))
                        with uc3:
                            h = user.get("height_cm", 0)
                            w = user.get("weight_kg", 0)
                            bmi = round(w / ((h / 100) ** 2), 1) if h > 0 and w > 0 else 0
                            st.metric("BMI", bmi)
                        with uc4:
                            st.metric("Diet", user.get("diet_preference", "—"))

                        if user.get("diseases"):
                            st.info(f"🩺 **Diseases:** {user['diseases']}")
                        if user.get("medications"):
                            st.warning(f"💊 **Medications:** {user['medications']}")
                        if user.get("allergies"):
                            st.error(f"⚠️ **Allergies:** {user['allergies']}")
                        if not user.get("diseases") and not user.get("medications") and not user.get("allergies"):
                            st.caption("No medical data on file.")
            else:
                st.info("Click 'Load All Users' to view registered users and their profiles.")

    # ── TAB 2: Medical Reports ──
    with admin_tab2:
        st.markdown("#### 📄 Uploaded Medical Reports")
        st.markdown("Reports uploaded by users during their session are listed below.")

        if st.session_state.get("medical_reports"):
            reports = st.session_state["medical_reports"]
            report_rows = []
            for r in reports:
                report_rows.append({
                    "File Name": r.get("name", "Unknown"),
                    "Size (KB)": round(r.get("size", 0) / 1024, 1),
                    "Type": r.get("type", "Unknown"),
                })
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
        else:
            st.info("No medical reports have been uploaded yet by any user in this session.")

        st.markdown("---")
        st.markdown("##### 📊 Data Source Usage")
        st.caption("Data source search activity is tracked per session. Users can search Wikipedia, arXiv, and PubMed from the Risk Prediction page and Data Sources page.")

        ds_c1, ds_c2, ds_c3 = st.columns(3)
        with ds_c1:
            st.metric("📖 Wikipedia", "Available")
        with ds_c2:
            st.metric("📄 arXiv", "Available")
        with ds_c3:
            st.metric("🏥 PubMed", "Available")

    # ── TAB 3: Dataset Import ──
    with admin_tab3:
        st.markdown("#### 📥 Import Patient Profile Dataset")
        st.info(
            "Keep **medgraphx_300_profiles_dataset.xlsx** in the same folder as app.py. "
            "Make sure the Flask backend is running, then click the button below."
        )

        if st.button("📥 Import 300 Patient Profiles Dataset", key="admin_import_dataset"):
            try:
                response = requests.post(f"{API_BASE}/import_profiles", timeout=60)
                data = safe_json(response)
                if response.status_code == 200:
                    st.success(f"✅ Dataset imported | Imported: {data.get('imported', 0)} | Skipped: {data.get('skipped', 0)}")
                else:
                    st.error(data.get("error", "Import failed."))
            except Exception as e:
                st.error(f"Backend connection error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    with admin_tab4:

        st.markdown("### ⭐ User Feedback")

        if st.button("🔄 Load Feedback"):

            try:
                res = requests.get(
                    "http://127.0.0.1:5000/admin/feedback",
                    headers={
                        "Authorization": f"Bearer {st.session_state.get('token')}"
                    }
                )

                if res.status_code == 200:
                    data = res.json()
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error("Failed to fetch feedback")

            except Exception as e:
                st.error(f"Error: {e}")

    with admin_tab5:

        st.markdown("## 📊 System Analytics Dashboard")

        # ---------------------------
        # FETCH METRICS FROM BACKEND
        # --------------------------
        metrics = {}
        dataset = st.session_state.get("selected_dataset")

        if not dataset:
        
            st.warning("⚠ Please select a dataset from Data Sources first.")
        else:
            try:
                res = requests.post(
                    "http://127.0.0.1:5000/admin/system_metrics",
                    json=dataset,
                    headers={"Authorization": f"Bearer {st.session_state.token}"})
                if res.status_code == 200:
                    metrics = res.json()
                else:
                    st.error("Failed to fetch metrics")
            except Exception as e:
                st.error(f"Error: {e}")
        # ---------------------------
        # SHOW METRICS
        # ---------------------------
        st.markdown("### 🔍 Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("NLP Accuracy", f"{metrics.get('nlp_accuracy', 0)}%")
        col2.metric("Semantic Accuracy", f"{metrics.get('semantic_accuracy', 0)}%")
        col3.metric("Entity Extraction", f"{metrics.get('entity_extraction_accuracy', 0)}%")
        col4.metric("KG Accuracy", f"{metrics.get('knowledge_graph_accuracy', 0)}%")

        st.markdown("---")

        # ---------------------------
        # BAR GRAPH (ACCURACY)
        # ---------------------------
        import plotly.graph_objects as go

        labels = ["NLP", "Semantic", "Entity", "KG"]
        values = [
            metrics.get("nlp_accuracy", 0),
            metrics.get("semantic_accuracy", 0),
            metrics.get("entity_extraction_accuracy", 0),
            metrics.get("knowledge_graph_accuracy", 0)
        ]

        fig = go.Figure(data=[go.Bar(x=labels, y=values)])
        fig.update_layout(
            title="📈 Accuracy Comparison",
            yaxis_title="Accuracy (%)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ---------------------------
        # SYSTEM FLOW GRAPH
        # ---------------------------
        st.markdown("### 🔄 System Workflow Graph")

        import networkx as nx

        G = nx.DiGraph()

        G.add_edges_from([
            ("Data Collection", "Data Extraction"),
            ("Data Extraction", "NLP Processing"),
            ("NLP Processing", "Entity Extraction"),
            ("Entity Extraction", "Knowledge Graph"),
            ("Knowledge Graph", "Risk Analysis"),
            ("Risk Analysis", "Meal Planning")
        ])

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=text,
            textposition="bottom center",
            marker=dict(size=30)
        )

        fig2 = go.Figure(data=[edge_trace, node_trace])
        fig2.update_layout(
            showlegend=False,
            title="System Pipeline Flow",
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)


# ----------------------------
# DATA SOURCES PAGE
# ----------------------------
def data_sources_page():
    st.markdown("""
    <style>
        .stApp {
            background: none !important;
            background-color: #f4f8fb !important; /* A light grayish-blue to match your theme, or use #ffffff for pure white */
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="top-bar">
<div class="top-title">MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)

    nav_spacer, btn_col1, btn_col2 = st.columns([7, 1, 1])
    with btn_col1:
        if st.button("📊 Dashboard", key="nav_login"):
            st.session_state.page = "dashboard"
            st.rerun()
    with btn_col2:
        if st.button("🚪 Logout", key="nav_register"):
            logout_to_login()

    if not st.session_state.token or not st.session_state.user:
        st.warning("Please login first.")
        return

    st.markdown('<div style="padding: 10px 40px 30px 40px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📚 Health Data Sources</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Search and retrieve health, medication, and nutrition information from Wikipedia, arXiv, and medical newsletters.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <span class="badge">👤 {st.session_state.user['full_name']}</span>
        <span class="badge">📧 {st.session_state.user['email']}</span>
        <span class="badge">📚 DATA SOURCES</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    source = st.selectbox("Select Data Source", ["Wikipedia", "arXiv", "PubMed"])
    query = st.text_input("Search Query", placeholder="e.g. Metformin, Diabetes diet, Drug interactions")

    # Dataset selection and workflow integration
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None
    if "temp_data" not in st.session_state:
        st.session_state.temp_data = None

    if st.button("🔍 Fetch Dataset from Backend"):

        if not query.strip():
            st.error("Please enter a search query.")
            return

        with st.spinner(f"Fetching data from {source}..."):

            payload = {
                "source": source.lower(),
                "query": query.strip()
            }

            try:
                response = requests.post(f"{API_BASE}/collect_data", json=payload)
                st.write("STATUS:", response.status_code)
                st.write("RAW RESPONSE:", response.text)
                try:
                    data = response.json()
                except Exception:
                    st.error("Backend did not return JSON")
                    st.text(response.text)
                    return

                if "error" in data:
                    st.error(data["error"])
                    return

                # ✅ TEMP STORE FETCHED DATA
                st.session_state.temp_data = data

                st.success("Data fetched successfully ✅")

            except Exception as e:
                st.error(f"Backend connection failed: {e}")

    # --- Preview temporary fetched data and allow adding to knowledge base ---
    if st.session_state.get("temp_data"):
        data = st.session_state.temp_data

        st.markdown("### 📊 Preview Data")

        if data.get("source") == "wikipedia":
            st.subheader(data.get("entity", ""))
            st.write(data.get("description", ""))

        elif data.get("source") == "arxiv":
            for paper in data.get("papers", []):
                st.markdown(f"#### {paper.get('title', '')}")

        elif data.get("source") == "pubmed":
            for article in data.get("articles", []):
                st.markdown(f"#### {article.get('title', '')}")

        if st.button("➕ Add Dataset to Knowledge Base"):
            st.session_state.selected_dataset = data
            st.success("Dataset added successfully 🚀")

    # Display results if dataset is stored in session
    if st.session_state.get("selected_dataset"):
        data = st.session_state.selected_dataset

        st.markdown("### 📊 Collected Data")

        src = (data.get("source") or "").lower()

        if src == "wikipedia":
            st.subheader(data.get("entity", ""))
            st.write(data.get("description", ""))
            st.markdown(f"[Read more]({data.get('url', '')})")

        elif src == "arxiv":
            for paper in data.get("papers", []):
                st.markdown(f"#### {paper.get('title', '')}")
                st.caption(paper.get("published", ""))
                st.write((paper.get("summary", "") or "")[:300] + "...")
                st.markdown(f"[View Paper]({paper.get('link', '')})")
                st.markdown("---")

        elif src == "pubmed":
            for article in data.get("articles", []):
                st.markdown(f"#### {article.get('title', '')}")
                authors = article.get("authors", [])
                st.caption(f"{', '.join(authors)} — {article.get('journal', '')}")
                st.markdown(f"[View Article]({article.get('link', '')})")
                st.markdown("---")

        with st.expander("🔎 View Raw JSON"):
            st.json(data)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
        .block-container {
            padding-left: 40px !important;
            padding-right: 40px !important;
        }
    </style>
    """, unsafe_allow_html=True)


def _search_wikipedia(query):
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 5,
        "utf8": 1
    }
    try:
        resp = requests.get(api_url, params=params, timeout=10)
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            st.warning("No Wikipedia results found.")
            return
        st.markdown("### 📖 Wikipedia Results")
        for r in results:
            title = r["title"]
            snippet = r["snippet"].replace("<span class=\"searchmatch\">", "**").replace("</span>", "**")
            link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
            st.markdown(f"#### [{title}]({link})")
            st.markdown(snippet, unsafe_allow_html=True)
            st.markdown("---")
    except Exception as e:
        st.error(f"Wikipedia search failed: {e}")


def _search_arxiv(query):
    api_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 5,
        "sortBy": "relevance"
    }
    try:
        resp = requests.get(api_url, params=params, timeout=15)
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            st.warning("No arXiv results found.")
            return
        st.markdown("### 📄 arXiv Research Papers")
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()[:300] + "..."
            link = entry.find("atom:id", ns).text.strip()
            published = entry.find("atom:published", ns).text.strip()[:10]
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)[:3]]
            st.markdown(f"#### [{title}]({link})")
            st.caption(f"{', '.join(authors)} — {published}")
            st.markdown(summary)
            st.markdown("---")
    except Exception as e:
        st.error(f"arXiv search failed: {e}")


def _search_pubmed(query):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    try:
        search_resp = requests.get(search_url, params={
            "db": "pubmed",
            "term": query,
            "retmax": 5,
            "retmode": "json"
        }, timeout=10)
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            st.warning("No PubMed results found.")
            return
        fetch_resp = requests.get(fetch_url, params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }, timeout=10)
        result = fetch_resp.json().get("result", {})
        st.markdown("### 🏥 PubMed / Medical Newsletter Results")
        for pid in ids:
            article = result.get(pid, {})
            if not article:
                continue
            title = article.get("title", "Untitled")
            authors_list = article.get("authors", [])
            authors = ", ".join([a.get("name", "") for a in authors_list[:3]])
            pub_date = article.get("pubdate", "")
            source = article.get("source", "")
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            st.markdown(f"#### [{title}]({link})")
            st.caption(f"{authors} — {source}, {pub_date}")
            st.markdown("---")
    except Exception as e:
        st.error(f"PubMed search failed: {e}")


def show_dashboard_page():
    """Wrapper to prevent admin users from accessing the regular user dashboard."""
    # Not logged in
    if not st.session_state.get("user"):
        st.warning("Please login to view the dashboard.")
        st.session_state.page = "auth"
        st.rerun()

    # Admins should not access the user dashboard
    if st.session_state.user.get("role") == "admin":
        st.error("Admin users do not have access to the user dashboard. Redirecting to Admin Dashboard.")
        st.session_state.page = "admin"
        st.rerun()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url('data:image/png;base64,{DASHBOARD_BANNER_B64}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
""", unsafe_allow_html=True)
    # Regular user -> show dashboard
    show_dashboard()


# ----------------------------
# ROUTER
# ----------------------------
if st.session_state.page == "auth":
    auth_page()
elif st.session_state.page == "profile":
    profile_page()
elif st.session_state.page == "admin":
    admin_page()
elif st.session_state.page == "data_sources":
    data_sources_page()
elif st.session_state.page == "analysis":
    st.markdown("""<div class="top-bar">
<div class="top-title">MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)

    nav_spacer, btn_col0, btn_col1, btn_col2 = st.columns([6, 1, 1, 1])
    with btn_col0:
        if st.button("📊 Dashboard", key="nav_data_source"):
            st.session_state.page = "dashboard"
            st.rerun()
            
    with btn_col1:
        if st.button("👤 Profile", key="nav_login"):
            st.session_state.page = "profile"
            st.rerun()
    with btn_col2:
        if st.button("🚪 Logout", key="nav_register"):
            logout_to_login()

    show_nutrient_page()

elif st.session_state.page == "risk_prediction":
    st.markdown("""<div class="top-bar">
<div class="top-title">MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)

    nav_spacer, btn_col_ds, btn_col0, btn_col1, btn_col2 = st.columns([5, 1, 1, 1, 1])
    with btn_col_ds:
        if st.button("📚 Data Sources", key="nav_data_source"):
            st.session_state.page = "data_sources"
            st.rerun()
    with btn_col0:
        if st.button("📊 Dashboard", key="nav_analysis"):
            st.session_state.page = "dashboard"
            st.rerun()
    with btn_col1:
        if st.button("👤 Profile", key="nav_login"):
            st.session_state.page = "profile"
            st.rerun()
    with btn_col2:
        if st.button("🚪 Logout", key="nav_register"):
            logout_to_login()

    if "show_risk" not in st.session_state:
        st.session_state.show_risk = False

    st.markdown("<div style='height: 44px;'></div>", unsafe_allow_html=True)
    st.markdown("## 🔍 Risk Prediction & Knowledge Graph")
    st.markdown("Enter any food, disease, and medication to analyze interactions using the Rule Engine and XGBoost ML model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        food = st.text_input("🍎 Food", placeholder="e.g. Spinach, Salmon, Rice...")
    with col2:
        disease = st.text_input("🩺 Disease", placeholder="e.g. Diabetes, Heart Disease...")
    with col3:
        medicine = st.text_input("💊 Medication", placeholder="e.g. Warfarin, Metformin...")

    if st.button("🔍 Predict Risk"):
        if not food.strip():
            st.warning("Please enter a food item.")
        else:
            st.session_state.show_risk = True
            st.session_state.risk_food = food.strip()
            st.session_state.risk_disease = disease.strip() if disease.strip() else "Healthy"
            st.session_state.risk_medicine = medicine.strip() if medicine.strip() else "None"

    if st.session_state.get("show_risk", False):
        r_food = st.session_state.get("risk_food", "")
        r_disease = st.session_state.get("risk_disease", "Healthy")
        r_medicine = st.session_state.get("risk_medicine", "None")

        with st.spinner("Fetching nutrients & analyzing risk..."):
            nutrients = get_food_nutrients(r_food)

        if nutrients is None:
            st.error(f"Could not fetch nutrient data for '{r_food}' from USDA database.")
        else:
            st.markdown("### 🧪 Nutrient Analysis")
            ncol1, ncol2, ncol3, ncol4 = st.columns(4)
            with ncol1:
                st.metric("Carbs", f"{nutrients['carbs']}g")
            with ncol2:
                st.metric("Sugar", f"{nutrients['sugar']}g")
            with ncol3:
                st.metric("Protein", f"{nutrients['protein']}g")
            with ncol4:
                st.metric("Fat", f"{nutrients['fat']}g")

            rule_risk, rule_reason = rule_engine(r_disease, r_medicine, nutrients, r_food)

            model, encoder = train_model()
            known_diseases = {c.lower(): c for c in encoder.classes_}
            m6_key = r_disease.strip().lower()

            if m6_key in known_diseases:
                d_encoded = encoder.transform([known_diseases[m6_key]])[0]
                features = [[
                    d_encoded,
                    nutrients["carbs"],
                    nutrients["sugar"],
                    nutrients["protein"],
                    nutrients["fat"]
                ]]
                ml_pred = int(model.predict(features)[0])
            else:
                ml_pred = 0

            if rule_risk is not None:
                final_risk = max(rule_risk, ml_pred)
            else:
                final_risk = ml_pred

            risk_labels = {0: ("Safe ✅", "green"), 1: ("Moderate Risk ⚠️", "orange"), 2: ("High Risk ❌", "red")}
            risk_label, risk_clr = risk_labels.get(final_risk, ("Unknown", "gray"))

            st.markdown("---")
            st.markdown(f"<h3 style='color:{risk_clr};font-weight:bold;'>🔬 Risk Level: {risk_label}</h3>", unsafe_allow_html=True)
            if rule_reason:
                st.info(f"Rule Engine: {rule_reason}")
            st.caption("Prediction based on combined Rule Engine + XGBoost ML model")

            # --- Interactive Knowledge Graph (Plotly) ---
            st.markdown("---")
            st.markdown("### 🧠 Interactive Knowledge Graph")

            

            # Get safe foods for this disease
            safe_foods = get_safe_foods(r_disease)

            def format_graph_label(label: str, width: int = 14) -> str:
                wrapped = textwrap.wrap(label, width=width) or [label]
                return "<br>".join(wrapped[:2])

            # Build node lists
            node_labels = [r_disease, r_medicine, r_food]
            node_text_labels = [format_graph_label(label) for label in node_labels]
            node_colors = ["#7b1fa2", "#1565c0",
                           "#e53935" if final_risk == 2 else "#fdd835" if final_risk == 1 else "#43a047"]
            node_sizes = [40, 35, 35]
            node_symbols = ["circle", "diamond", "square"]

            # Positions: keep the top label inside the chart area to prevent overflow.
            node_x = [0.5, 0.15, 0.85]
            node_y = [0.84, 0.56, 0.56]

            # Add safe food nodes in an arc at the bottom
            num_safe = min(len(safe_foods), 8)
            for i in range(num_safe):
                angle = math.pi * (0.15 + 0.7 * i / max(num_safe - 1, 1))
                sx = 0.5 - 0.45 * math.cos(angle)
                sy = 0.15 - 0.1 * math.sin(angle)
                node_x.append(sx)
                node_y.append(sy)
                node_labels.append(safe_foods[i])
                node_text_labels.append(format_graph_label(safe_foods[i], width=12))
                node_colors.append("#43a047")
                node_sizes.append(25)
                node_symbols.append("circle")

            # Edges: (from_idx, to_idx, label, color, dash)
            edge_label_food = "avoid" if final_risk == 2 else "moderate" if final_risk == 1 else "safe"
            edge_color_food = "#e53935" if final_risk == 2 else "#fdd835" if final_risk == 1 else "#43a047"
            edges = [
                (0, 1, "treated with", "#4a148c", "solid"),
                (0, 2, edge_label_food, edge_color_food, "solid"),
                (1, 2, "interaction", "#1565c0", "dash"),
            ]
            # Safe food edges from disease node
            for i in range(num_safe):
                edges.append((0, 3 + i, "safe", "#43a047", "dot"))

            # Build edge traces
            edge_traces = []
            edge_annotations = []
            for (si, ei, elabel, ecolor, edash) in edges:
                x0, y0 = node_x[si], node_y[si]
                x1, y1 = node_x[ei], node_y[ei]
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=2 if edash != "dot" else 1.2, color=ecolor, dash=edash),
                    hoverinfo="none",
                    showlegend=False
                ))
                # Edge label at midpoint (only for main edges, not safe food edges)
                if si < 3 and ei < 3:
                    edge_annotations.append(dict(
                        x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                        text=f"<b>{elabel}</b>", showarrow=False,
                        font=dict(size=11, color=ecolor),
                        bgcolor="white", borderpad=2
                    ))

            # Node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=node_text_labels,
                textposition=["top center", "bottom center", "bottom center"] + ["bottom center"] * num_safe,
                textfont=dict(size=[14, 13, 13] + [11] * num_safe, color="#0f172a"),
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color="white"),
                    symbol=node_symbols
                ),
                hovertext=[f"🏥 {r_disease}", f"💊 {r_medicine}",
                           f"🍎 {r_food} ({risk_label})"] + [f"✅ {sf} (Safe)" for sf in safe_foods[:num_safe]],
                hoverinfo="text",
                showlegend=False
            )

            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                height=700,
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 1.05]),
                margin=dict(l=30, r=30, t=80, b=40),
                annotations=edge_annotations,
                title=dict(text="Food-Disease-Medicine Interaction Graph", font=dict(size=18, color="#0f4fa8"))
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Safe Food Recommendations ---
            st.markdown("---")
            st.markdown(f"### ✅ Safe Foods for {r_disease}")
            safe_foods_list = get_safe_foods(r_disease)
            if safe_foods_list:
                scols = st.columns(min(len(safe_foods_list), 5))
                for i, sf in enumerate(safe_foods_list):
                    with scols[i % min(len(safe_foods_list), 5)]:
                        st.success(f"🌿 {sf}")
            else:
                st.info("No specific safe food recommendations found for this condition.")

            st.success("Knowledge Graph Generated Successfully ✅")

elif st.session_state.page == "dashboard":
    st.markdown("""<div class="top-bar">
<div class="top-title">MedGraphX knowledge mapping tool</div>
</div>""", unsafe_allow_html=True)
    st.markdown('<div class="top-bar-spacer"></div>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        top: 100px !important;
        left: 0 !important;
        height: calc(100vh - 100px) !important;
        bottom: 0 !important;
        position: fixed !important;
        min-width: 22rem !important;
        width: 22rem !important;
        max-width: 22rem !important;
        transform: translateX(0) !important;
        visibility: visible !important;
        display: block !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 22rem !important;
        width: 22rem !important;
        max-width: 22rem !important;
        transform: translateX(0) !important;
        margin-left: 0 !important;
    }

    [data-testid="stSidebarContent"] {
        width: 22rem !important;
    }

    [data-testid="stAppViewContainer"] {
        margin-left: 22rem !important;
    }

    .st-key-nav_analysis,
    .st-key-nav_risk_prediction,
    .st-key-nav_data_source,
    .st-key-nav_login,
    .st-key-nav_register {
        position: fixed !important;
        top: 28px !important;
        z-index: 99995 !important;
        margin: 0 !important;
    }

    .st-key-nav_analysis { right: 690px !important; }
    .st-key-nav_risk_prediction { right: 540px !important; }
    .st-key-nav_data_source { right: 350px !important; }
    .st-key-nav_login { right: 190px !important; }
    .st-key-nav_register { right: 40px !important; }

    .st-key-nav_analysis button,
    .st-key-nav_risk_prediction button,
    .st-key-nav_data_source button,
    .st-key-nav_login button,
    .st-key-nav_register button {
        width: 130px !important;
        min-width: 130px !important;
        justify-content: center !important;
        padding: 10px 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # We decrease the spacer and increase the button column widths
    # We give the Data Sources column (btn_col0) a ratio of 2.0 to ensure it fits
    nav_spacer, btn_col_a, btn_col_b, btn_col0, btn_col1, btn_col2 = st.columns([6, 2, 2, 2.5, 2, 2])
    with btn_col_a:
        if st.button("📊 Analysis", key="nav_analysis"):
            st.session_state.page = "analysis"
            st.rerun()
    with btn_col_b:
        if st.button("🔍 Risk", key="nav_risk_prediction"):
            st.session_state.page = "risk_prediction"
            st.rerun()
    with btn_col0:
        if st.button("📚 Data Sources", key="nav_data_source"):
            st.session_state.page = "data_sources"
            st.rerun()
    with btn_col1:
        if st.button("👤 Profile", key="nav_login"):
            st.session_state.page = "profile"
            st.rerun()
    with btn_col2:
        if st.button("🚪 Logout", key="nav_register"):
            logout_to_login()

    show_dashboard_page()
