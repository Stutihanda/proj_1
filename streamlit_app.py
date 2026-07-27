import streamlit as st
import requests

st.set_page_config(
    page_title="Climate Guardian AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM (FADED MEDICAL TEAL THEME)
# ============================================================

RISK_META = {
    0: {"label": "LOW", "color": "#059669", "glow": "rgba(5, 150, 105, 0.12)"},
    1: {"label": "MEDIUM", "color": "#D97706", "glow": "rgba(217, 119, 6, 0.12)"},
    2: {"label": "HIGH", "color": "#DC2626", "glow": "rgba(220, 38, 38, 0.12)"},
}

PIPELINE_STAGES = [
    "Climate", "Health", "Social", "Fusion", "Validation",
    "Model Selection", "AutoML", "Explainability", "Decision",
]


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Soft faded medical teal gradient background */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(160deg, #E0F2F1 0%, #EBF8F7 45%, #F0FDFA 100%) !important;
            font-family: 'Inter', sans-serif;
            color: #0F172A;
        }

        [data-testid="stHeader"] { background: transparent; }
        
        [data-testid="stSidebar"] { 
            background-color: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid #CCECE6;
        }

        /* Subtle grid overlay */
        [data-testid="stAppViewContainer"] > .main {
            background-image:
                linear-gradient(rgba(13, 148, 136, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(13, 148, 136, 0.04) 1px, transparent 1px);
            background-size: 40px 40px;
        }

        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        [data-testid="stSidebarCollapsedControl"] { color: #0D9488; }

        h1, h2, h3, .cg-heading {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.01em;
            color: #0F172A;
        }

        /* ---- Header banner ---- */
        .cg-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.1rem 1.5rem;
            border: 1px solid #B2DFDB;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(8px);
            margin-bottom: 1.4rem;
            box-shadow: 0 4px 15px rgba(13, 148, 136, 0.05);
        }
        .cg-header-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
        }
        .cg-header-sub {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #475569;
            margin-top: 0.15rem;
        }
        .cg-status-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            border: 1px solid;
        }
        .cg-status-idle { color: #64748B; border-color: #CBD5E1; background: #F8FAFC; }
        .cg-status-running { color: #D97706; border-color: #FDE68A; background: #FEF3C7; }
        .cg-status-done { color: #059669; border-color: #A7F3D0; background: #D1FAE5; }

        /* ---- Live Phone-style Weather Widget ---- */
        .weather-widget {
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
            color: #FFFFFF;
            border-radius: 16px;
            padding: 1.1rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 20px rgba(13, 148, 136, 0.25);
            position: relative;
            overflow: hidden;
        }
        .weather-location {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            opacity: 0.9;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .weather-badge {
            font-size: 0.65rem;
            background: rgba(255, 255, 255, 0.2);
            padding: 2px 8px;
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
        }
        .weather-main {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0.6rem 0;
        }
        .weather-temp {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1;
        }
        .weather-icon {
            font-size: 2.2rem;
        }
        .weather-desc {
            font-size: 0.82rem;
            font-weight: 500;
            opacity: 0.95;
            margin-bottom: 0.6rem;
        }
        .weather-details {
            display: flex;
            gap: 0.8rem;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            padding-top: 0.55rem;
            font-size: 0.73rem;
            opacity: 0.85;
            font-family: 'JetBrains Mono', monospace;
        }

        /* ---- Pipeline diagram ---- */
        .cg-pipeline-wrap {
            border: 1px solid #B2DFDB;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(8px);
            padding: 1rem 1.2rem 1.1rem 1.2rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 4px 15px rgba(13, 148, 136, 0.04);
        }
        .cg-pipeline-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            color: #0D9488;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .cg-pipeline-row { display: flex; align-items: center; overflow-x: auto; }
        .cg-pipeline-step { display: flex; flex-direction: column; align-items: center; min-width: 92px; flex-shrink: 0; }
        .cg-pipeline-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #0D9488; box-shadow: 0 0 8px rgba(13, 148, 136, 0.4);
            margin-bottom: 6px;
        }
        .cg-pipeline-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem; color: #334155; text-align: center; white-space: nowrap;
        }
        .cg-pipeline-line {
            height: 1px; flex-grow: 1;
            background: repeating-linear-gradient(90deg, #CBD5E1 0 6px, transparent 6px 12px);
            margin: 0 -4px; margin-bottom: 22px;
        }

        /* ---- Upload cards ---- */
        [data-testid="stFileUploader"] {
            border: 1px solid #B2DFDB; border-radius: 12px;
            background: rgba(255, 255, 255, 0.9); padding: 0.9rem;
            box-shadow: 0 2px 8px rgba(13, 148, 136, 0.03);
        }

        /* ---- Buttons ---- */
        .stButton button {
            font-family: 'Space Grotesk', sans-serif; font-weight: 600;
            border-radius: 10px; border: 1px solid #B2DFDB;
            background: #FFFFFF; color: #0F172A;
        }
        .stButton button:hover {
            border-color: #0D9488; color: #0D9488; background: #F0FDFA;
        }
        .stButton button[kind="primary"] { background: #0D9488; color: #FFFFFF; border: none; }
        .stButton button[kind="primary"]:hover { background: #0F766E; color: #FFFFFF; }

        /* ---- KPI stat cards ---- */
        .cg-kpi-row { display: flex; gap: 0.9rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
        .cg-kpi-card {
            flex: 1; min-width: 190px; border: 1px solid #B2DFDB;
            border-radius: 12px; background: rgba(255, 255, 255, 0.9); padding: 0.9rem 1.1rem;
            box-shadow: 0 2px 8px rgba(13, 148, 136, 0.04);
        }
        .cg-kpi-label {
            font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
            letter-spacing: 0.1em; text-transform: uppercase; color: #0D9488; font-weight: 600; margin-bottom: 0.35rem;
        }
        .cg-kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #0F172A; }
        .cg-kpi-note { font-size: 0.75rem; color: #64748B; margin-top: 0.2rem; }

        /* ---- Section eyebrow ---- */
        .cg-eyebrow {
            font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
            letter-spacing: 0.12em; text-transform: uppercase; color: #0D9488; margin-bottom: 0.3rem;
            font-weight: 600;
        }

        /* ---- Region risk cards ---- */
        .cg-region-card {
            border: 1px solid #B2DFDB; border-left: 4px solid var(--risk-color);
            border-radius: 10px; background: rgba(255, 255, 255, 0.9); padding: 0.85rem 1rem; margin-bottom: 0.6rem;
            box-shadow: 0 2px 8px rgba(13, 148, 136, 0.04);
        }
        .cg-region-top { display: flex; justify-content: space-between; align-items: center; }
        .cg-region-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem; color: #0F172A; }
        .cg-risk-badge {
            font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600;
            letter-spacing: 0.06em; padding: 0.22rem 0.6rem; border-radius: 999px;
            color: var(--risk-color); background: var(--risk-glow); border: 1px solid var(--risk-color);
        }
        .cg-region-actions { margin-top: 0.55rem; font-size: 0.86rem; color: #334155; padding-left: 1.1rem; }
        .cg-region-actions li { margin-bottom: 0.2rem; }

        /* ---- Feature importance bars ---- */
        .cg-feat-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.45rem; }
        .cg-feat-name { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #334155; width: 170px; flex-shrink: 0; text-align: right; }
        .cg-feat-bar-track { flex-grow: 1; background: #CBD5E1; border-radius: 6px; height: 10px; overflow: hidden; }
        .cg-feat-bar-fill { height: 100%; background: linear-gradient(90deg, #14B8A6, #0D9488); border-radius: 6px; }
        .cg-feat-val { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748B; width: 48px; }

        /* ---- Ask AI console ---- */
        .cg-ask-panel {
            border: 1px solid #B2DFDB; border-radius: 14px;
            background: rgba(255, 255, 255, 0.9); padding: 1.2rem 1.3rem;
            box-shadow: 0 4px 15px rgba(13, 148, 136, 0.05);
        }
        .cg-chip button {
            font-family: 'JetBrains Mono', monospace !important; font-size: 0.76rem !important;
            font-weight: 400 !important; background: #F0FDFA !important; border: 1px solid #B2DFDB !important;
            color: #0F766E !important; border-radius: 999px !important; padding: 0.3rem 0.85rem !important;
        }
        .cg-chip button:hover { border-color: #0D9488 !important; color: #0D9488 !important; background: #E0F2F1 !important; }
        .cg-answer-card {
            border: 1px solid #A7F3D0; border-left: 4px solid #059669; border-radius: 10px;
            background: #D1FAE5; padding: 0.9rem 1.1rem; margin-top: 0.9rem;
            font-size: 0.92rem; line-height: 1.55; color: #064E3B;
        }
        [data-testid="stTextInput"] input {
            background: #FFFFFF !important; border: 1px solid #B2DFDB !important;
            color: #0F172A !important; font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #0D9488 !important;
            box-shadow: 0 0 0 1px #0D9488 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LIVE WEATHER WIDGET (OPEN-METEO API + IP GEOLOCATION)
# ============================================================

@st.cache_data(ttl=3600)
def get_location_by_ip():
    """Attempts to find the user's location via IP address."""
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data.get("lat"), data.get("lon"), data.get("city"), data.get("countryCode")
    except Exception:
        pass
    # Fallback coordinates if the API fails
    return 13.0827, 80.2707, "Chennai", "IN"


@st.cache_data(ttl=600)
def get_weather_data(lat, lon):
    """Fetches real-time weather using free Open-Meteo API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json().get("current", {})
    except Exception:
        return None
    return None


def get_weather_info(code):
    """Maps Open-Meteo weather code to condition name and icon."""
    if code == 0:
        return "Clear Sky", "☀️"
    elif code in [1, 2, 3]:
        return "Partly Cloudy", "⛅"
    elif code in [45, 48]:
        return "Foggy", "🌫️"
    elif code in [51, 53, 55, 61, 63, 65]:
        return "Rain / Drizzle", "🌧️"
    elif code in [80, 81, 82]:
        return "Heavy Showers", "🌦️"
    elif code in [95, 96, 99]:
        return "Thunderstorm", "⛈️"
    return "Overcast", "☁️"


def render_weather_widget():
    # 1. Grab location dynamically
    lat, lon, city, country = get_location_by_ip()
    
    # 2. Fetch weather for that location
    weather = get_weather_data(lat, lon)
    
    if weather:
        temp = round(weather.get("temperature_2m", 0))
        humidity = weather.get("relative_humidity_2m", 0)
        wind = weather.get("wind_speed_10m", 0)
        code = weather.get("weather_code", 0)
        desc, icon = get_weather_info(code)
    else:
        # Fallback offline values
        temp, humidity, wind, desc, icon = 32, 78, 12.5, "Partly Humid", "⛅"

    st.markdown(
        f"""
        <div class="weather-widget">
            <div class="weather-location">
                <span>📍 {city}, {country}</span>
                <span class="weather-badge">LIVE</span>
            </div>
            <div class="weather-main">
                <div class="weather-temp">{temp}°C</div>
                <div class="weather-icon">{icon}</div>
            </div>
            <div class="weather-desc">{desc}</div>
            <div class="weather-details">
                <span>💧 {humidity}% Humidity</span>
                <span>💨 {wind} km/h</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# COMPONENT RENDERERS
# ============================================================

def render_header(step):
    status_map = {
        1: ("cg-status-idle", "AWAITING DATA"),
        2: ("cg-status-running", "PIPELINE RUNNING"),
        3: ("cg-status-done", "ANALYSIS COMPLETE"),
        4: ("cg-status-done", "AI ANALYST ACTIVE"),
    }
    cls, text = status_map.get(step, status_map[1])
    st.markdown(
        f"""
        <div class="cg-header">
            <div>
                <p class="cg-header-title">⚕️ Climate Guardian AI</p>
                <p class="cg-header-sub">Step {step} of 4 &nbsp;—&nbsp; Dengue outbreak risk fusion</p>
            </div>
            <div class="cg-status-pill {cls}">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_diagram():
    steps_html = ""
    for i, stage in enumerate(PIPELINE_STAGES):
        steps_html += f"""
        <div class="cg-pipeline-step">
            <div class="cg-pipeline-dot"></div>
            <div class="cg-pipeline-name">{stage}</div>
        </div>
        """
        if i < len(PIPELINE_STAGES) - 1:
            steps_html += '<div class="cg-pipeline-line"></div>'

    st.markdown(
        f"""
        <div class="cg-pipeline-wrap">
            <div class="cg-pipeline-label">Live Agent Pipeline</div>
            <div class="cg-pipeline-row">{steps_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(results):
    model_info = results.get("model_comparison", {})
    region_risk = results.get("region_risk_table", {})
    validation = results.get("validation", {})

    n_regions = len(region_risk)
    high_risk_regions = [r for r, v in region_risk.items() if v == 2]

    cards = [
        ("BEST MODEL", model_info.get("best_name", "—"), f"{model_info.get('best_accuracy', 0):.2%} accuracy" if model_info.get("best_accuracy") is not None else ""),
        ("REGIONS ANALYZED", str(n_regions), ""),
        ("HIGH RISK REGIONS", str(len(high_risk_regions)), ", ".join(high_risk_regions[:3]) if high_risk_regions else "none flagged"),
        ("DUPLICATE ROWS REMOVED", str(validation.get("duplicate_rows", 0)), ""),
    ]

    html = '<div class="cg-kpi-row">\n'
    for label, value, note in cards:
        html += '<div class="cg-kpi-card">\n'
        html += f'<div class="cg-kpi-label">{label}</div>\n'
        html += f'<div class="cg-kpi-value">{value}</div>\n'
        html += f'<div class="cg-kpi-note">{note}</div>\n'
        html += '</div>\n'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)


def render_region_cards(decisions, region_risk):
    for region, decision in decisions.items():
        risk_level = decision.get("risk_level", region_risk.get(region, 0))
        meta = RISK_META.get(risk_level, RISK_META[0])
        actions_html = "".join(f"<li>{a}</li>" for a in decision.get("recommendations", []))
        st.markdown(
            f"""
            <div class="cg-region-card" style="--risk-color:{meta['color']}; --risk-glow:{meta['glow']};">
                <div class="cg-region-top">
                    <div class="cg-region-name">{region}</div>
                    <div class="cg-risk-badge">{meta['label']} RISK</div>
                </div>
                <ul class="cg-region-actions">{actions_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_bars(top_features):
    if not top_features:
        return
    max_val = max(f["importance"] for f in top_features) or 1
    html = ""
    for f in top_features:
        pct = round((f["importance"] / max_val) * 100, 1)
        html += f"""
        <div class="cg-feat-row">
            <div class="cg-feat-name">{f['feature']}</div>
            <div class="cg-feat-bar-track"><div class="cg-feat-bar-fill" style="width:{pct}%;"></div></div>
            <div class="cg-feat-val">{f['importance']}</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# STATE MANAGEMENT & MAIN APP
# ============================================================

inject_css()

if "step" not in st.session_state:
    st.session_state.step = 1

with st.sidebar:
    # Smartphone weather widget displayed at the top of sidebar
    render_weather_widget()

    st.markdown("**Connection Settings**")
    API_URL = st.text_input(
        "FastAPI Server URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        help="The local or remote URL of your FastAPI backend.",
    )
    st.session_state["api_url"] = API_URL
    
    if st.session_state.step > 1:
        st.markdown("---")
        if st.button("Start Over", use_container_width=True):
            st.session_state.clear()
            st.rerun()

render_header(st.session_state.step)


# ------------------------------------------------------------
# WINDOW 1: INGESTION
# ------------------------------------------------------------
if st.session_state.step == 1:
    st.markdown('<div class="cg-eyebrow">01 · Data Ingestion</div>', unsafe_allow_html=True)
    st.write("Upload your segmented operational datasets below. The pipeline requires all three views to perform matrix fusion.")

    col1, col2, col3 = st.columns(3)
    with col1:
        climate_file = st.file_uploader("Climate Dataset", type="csv")
    with col2:
        health_file = st.file_uploader("Health Dataset", type="csv")
    with col3:
        social_file = st.file_uploader("Social Dataset", type="csv")

    st.write("")
    if st.button("Initialize Pipeline & Fusion", type="primary", use_container_width=True):
        if not (climate_file and health_file and social_file):
            st.warning("Upload all three datasets before running.")
        else:
            st.session_state.climate_data = climate_file.getvalue()
            st.session_state.health_data = health_file.getvalue()
            st.session_state.social_data = social_file.getvalue()
            
            st.session_state.step = 2
            st.rerun()


# ------------------------------------------------------------
# WINDOW 2: PIPELINE EXECUTION
# ------------------------------------------------------------
elif st.session_state.step == 2:
    st.markdown('<div class="cg-eyebrow">02 · Pipeline Execution</div>', unsafe_allow_html=True)
    render_pipeline_diagram()
    
    with st.spinner("Pipeline is actively running. Agents are validating, training, and building explanations..."):
        try:
            files = {
                "climate_file": ("climate.csv", st.session_state.climate_data, "text/csv"),
                "health_file": ("health.csv", st.session_state.health_data, "text/csv"),
                "social_file": ("social.csv", st.session_state.social_data, "text/csv"),
            }
            
            response = requests.post(f"{API_URL}/run-pipeline", files=files, timeout=600)
            response.raise_for_status()
            payload = response.json()
            
            if payload.get("status") == "success":
                st.session_state.pipeline_result = payload
                st.session_state.step = 3
                st.rerun()
            else:
                st.error(payload.get("message", "Pipeline error."))
                if st.button("Go Back"):
                    st.session_state.step = 1
                    st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Could not reach the backend API. Confirm FastAPI is running and the URL in the sidebar is correct.")
            if st.button("Go Back"):
                st.session_state.step = 1
                st.rerun()
        except Exception as e:
            st.error(f"Pipeline request failed: {e}")
            if st.button("Go Back"):
                st.session_state.step = 1
                st.rerun()


# ------------------------------------------------------------
# WINDOW 3: ANALYTICAL RESULTS
# ------------------------------------------------------------
elif st.session_state.step == 3:
    payload = st.session_state.pipeline_result
    results = payload.get("results", {})

    st.markdown('<div class="cg-eyebrow">03 · Risk Assessment</div>', unsafe_allow_html=True)
    render_kpi_row(results)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Predicted risk by region**")
        decisions = results.get("decisions", {})
        region_risk = results.get("region_risk_table", {})
        if decisions:
            render_region_cards(decisions, region_risk)
        else:
            st.info("No region-level predictions returned.")

    with right:
        st.markdown("**Model comparison**")
        model_info = results.get("model_comparison", {})
        if model_info.get("scores"):
            st.bar_chart(model_info["scores"])

        st.markdown("**Most important features**")
        render_feature_bars(results.get("top_features", []))

    shap_path = results.get("shap_image_path")
    if shap_path:
        st.markdown('<div class="cg-eyebrow" style="margin-top:1.4rem;">SHAP Global Explainer</div>', unsafe_allow_html=True)
        try:
            st.image(shap_path, caption="SHAP feature impact across all predictions")
        except Exception:
            st.info("SHAP image not accessible from this machine.")
            
    st.write("")
    if st.button("Proceed to AI Analyst", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()


# ------------------------------------------------------------
# WINDOW 4: ASK AI CHATBOT
# ------------------------------------------------------------
elif st.session_state.step == 4:
    st.markdown('<div class="cg-eyebrow">04 · Ask the Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="cg-ask-panel">', unsafe_allow_html=True)

    st.markdown("Ask a question about the last analysis — answered by the AI agent, grounded in your actual results.")

    example_questions = [
        "Which city has the highest outbreak risk?",
        "Explain why Chennai is High Risk.",
        "What are the important features?",
        "Give recommendations.",
    ]
    chip_cols = st.columns(len(example_questions))

    def set_question(q):
        st.session_state["question_input"] = q
        st.session_state["auto_ask"] = True

    for i, q in enumerate(example_questions):
        with chip_cols[i]:
            st.markdown('<div class="cg-chip">', unsafe_allow_html=True)
            st.button(q, key=f"chip_{i}", on_click=set_question, args=(q,), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    question = st.text_input("Your question", key="question_input", label_visibility="collapsed", placeholder="Type your question here...")
    ask_cols = st.columns([1, 5])
    
    with ask_cols[0]:
        ask_clicked = st.button("Submit Inquiry", type="primary", use_container_width=True)
    with ask_cols[1]:
        if st.button("Return to Results Dashboard"):
            st.session_state.step = 3
            st.rerun()

    should_ask = ask_clicked or st.session_state.pop("auto_ask", False)

    if should_ask and question.strip():
        with st.spinner("Consulting the model..."):
            try:
                response = requests.post(f"{API_URL}/ask", params={"question": question}, timeout=120)
                response.raise_for_status()
                answer_payload = response.json()

                if answer_payload.get("status") == "success":
                    st.markdown(f'<div class="cg-answer-card">{answer_payload.get("answer")}</div>', unsafe_allow_html=True)
                else:
                    st.error(answer_payload.get("message", "The analyst couldn't answer that."))
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend API.")
            except Exception as e:
                st.error(f"Request failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
