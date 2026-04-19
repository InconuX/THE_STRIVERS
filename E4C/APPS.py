import streamlit as st
import time
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# CONFIGURATION & PROFESSIONAL CSS
# ==========================================
st.set_page_config(page_title="THE STRIVERS | Industrial AI", layout="wide", initial_sidebar_state="collapsed")

PRIMARY_COLOR = "#00d4ff"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Roboto:wght@400;500&display=swap');
    
    .strivers-header {{ 
        background: linear-gradient(135deg, #0a0f18 0%, #1a2639 100%); 
        padding: 25px; border-radius: 10px; color: white; 
        display: flex; justify-content: space-between; align-items: center; 
        border-left: 6px solid {PRIMARY_COLOR}; margin-bottom: 30px; 
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.15);
    }}
    .logo-text {{ font-family: 'Orbitron', sans-serif; font-size: 34px; font-weight: bold; color: {PRIMARY_COLOR}; letter-spacing: 2px; text-shadow: 0 0 10px {PRIMARY_COLOR};}}
    .system-label {{ font-family: 'Roboto', sans-serif; font-size: 13px; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px;}}
    
    .metric-card {{ 
        background: rgba(30, 34, 45, 0.7); padding: 18px; border-radius: 10px; 
        border: 1px solid #2d3748; text-align: center; backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
    }}
    .val-text {{ font-family: 'Orbitron', sans-serif; font-size: 28px; font-weight: bold; color: {PRIMARY_COLOR}; }}
    .metric-label {{ color: #a0aec0; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;}}
    
    .spacer-medium {{ margin-top: 40px; margin-bottom: 40px; }}
    .spacer-small {{ margin-top: 20px; margin-bottom: 20px; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# AI MODEL INTEGRATION & CACHING
# ==========================================
@st.cache_resource
def load_ai_model():
    """Entraîne et met en cache le modèle ML une seule fois au lancement de l'application."""
    np.random.seed(42)
    X_normal = np.random.normal(loc=[800, 8.0, 35, 1.5, 50, 30], scale=[10, 0.2, 1, 0.1, 1, 1], size=(100, 6))
    X_dry = np.random.normal(loc=[800, 5.5, 65, 2.5, 5, 10], scale=[10, 0.3, 3, 0.3, 1, 2], size=(100, 6))
    X_sand = np.random.normal(loc=[800, 11.5, 52, 3.5, 15, 45], scale=[10, 0.4, 2, 0.4, 2, 2], size=(100, 6))
    X_cav = np.random.normal(loc=[800, 8.1, 40, 9.5, 40, 25], scale=[10, 0.2, 1, 1.0, 3, 2], size=(100, 6))

    X_train = np.vstack([X_normal, X_dry, X_sand, X_cav])
    y_train = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    return model

# Chargement du modèle depuis le cache
model = load_ai_model()

def get_prediction(ir, cu, te, vi, fl, he):
    """Logique native remplaçant l'ancienne API FastAPI"""
    raw = np.array([[ir, cu, te, vi, fl, he]])
    pred = int(model.predict(raw)[0])
    conf = float(max(model.predict_proba(raw)[0])) * 100
    
    diag = {
        0: {"status": "Healthy", "issue": "Normal Operation", "health": 100, "exp": "Telemetry signature is optimal. Current matches RainMaker 2 baseline (~8.0A).", "sop": "Maintain standard monthly visual inspection."},
        1: {"status": "Critical", "issue": "Dry Running", "health": int(100 - conf*0.8), "exp": f"Current ({cu:.2f}A) is abnormally low indicating loss of prime. Stator overheating risk.", "sop": "IMMEDIATE SHUTDOWN. Verify static well water level and low-water probe calibration."},
        2: {"status": "Critical", "issue": "Sand Clogging", "health": int(100 - conf*0.9), "exp": f"Mechanical overload detected ({cu:.2f}A > 8A nominal baseline). High impeller friction due to sand drag.", "sop": "Extract pump from borehole. Clean intake strainers and inspect impellers for abrasive wear."},
        3: {"status": "Critical", "issue": "Severe Cavitation", "health": int(100 - conf*0.7), "exp": f"Extreme vibrations ({vi:.2f}mm/s) exceeding ISO 10816 limits. Net Positive Suction Head (NPSH) critically low.", "sop": "Reduce MPPT controller frequency to stabilize fluid velocity and check pump submergence depth."}
    }[pred]

    # XAI Calculation
    feature_names = ["Irradiance", "Current", "Temperature", "Vibration", "Flow", "Pressure"]
    ref = np.array([800, 8.0, 35, 1.5, 50, 30])
    weights = np.abs(raw[0] - ref) / (ref + 0.1)
    xai_dict = dict(zip(feature_names, (weights / np.sum(weights)).tolist()))
    
    return {**diag, "xai": xai_dict, "confidence": f"{conf:.1f}%"}

# ==========================================
# DATA PERSISTENCE
# ==========================================
HIST_FILE = "strivers_data.csv"
def load_data():
    if os.path.exists(HIST_FILE): return pd.read_csv(HIST_FILE).to_dict('records')
    return []

if 'history' not in st.session_state: st.session_state.history = load_data()
if 'refresh_rate' not in st.session_state: st.session_state.refresh_rate = 2
if 'chat_history' not in st.session_state: st.session_state.chat_history = [
    {"role": "assistant", "content": "Welcome! I am the STRIVERS AI Assistant. Ask me anything about E4C pump specifications or system diagnostics."}
]

# ==========================================
# POPUP DIALOGS (MODALS)
# ==========================================
@st.dialog("Detailed Audit Report", width="large")
def show_audit_report(log):
    st.write(f"**Timestamp:** {log['Time']}")
    st.write(f"**Status:** {log['Issue']} | **System Health:** {log['Health']}%")
    st.divider()
    st.markdown("#### AI Diagnostic Inference")
    st.info(log['Report'])
    st.markdown("#### Maintenance Standard Operating Procedure (SOP)")
    st.warning(log['SOP'])
    st.divider()
    st.write("**Technical Snapshot Recorded:**")
    st.json({"Current_A": log.get('Cur', 0), "Vibration_mm_s": log.get('Vib', 0), "Health_Score": log['Health']})
    
    csv_data = f"STRIVERS REPORT\nTime:{log['Time']}\nStatus:{log['Issue']}\nAction:{log['SOP']}"
    st.download_button(label="Download Technical Data (CSV)", data=csv_data, file_name=f"report_{log['Time'].replace(':','')}.csv", use_container_width=True)

# ==========================================
# HEADER
# ==========================================
st.markdown(f"""
    <div class="strivers-header">
        <div>
            <div class="logo-text">THE STRIVERS</div>
            <div class="system-label">E4C Smart Community Solar Pump Monitoring System</div>
        </div>
        <div style="text-align: right; font-family: 'Orbitron';">
            <div style="font-size: 20px; color: #a0aec0;">SYSTEM TIME</div>
            <div style="font-size: 16px;">{datetime.now().strftime('%H:%M:%S UTC')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### MISSION CONTROL")
is_live = st.sidebar.toggle("Live Telemetry Feed", value=True, help="Disable to freeze data for log analysis.")
auto_mode = st.sidebar.toggle("Auto-Showroom Mode", value=True)

if not auto_mode:
    manual_mode = st.sidebar.radio("Force Inject Fault:", ["Normal", "Dry Run", "Sand Clogging", "Cavitation"])
else:
    st.sidebar.info(" Auto-Showroom Active. Rotating scenarios automatically.")

def generate_telemetry(mode):
    hour = datetime.now().hour + datetime.now().minute/60
    ir = max(0, 950 * np.sin(np.pi * (hour - 6) / 12)) + np.random.normal(0, 5)
    if mode == "Normal": cu, te, vi, fl, he = 8.0, 35.0, 1.4, 52.0, 30.0
    elif mode == "Dry Run": cu, te, vi, fl, he = 5.2, 67.0, 2.2, 5.0, 10.0
    elif mode == "Sand Clogging": cu, te, vi, fl, he = 11.7, 51.0, 3.4, 15.0, 46.0
    else: cu, te, vi, fl, he = 8.1, 39.0, 9.7, 38.0, 24.0

    cu+=np.random.normal(0,0.1); te+=np.random.normal(0,0.3); vi+=np.random.normal(0,0.1); fl+=np.random.normal(0,0.5)
    return ir, cu, te, vi, fl, he

# ==========================================
# MAIN TABS
# ==========================================
tab_live, tab_hist, tab_asset, tab_chat, tab_settings = st.tabs([
    "🔴 LIVE DASHBOARD", " AUDIT LOG", " TECHNICAL SPECS (E4C)", " AI ASSISTANT", " SETTINGS"
])

# ==========================================
# FLICKER-FREE LIVE DASHBOARD (Using st.fragment)
# ==========================================
@st.fragment(run_every=st.session_state.refresh_rate if is_live else None)
def render_live_dashboard():
    if not auto_mode:
        mode = manual_mode
        st.caption(f" **Manual Override Active:** Injecting {mode.upper()} fault data.")
    else:
        modes = ["Normal", "Normal", "Dry Run", "Normal", "Sand Clogging", "Normal", "Cavitation"]
        mode = modes[int((time.time() // 8) % len(modes))]
        st.caption(f" **Auto-Sequence Active:** Currently testing {mode.upper()} scenario.")

    ir, cu, te, vi, fl, he = generate_telemetry(mode)

    cols = st.columns(6)
    labels = ["Irradiance (W)", "Current (A)", "Motor Temp (C)", "Flow Rate (L)", "Head Press. (m)", "Vibration (mm/s)"]
    vals = [f"{ir:.0f}", f"{cu:.2f}", f"{te:.1f}", f"{fl:.1f}", f"{he:.1f}", f"{vi:.2f}"]
    for i in range(6):
        cols[i].markdown(f'<div class="metric-card"><div class="metric-label">{labels[i]}</div><div class="val-text">{vals[i]}</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)

    # DIRECT FUNCTION CALL (Replaces requests.post API call)
    res = get_prediction(ir, cu, te, vi, fl, he)
    
    col_g1, col_g2, col_xai = st.columns([1, 1, 1.5])
    
    with col_g1:
        st.markdown("#### Load Current")
        fig_c = go.Figure(go.Indicator(mode="gauge+number", value=cu, gauge={'axis': {'range': [0, 15]}, 'bar': {'color': PRIMARY_COLOR}}))
        fig_c.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_c, use_container_width=True)

    with col_g2:
        st.markdown("#### Vibration Profile")
        fig_v = go.Figure(go.Indicator(mode="gauge+number", value=vi, gauge={'axis': {'range': [0, 12]}, 'bar': {'color': PRIMARY_COLOR}}))
        fig_v.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_v, use_container_width=True)

    with col_xai:
        st.markdown("#### AI Influence Factors")
        fig_xai = px.bar(x=list(res['xai'].values()), y=list(res['xai'].keys()), orientation='h', template="plotly_dark")
        fig_xai.update_traces(marker_color=PRIMARY_COLOR)
        fig_xai.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Factor Weight", yaxis_title="")
        st.plotly_chart(fig_xai, use_container_width=True)

    st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
    
    st.markdown("#### Global System Health")
    if res['status'] == "Healthy": 
        st.success(f" STATUS: {res['issue'].upper()} | HEALTH SCORE: {res['health']}%")
    else: 
        st.error(f" STATUS: {res['issue'].upper()} | HEALTH SCORE: {res['health']}%")
    
    st.info(f"**AI Diagnostic:** {res['exp']}")
    st.warning(f"**Field Operator Recommendation (SOP):** {res['sop']}")

    if is_live:
        new_row = {"Time": datetime.now().strftime("%H:%M:%S"), "Issue": res['issue'], "Health": res['health'], "Report": res['exp'], "SOP": res['sop'], "Cur": round(cu,2), "Vib": round(vi,2)}
        st.session_state.history.insert(0, new_row)
        if len(st.session_state.history) > 100: st.session_state.history.pop()
        pd.DataFrame(st.session_state.history).to_csv(HIST_FILE, index=False)
        
        hist_df = pd.DataFrame(st.session_state.history)
        if not hist_df.empty:
            st.markdown("#### Real-Time Health Trend")
            fig_trend = px.line(hist_df, x='Time', y='Health', template="plotly_dark", markers=True)
            fig_trend.update_traces(line_color=PRIMARY_COLOR)
            st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")


with tab_live:
    render_live_dashboard()

with tab_hist:
    st.markdown("### Historic Audit Journal")
    if not is_live: st.info("Feed paused for analysis.")
    else: st.warning("Live feed active. Turn off in sidebar to freeze view.")
    
    if st.session_state.history:
        for i, log in enumerate(st.session_state.history[:20]):
            col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
            col1.write(f"**{log['Time']}**")
            col2.write(log['Issue'])
            col3.write(f"Health: {log['Health']}%")
            if col4.button("Deep View", key=f"btn_{i}"):
                show_audit_report(log)
            st.markdown("---")

with tab_asset:
    st.markdown("### Technical Specifications (Sourced from E4C Solutions Library)")
    st.info("**Engineering Note:** The E4C library provides excellent product baselines, but lacks detailed fault signatures (e.g., vibration alarms, exact current for clogging). STRIVERS AI bridges this gap.")
    
    c_t1, c_t2, c_t3 = st.columns(3)
    with c_t1:
        st.markdown(f"####  RainMaker 2 (Primary Baseline)")
        st.markdown("- **Type:** Solar-battery-powered submersible")
        st.markdown("- **Performance:** 50 L/min at 70m depth")
        st.markdown(f"- **Max Current:** <span style='color:{PRIMARY_COLOR}; font-weight:bold;'>8.0 A</span>", unsafe_allow_html=True)
        st.markdown("- **Voltage Range:** 24–34 V")
    
    with c_t2:
        st.markdown("####  Shurflo 9300 Series")
        st.markdown("- **Type:** Solar submersible (low-flow)")
        st.markdown("- **Performance:** 82 gal/h at 230 ft")
        st.markdown("- **Min Solar Array:** 155 W")
        st.markdown("- **Voltage:** 24 VDC")

    with c_t3:
        st.markdown("####  Pumpmakers DIY & SDS-T-128")
        st.markdown("- **Pumpmakers:** Open-source, surface crankshaft, up to 100m depth, 18,000 L/day.")
        st.markdown("- **SDS-T-128:** Low-cost positive displacement, 12-30 V DC, 4-inch well casing.")
        st.markdown("- **GF1:** 12/24 V DC for remote irrigation.")

    st.divider()
    st.markdown("###  AI Diagnostic Thresholds (Strivers Predictive Layer)")
    st.markdown(f"""
    - **Nominal Current (FLA):** 8.0 A (RainMaker 2 Spec)
    - **Sand Clogging Alert:** > 11.0 A *(High impeller friction detected)*
    - **Dry Run Alert:** < 6.0 A *(Loss of fluid load on motor)*
    - **Critical Vibration:** > 7.1 mm/s *(ISO 10816 Standard Compliance)*
    """)

with tab_chat:
    st.markdown("###  STRIVERS Field AI Assistant")
    st.caption("Ask questions about E4C pump specifications, diagnostic thresholds, or SOPs.")
    
    chat_container = st.container(height=400)
    
    for msg in st.session_state.chat_history:
        chat_container.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("E.g., What is the baseline current for the RainMaker 2?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        chat_container.chat_message("user").write(prompt)
        
        prompt_lower = prompt.lower()
        if "rainmaker" in prompt_lower:
            reply = "According to the E4C library, the RainMaker 2 pumps from a 70m depth at 50 L/min, with a maximum motor current of 8 A and a voltage range of 24–34 V."
        elif "shurflo" in prompt_lower:
            reply = "The E4C report states the Shurflo 9300 Series is for low-flow systems, delivering 82 gallons per hour at 230 feet using 24 VDC and a minimum 155 W solar array."
        elif "clogging" in prompt_lower or "sand" in prompt_lower:
            reply = "E4C doesn't provide specific fault signatures for sand clogging. However, the STRIVERS AI detects this when the current exceeds the 8.0A nominal baseline (e.g., > 11.0A), indicating high impeller friction."
        elif "dry run" in prompt_lower:
            reply = "Dry running is detected when the motor load drops abnormally low (e.g., < 6.0A compared to the 8.0A baseline). Immediate shutdown is recommended to prevent stator overheating."
        else:
            reply = "That's a great question. Based on the E4C data, I have specs for RainMaker 2, Shurflo 9300, SDS-T-128, GF1, and Pumpmakers DIY. I also track STRIVERS telemetry. What specific system are you inquiring about?"
            
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        chat_container.chat_message("assistant").write(reply)

with tab_settings:
    st.markdown("### System Configuration")
    st.session_state.refresh_rate = st.slider("Telemetry Refresh Rate (seconds)", 1, 10, 2)
    st.text_input("Admin Alert Gateway", "monitoring@strivers.org")

# FOOTER
st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #444; font-size: 11px; padding: 20px; border-top: 1px solid #222;">
        COPYRIGHT © 2026 THE STRIVERS. E4C HACKATHON ASSET MONITORING. ALL RIGHTS RESERVED.
    </div>
    """, unsafe_allow_html=True)
