"""
app.py ─── Gabès Regenerate AI ─── v5.2
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math, os, sys, json
from openai import OpenAI
import folium
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(__file__))
from config import (
    CD_SAFE_THRESHOLD, CD_EU_LIMIT, EC_SALINITY_SEVERE,
    PB_MODERATE_RISK, PB_HIGH_RISK,
    EC_SALINITY_STRESS, INFILTRATION_FACTORS,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, PH_ACIDIC_SEVERE, PH_ACIDIC_STRESS, PH_ALKALINE_SEVERE, PH_ALKALINE_STRESS,
    PRESCRIPTION_SYSTEM_PROMPT, PRESCRIPTION_USER_PROMPT,
    CHATBOT_SYSTEM_PROMPT,
)
from agents.prediction_agent import get_agent
from agents.water_engine import get_water_engine
from auth import is_logged_in, current_user, logout, show_auth_page

try:
    from utils.passport_generator import generate_qr_bytes, generate_passport_pdf
    PASSPORT_OK = True
except ImportError:
    PASSPORT_OK = False

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gabes Regenerate AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── AUTH GATE ────────────────────────────────────────────────────────────────
if not is_logged_in():
    show_auth_page()
    st.stop()

user     = current_user()
initials = "".join(w[0].upper() for w in user.get("name", "U").split()[:2])

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --g900:#1B5E20; --g800:#2E7D32; --g700:#388E3C; --g200:#C8E6C9;
    --g100:#E8F5E9; --g50:#F4FAF4;
    --amber:#E65100; --amber-l:#FFF3E0; --amber-b:#FFCC80;
    --red:#B71C1C;   --red-l:#FFEBEE;   --red-b:#EF9A9A;
    --blue:#1565C0;  --blue-l:#E3F2FD;
    --text:#111827;  --muted:#6B7280;
    --border:#E5E7EB; --white:#FFFFFF;  --bg:#F7FAF7;
    --r8:8px; --r12:12px; --r16:16px;
    --shadow:0 1px 4px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.04);
    --shadow-md:0 4px 16px rgba(0,0,0,.08);
}
*, *::before, *::after { font-family:'DM Sans',sans-serif !important; box-sizing:border-box; }
.stApp { background:var(--bg) !important; }
header[data-testid="stHeader"]   { display:none !important; }
section[data-testid="stSidebar"] { display:none !important; }
.main { margin-left:0 !important; }
.block-container { padding:0 40px 60px !important; max-width:1440px !important; }

.brand-bar  { display:flex; align-items:center; justify-content:space-between; padding:16px 0 0; }
.brand-name { font-size:1.05rem; font-weight:800; color:var(--g800);
    display:flex; align-items:center; gap:8px; letter-spacing:-.02em; }
.brand-dot  { width:9px; height:9px; border-radius:50%; background:var(--g700); display:inline-block; }
.brand-tag  { font-size:.72rem; color:var(--muted); background:var(--g50);
    border:1px solid var(--g200); border-radius:20px; padding:2px 10px; margin-left:6px; }
.user-pill  { display:flex; align-items:center; gap:8px; background:var(--white);
    border:1px solid var(--border); border-radius:99px; padding:5px 14px 5px 5px; box-shadow:var(--shadow); }
.user-av    { width:28px; height:28px; border-radius:50%; background:var(--g800); color:white;
    font-size:.75rem; font-weight:700; display:flex; align-items:center; justify-content:center; }
.user-name  { font-size:.82rem; font-weight:600; color:var(--text); }

div[data-testid="stTabs"] {
    background:var(--white); border-bottom:1px solid var(--border);
    margin:8px -40px 32px !important; padding:0 40px !important;
    box-shadow:0 2px 8px rgba(0,0,0,.04); }
div[data-testid="stTabs"] > div:first-child { gap:0 !important; background:transparent !important; }
div[data-testid="stTabs"] button[role="tab"] {
    font-size:.87rem !important; font-weight:500 !important; color:var(--muted) !important;
    padding:14px 18px !important; border:none !important; background:transparent !important;
    border-bottom:3px solid transparent !important; border-radius:0 !important; transition:all .18s; }
div[data-testid="stTabs"] button[role="tab"]:hover { color:var(--g800) !important; background:var(--g50) !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color:var(--g800) !important; font-weight:700 !important;
    border-bottom-color:var(--g800) !important; background:transparent !important; }
div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none !important; }
div[data-testid="stTabsContent"] { padding-top:0 !important; }

.card { background:var(--white); border:1px solid var(--border); border-radius:var(--r16);
    padding:24px; box-shadow:var(--shadow); margin-bottom:20px; }
.card-label { font-size:.7rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:var(--muted); margin-bottom:14px; }

.ph { margin-bottom:24px; }
.ph-badge { display:inline-block; background:var(--g100); color:var(--g800);
    font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.08em; border-radius:20px; padding:3px 10px; margin-bottom:8px; }
.ph h1 { font-size:1.7rem; font-weight:800; color:var(--g800); margin:0 0 4px; }
.ph p  { font-size:.92rem; color:var(--muted); margin:0; }

.zb { display:inline-flex; align-items:center; gap:7px; padding:7px 16px;
    border-radius:24px; font-weight:700; font-size:.88rem; letter-spacing:.04em; }
.zb-dot { width:9px; height:9px; border-radius:50%; }
.zb-green  { background:var(--g100); color:var(--g900); }
.zb-green  .zb-dot { background:var(--g800); }
.zb-orange { background:var(--amber-l); color:var(--amber); }
.zb-orange .zb-dot { background:var(--amber); }
.zb-red    { background:var(--red-l); color:var(--red); }
.zb-red    .zb-dot { background:var(--red); }

.si { display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; background:var(--bg); border:1px solid var(--border);
    border-radius:var(--r8); margin-bottom:6px; }
.si-key { font-size:.84rem; font-weight:600; color:var(--text); }
.si-val { font-size:.83rem; color:var(--muted); }
.sb { font-size:.7rem; font-weight:700; padding:3px 10px; border-radius:12px; letter-spacing:.04em; }
.sb-safe     { background:var(--g100);    color:var(--g900); }
.sb-moderate { background:var(--amber-l); color:var(--amber); }
.sb-danger   { background:var(--red-l);   color:var(--red); }

.mb-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; }
.mb { background:var(--g50); border:1px solid var(--border); border-radius:var(--r12); padding:16px; }
.mb-val  { font-size:1.6rem; font-weight:800; color:var(--g800); line-height:1; }
.mb-unit { font-size:.78rem; color:var(--muted); margin-top:2px; }
.mb-lbl  { font-size:.72rem; color:var(--muted); margin-top:6px; text-transform:uppercase;
    letter-spacing:.06em; font-weight:600; }

.cdc { background:linear-gradient(135deg,var(--g800),var(--g900));
    border-radius:var(--r16); padding:22px 24px; color:white; margin-bottom:14px; }
.cdc-label { font-size:.7rem; opacity:.7; text-transform:uppercase; letter-spacing:.1em; }
.cdc-num   { font-size:2.8rem; font-weight:800; line-height:1; }
.cdc-unit  { font-size:.95rem; opacity:.8; margin-left:4px; }
.cdc-sub   { font-size:.76rem; opacity:.65; margin-top:6px; }

.pill   { display:inline-flex; align-items:center; background:var(--g100); color:var(--g800);
    border-radius:20px; padding:5px 12px; font-size:.8rem; font-weight:600; margin:3px; }
.pill-b { background:var(--blue-l); color:var(--blue); }
.pill-section h4 { font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.08em; color:var(--muted); margin:0 0 8px; }

.rx-box { background:var(--g50); border-left:4px solid var(--g700);
    border-radius:0 var(--r8) var(--r8) 0; padding:15px 18px;
    font-size:.86rem; color:var(--text); line-height:1.65; margin-top:14px; }

.winfo { background:var(--blue-l); border-left:4px solid var(--blue);
    border-radius:0 var(--r8) var(--r8) 0; padding:14px 16px;
    font-size:.84rem; color:#0d3a6e; line-height:1.65; margin-top:12px; }

.pcard { background:var(--white); border:2px solid var(--g700);
    border-radius:var(--r16); padding:26px; box-shadow:var(--shadow-md); }
.pcard-hdr { display:flex; justify-content:space-between; align-items:flex-start;
    padding-bottom:16px; border-bottom:1px solid var(--border); margin-bottom:16px; }
.pcard-hdr h2  { margin:0; font-size:1rem; font-weight:800; color:var(--g800); }
.pcard-hdr span{ font-size:.75rem; color:var(--muted); }
.pf { display:flex; justify-content:space-between; align-items:center;
    padding:9px 0; border-bottom:1px solid var(--g50); }
.pf:last-of-type { border-bottom:none; }
.pf-k { font-size:.8rem; color:var(--muted); font-weight:500; }
.pf-v { font-size:.85rem; color:var(--text); font-weight:700; }
.cert { border-radius:var(--r8); padding:12px 16px; font-size:.84rem; font-weight:600; margin-top:16px; }
.cert-ok  { background:var(--g100);    color:var(--g900); border:1px solid var(--g200); }
.cert-mid { background:var(--amber-l); color:var(--amber); border:1px solid var(--amber-b); }
.cert-bad { background:var(--red-l);   color:var(--red);   border:1px solid var(--red-b); }

.stButton > button { background:var(--g800) !important; color:white !important;
    border:none !important; border-radius:var(--r8) !important;
    padding:9px 22px !important; font-weight:600 !important;
    font-size:.88rem !important; transition:background .18s !important; }
.stButton > button:hover { background:var(--g900) !important; }

.stTextInput input, .stNumberInput input, .stSelectbox > div > div, .stTextArea textarea {
    border-radius:var(--r8) !important; font-size:.88rem !important; }

div[data-testid="stAlert"] { border-radius:var(--r12) !important; }

.hero { background:linear-gradient(135deg,var(--g800) 0%,var(--g900) 100%);
    border-radius:var(--r16); padding:40px 36px; color:white; margin-bottom:28px; }
.hero-eyebrow { display:inline-block; background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.2); border-radius:20px;
    padding:4px 14px; font-size:.75rem; font-weight:600; letter-spacing:.06em; margin-bottom:14px; }
.hero h1 { font-size:2.1rem; font-weight:800; margin:0 0 10px; color:white; line-height:1.15; }
.hero p  { font-size:.95rem; opacity:.85; max-width:520px; line-height:1.6; margin:0 0 20px; }
.hero-pills { display:flex; flex-wrap:wrap; gap:7px; }
.hero-pill  { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
    border-radius:20px; padding:4px 13px; font-size:.78rem; color:white; }

.sg { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:28px; }
.sc { background:var(--white); border:1px solid var(--border); border-radius:var(--r12);
    padding:20px 18px; box-shadow:var(--shadow); }
.sc-num  { font-size:1.9rem; font-weight:800; color:var(--g800); line-height:1; }
.sc-unit { font-size:.85rem; color:var(--muted); margin-top:3px; }
.sc-lbl  { font-size:.77rem; color:var(--muted); margin-top:8px; line-height:1.4; }

.pg { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:22px; }
.pc { background:var(--white); border:1px solid var(--border); border-radius:var(--r12);
    padding:20px; box-shadow:var(--shadow); position:relative; overflow:hidden; }
.pc::before { content:""; position:absolute; top:0; left:0; right:0; height:4px; background:var(--g700); }
.pc-num { font-size:.68rem; font-weight:700; color:var(--g800); background:var(--g100);
    border-radius:6px; padding:2px 8px; display:inline-block; margin-bottom:10px; }
.pc h4  { margin:0 0 6px; font-size:.88rem; font-weight:700; color:var(--text); }
.pc p   { margin:0; font-size:.78rem; color:var(--muted); line-height:1.5; }

.stChatMessage { background:var(--white) !important; border-radius:var(--r12) !important;
    border:1px solid var(--border) !important; margin-bottom:8px !important; }

.sep { border:none; border-top:1px solid var(--border); margin:22px 0; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
_FACTORY_LAT, _FACTORY_LON = 33.883, 10.100   # SIAPE phosphate plant

def _dist_km(lat: float, lon: float) -> float:
    R = 6371.0
    dlat = math.radians(lat - _FACTORY_LAT)
    dlon = math.radians(lon - _FACTORY_LON)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(_FACTORY_LAT)) * math.cos(math.radians(lat))
         * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def zone_badge(z: str) -> str:
    cls = {"GREEN":"zb-green","ORANGE":"zb-orange","RED":"zb-red"}.get(z,"zb-green")
    lbl = {"GREEN":"Safe Zone","ORANGE":"Moderate Zone","RED":"Contaminated Zone"}.get(z,z)
    return f'<div class="zb {cls}"><span class="zb-dot"></span>{lbl}</div>'

def classify_cd(v):
    if v < CD_SAFE_THRESHOLD: return "Safe","sb-safe"
    if v >= CD_EU_LIMIT:      return "High Risk","sb-danger"
    return "Moderate","sb-moderate"

def classify_pb(v):
    if v < PB_MODERATE_RISK: return "Safe","sb-safe"
    if v >= PB_HIGH_RISK:    return "High Risk","sb-danger"
    return "Moderate","sb-moderate"

def classify_ec(v):
    return ("No Stress","sb-safe") if v < EC_SALINITY_STRESS else ("Salinity Stress","sb-moderate")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
# ... (keep existing imports and _dist_km, zone_badge, classify_cd, classify_pb) ...

def classify_ph(v):
    """Classify pH for UI display."""
    if v <= PH_ACIDIC_SEVERE or v >= PH_ALKALINE_SEVERE:
        return "Critical","sb-danger"
    if v <= PH_ACIDIC_STRESS or v >= PH_ALKALINE_STRESS:
        return "Stress","sb-moderate"
    return "Optimal","sb-safe"

# ── SINGLE CANONICAL ZONE FUNCTION (FIXED WITH pH) ─────────────────────
def rule_zone(ec: float, cd: float, pb: float, ph: float) -> str:
    """
    Scientific logic accounting for EC, Cd, Pb, AND pH.
    Priority: RED > ORANGE > GREEN.
    """
    # 1. RED CONDITIONS (Critical)
    # High Metals, Severe Salinity, OR Extreme pH
    if (cd >= CD_EU_LIMIT or 
        pb >= PB_HIGH_RISK or 
        ec >= EC_SALINITY_SEVERE or 
        ph <= PH_ACIDIC_SEVERE or 
        ph >= PH_ALKALINE_SEVERE):
        return "RED"
    
    # 2. ORANGE CONDITIONS (Moderate Risk)
    # Moderate Metals, Moderate Salinity, OR Stressful pH
    if (ec >= EC_SALINITY_STRESS or 
        cd >= CD_SAFE_THRESHOLD or 
        pb >= PB_MODERATE_RISK or 
        ph <= PH_ACIDIC_STRESS or 
        ph >= PH_ALKALINE_STRESS):
        return "ORANGE"
    
    # 3. GREEN CONDITIONS
    return "GREEN"

def rule_plants(zone: str, ec: float) -> str:
    if zone == "RED":
        base = "Noccaea caerulescens (50%), Sedum alfredii (40%)"
        return base + ", Atriplex halimus (10%)" if ec >= EC_SALINITY_STRESS else base
    if zone == "ORANGE":
        return "Atriplex halimus (40%), Vetiver grass (30%), Local forage (30%)"
    return "Pomegranate (60%), Olive (40%)"

def rule_microbes(zone: str) -> str:
    if zone == "RED":    return "Pseudomonas fluorescens, Rhizobium leguminosarum, AM fungi"
    if zone == "ORANGE": return "Azospirillum brasilense, Pseudomonas fluorescens"
    return "Rhizobium spp., Trichoderma harzianum"

def rule_months(cd: float, zone: str) -> int:
    rate = 0.35 if zone == "RED" else 0.50
    return max(1, int(cd / rate))

def pills(text, cls="pill"):
    items = [i.strip() for i in str(text).replace(";",",").split(",") if i.strip()]
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)

def llm_client():
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

def call_llm_rx(inputs: dict) -> dict:
    try:
        client = llm_client()
        prompt = PRESCRIPTION_USER_PROMPT.format(
            ec=round(inputs.get("ec",0),2), ph=inputs.get("ph",7),
            cd=round(inputs.get("cd",0),3), pb=round(inputs.get("pb",0),1),
            soil_type="Loam",
        )
        r = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role":"system","content":PRESCRIPTION_SYSTEM_PROMPT},
                {"role":"user","content":prompt},
            ],
            temperature=0.2, max_tokens=900,
        )
        raw = r.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# ─── BRAND BAR ────────────────────────────────────────────────────────────────
bc, _, uc = st.columns([3,5,3])
with bc:
    st.markdown(f"""
    <div class="brand-bar">
      <div class="brand-name">
        <span class="brand-dot"></span> Gabes Regenerate
        <span class="brand-tag">SoilRevive Engine</span>
      </div>
    </div>""", unsafe_allow_html=True)
with uc:
    pu_c, lo_c = st.columns([3,1])
    with pu_c:
        st.markdown(f"""
        <div class="brand-bar" style="justify-content:flex-end">
          <div class="user-pill">
            <div class="user-av">{initials}</div>
            <span class="user-name">{user.get("name","User")}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with lo_c:
        if st.button("Out", key="logout"):
            logout(); st.rerun()

# ─── TABS ─────────────────────────────────────────────────────────────────────
t_home, t_p1, t_p2, t_p3, t_p4, t_chat = st.tabs([
    "Home",
    "Phase 1 — Diagnostics",
    "Phase 2 — Prescription",
    "Phase 3 — Water & Timeline",
    "Phase 4 — Soil Passport",
    "AI Assistant",
])

# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
with t_home:
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Gabès · Tunisia · Bioremediation AI</div>
      <h1>Healing Gabes,<br>One Field at a Time</h1>
      <p>An AI-powered soil remediation platform combining field sensors,
         machine learning, and agronomic expertise to reverse heavy-metal
         contamination across the Gabès basin.</p>
      <div class="hero-pills">
        <span class="hero-pill">Cadmium Remediation</span>
        <span class="hero-pill">Lead Detoxification</span>
        <span class="hero-pill">Salinity Management</span>
        <span class="hero-pill">ML Prescription Engine</span>
        <span class="hero-pill">Verifiable Soil Passport</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--g900);border-radius:var(--r8);padding:10px 16px;margin-bottom:24px">
      <div style="color:rgba(255,255,255,.6);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em">Context</div>
      <div style="color:white;font-size:.82rem;margin-top:3px">
        Gabès industrial zone · 15 000+ ha affected · Cd up to 12× EU limit ·
        SIAPE phosphate plant since 1972 · 360 000 residents in proximity ·
        Platform targets 5 000 small farms for precision remediation
      </div>
    </div>
    <div class="sg">
      <div class="sc"><div class="sc-num">15 k</div><div class="sc-unit">hectares</div>
        <div class="sc-lbl">Contaminated agricultural land in the Gabès basin</div></div>
      <div class="sc"><div class="sc-num">12×</div><div class="sc-unit">above EU Cd limit</div>
        <div class="sc-lbl">Peak cadmium near the SIAPE complex</div></div>
      <div class="sc"><div class="sc-num">360 k</div><div class="sc-unit">residents</div>
        <div class="sc-lbl">Population within the affected coastal catchment</div></div>
      <div class="sc"><div class="sc-num">3 – 5</div><div class="sc-unit">years</div>
        <div class="sc-lbl">Estimated remediation window using hyperaccumulators</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Platform Phases")
    st.markdown("""
    <div class="pg">
      <div class="pc"><div class="pc-num">Phase 1</div><h4>Low-Cost Diagnostics</h4>
        <p>Enter GPS, pH, EC and NIR readings — the AI predicts Cd and Pb without lab costs.</p></div>
      <div class="pc"><div class="pc-num">Phase 2</div><h4>AI Prescription</h4>
        <p>LLM + rule engine classifies the zone and prescribes a plant-microbe remediation plan.</p></div>
      <div class="pc"><div class="pc-num">Phase 3</div><h4>Water & Timeline</h4>
        <p>ML model computes seasonal irrigation needs and projects a Cd depletion curve.</p></div>
      <div class="pc"><div class="pc-num">Phase 4</div><h4>Soil Passport</h4>
        <p>Verifiable field certificate with QR code for premium-market access.</p></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    hc1, hc2 = st.columns([2,1])
    with hc1:
        df_z = pd.DataFrame({
            "Zone":["Critical (Red)","Moderate (Orange)","Recovering (Green)"],
            "ha":  [4800,6200,4000],
        })
        fig_z = px.bar(df_z, x="Zone", y="ha", color="Zone",
            color_discrete_map={"Critical (Red)":"#B71C1C","Moderate (Orange)":"#E65100","Recovering (Green)":"#2E7D32"},
            title="Estimated Land Classification – Gabès Basin")
        fig_z.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,height=280,margin=dict(t=40,b=0,l=0,r=0),
            font=dict(family="DM Sans"),title_font=dict(size=13,color="#111827"),yaxis_title="Hectares")
        fig_z.update_traces(marker_line_width=0,width=0.5)
        st.plotly_chart(fig_z, use_container_width=True)
    with hc2:
        st.markdown("""
        <div class="card" style="height:100%">
          <div class="card-label">Quick Start</div>
          <div style="font-size:.83rem;color:#555;line-height:1.85">
            <b>1.</b> Go to <b>Phase 1</b> — enter your 5 field readings<br>
            <b>2.</b> Click <b>Run AI Diagnostic</b> to predict Cd and Pb<br>
            <b>3.</b> Confirm and open <b>Phase 2</b> for the prescription<br>
            <b>4.</b> Use <b>Phase 3</b> for irrigation plan and timeline<br>
            <b>5.</b> Download your <b>Soil Passport</b> in Phase 4
          </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — exactly 5 inputs: lat, lon, pH, EC, NIR
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
with t_p1:
    st.markdown("""
    <div class="ph">
      <span class="ph-badge">Phase 1</span>
      <h1>Low-Cost Diagnostics</h1>
      <p>Use the interactive map and field sensors (pH, EC, NIR) to predict heavy-metal contamination.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize map state
    if "last_clicked_coords" not in st.session_state:
        st.session_state.last_clicked_coords = (33.87, 10.10)

    col1, col2 = st.columns([1.2, 1.8], gap="large")

    # ──────────────────────────────────────────────────────────────────────────
    # LEFT: MAP & COORDS
    # ──────────────────────────────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📍 Field Location</div>', unsafe_allow_html=True)

        m = folium.Map(location=st.session_state.last_clicked_coords, zoom_start=13, tiles="cartodbpositron")
        folium.Marker(st.session_state.last_clicked_coords, popup="Selected Zone", icon=folium.Icon(color="green")).add_to(m)

        st_data = st_folium(m, height=280, width=None, key="p1_map")
        if st_data and st_data.get('last_clicked'):
            st.session_state.last_clicked_coords = (st_data['last_clicked']['lat'], st_data['last_clicked']['lng'])

        c_lat, c_lon = st.columns(2)
        lat = c_lat.number_input("Latitude", value=st.session_state.last_clicked_coords[0], format="%.5f", help="Click map or type")
        lon = c_lon.number_input("Longitude", value=st.session_state.last_clicked_coords[1], format="%.5f", help="Click map or type")

        # Sync manual input back to map state
        if lat != st.session_state.last_clicked_coords[0] or lon != st.session_state.last_clicked_coords[1]:
            st.session_state.last_clicked_coords = (lat, lon)

        st.markdown("</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # RIGHT: SENSORS & CONTEXT
    # ──────────────────────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">🔬 Field Sensors & Context</div>', unsafe_allow_html=True)

        # ✅ pH starts at 3.0 as requested
        ph = st.slider("Soil pH Meter Reading", min_value=3.0, max_value=9.5, value=7.0, step=0.1,
                       help="Covers highly acidic to alkaline soils (min 3.0)")

        # ✅ EC free input (no max constraint)
        ec = st.number_input("Electrical Conductivity — EC (dS/m)", min_value=0.0, value=3.0, step=0.1,
                             format="%.2f", help="Enter exact field reading. Free typing enabled.")

        if 'spectrum_text' not in st.session_state:
            st.session_state.spectrum_text = "[]"
        spectrum_text = st.text_area(
            "NIR Spectrum (JSON array)",
            value=st.session_state.spectrum_text, height=90,
            help='Paste reflectance array: e.g. [0.12, 0.45, 0.78, ...]'
        )

        st.markdown("<hr style='margin:16px 0; border-top:1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown('<div class="card-label" style="margin-bottom:8px;">⚙️ Field Context</div>', unsafe_allow_html=True)

        dist = st.slider("Distance from SIAPE Factory (km)", min_value=0.0, max_value=20.0, value=2.5, step=0.5)
        soil_type = st.selectbox("Soil Texture", ["Sandy", "Sandy-Loam", "Loam", "Clay-Loam"], index=1)

        st.markdown("</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION & PREDICTION
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("<hr style='margin:24px 0; border-top:1px solid var(--border);'>", unsafe_allow_html=True)

    if st.button("🤖 Run AI Diagnostic (Predict Cd & Pb)", use_container_width=True, type="primary"):
        st.session_state['inputs_ready'] = False

        from agents.lucas_agent import get_lucas_agent
        lucas_agent = get_lucas_agent()

        if lucas_agent.loaded:
            with st.spinner("Running AI Model..."):
                try:
                    spectrum = json.loads(spectrum_text)
                    st.session_state['spectrum_text'] = spectrum_text

                    cd_pred, pb_pred = lucas_agent.predict_heavy_metals(
                        lat=lat, lon=lon, ph=ph, ec=ec, dist=dist, spectrum=spectrum
                    )

                    st.session_state.update({
                        'pred_cd': cd_pred,
                        'pred_pb': pb_pred,
                        'spectrum': spectrum,
                        'inputs_ready': True
                    })
                    st.rerun()  # Immediately refresh UI with results
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format for NIR spectrum.")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
        else:
            st.error("⚠️ AI Model not loaded. Please check the agents/models directory.")

    
    # ──────────────────────────────────────────────────────────────────────────
    # RESULTS & CONFIRMATION (FIXED LOGIC)
    # ──────────────────────────────────────────────────────────────────────────
    if st.session_state.get('inputs_ready'):
        cd = st.session_state['pred_cd']
        pb = st.session_state['pred_pb']
        ec_val = ec 
        ph_val = ph # Capture pH from the input widget

        # --- FIX: Pass pH to rule_zone ---
        zone = rule_zone(ec_val, cd, pb, ph_val)
        
        # Generate Labels
        cd_lbl, cd_cls = classify_cd(cd)
        pb_lbl, pb_cls = classify_pb(pb)
        ec_lbl, ec_cls = ("Catastrophic","sb-danger") if ec_val >= EC_SALINITY_SEVERE else \
                         ("Salinity Stress","sb-moderate") if ec_val >= EC_SALINITY_STRESS else \
                         ("Normal","sb-safe")
        
        # NEW: pH Classification
        ph_lbl, ph_cls = classify_ph(ph_val)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">🎯 AI Prediction Results</div>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        r1.metric("Predicted Cadmium (Cd)", f"{cd:.3f} mg/kg")
        r2.metric("Predicted Lead (Pb)", f"{pb:.1f} mg/kg")

        st.markdown("<hr style='margin:16px 0; border-top:1px solid var(--border);'>", unsafe_allow_html=True)

        # Display Logic
        overall_status = {
            "GREEN": ("🟢 Safe Zone", "sb-safe"),
            "ORANGE": ("🟡 Moderate Risk", "sb-moderate"),
            "RED": ("🔴 Contaminated Zone", "sb-danger")
        }.get(zone, ("Unknown", "sb-safe"))
        
        st.markdown(f"""
        <div class="si">
          <span class="si-key">Overall Classification</span>
          <span class="si-val">{overall_status[0]}</span>
          <span class="sb {overall_status[1]}">Zone: {zone}</span>
        </div>
        <div class="si">
          <span class="si-key">Cadmium (Cd)</span>
          <span class="si-val">{cd:.2f} mg/kg</span>
          <span class="sb {cd_cls}">{cd_lbl}</span>
        </div>
        <div class="si">
          <span class="si-key">Lead (Pb)</span>
          <span class="si-val">{pb:.1f} mg/kg</span>
          <span class="sb {pb_cls}">{pb_lbl}</span>
        </div>
        <div class="si">
          <span class="si-key">Salinity (EC)</span>
          <span class="si-val">{ec_val:.1f} dS/m</span>
          <span class="sb {ec_cls}">{ec_lbl}</span>
        </div>
        <div class="si">
          <span class="si-key">Acidity (pH)</span>
          <span class="si-val">{ph_val:.1f}</span>
          <span class="sb {ph_cls}">{ph_lbl}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("✅ Confirm & Proceed to Phase 2", use_container_width=True):
            st.session_state['inputs'] = {
                'lat': lat, 'lon': lon,
                'ph': ph_val, 'ec': ec_val, 'nir_spectrum': st.session_state.get('spectrum', []),
                'cd': cd, 'pb': pb, 
                'zone': zone, # Saving the calculated zone
                'dist_km': dist, 'soil_type': soil_type, 'zn': 100
            }
            st.success("✅ Data Saved! Switch to Phase 2 in the navigation bar.")
   


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — zone ALWAYS from rule_zone — LLM only contributes reasoning & plants
# ═══════════════════════════════════════════════════════════════════════════════
with t_p2:
    st.markdown("""
    <div class="ph">
      <div class="ph-badge">Phase 2</div>
      <h1>AI Prescription Engine</h1>
      <p>The AI agronomist analyses your confirmed soil profile and delivers a full
         bioremediation plan with step-by-step reasoning.</p>
    </div>""", unsafe_allow_html=True)

    if "inputs" not in st.session_state:
        st.warning("Complete Phase 1 first, then click Confirm.")
    else:
        inputs = st.session_state["inputs"]
        zone   = inputs["zone"]   # always from rule_zone — never from LLM

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Confirmed Soil Profile</div>', unsafe_allow_html=True)
        sm = st.columns(4)
        for col,(lbl,val,unit) in zip(sm,[
            ("EC",      f"{inputs['ec']:.1f}", "dS/m"),
            ("pH",      f"{inputs['ph']:.1f}", ""),
            ("Cadmium", f"{inputs['cd']:.3f}", "mg/kg"),
            ("Lead",    f"{inputs['pb']:.1f}",  "mg/kg"),
        ]):
            col.metric(lbl, f"{val} {unit}".strip())
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(zone_badge(zone), unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        if st.button("Generate AI Prescription", key="run_rx"):
            with st.spinner("Consulting the AI Agronomist…"):
                agent  = get_agent()
                result = agent.predict(features=inputs)
                if result is None:
                    from agents.prediction_agent import ScientificRuleModel
                    result = ScientificRuleModel().predict(features=inputs)

                # Fetch LLM output for reasoning + plant mix
                llm_out = call_llm_rx(inputs)
                if llm_out and "error" not in llm_out:
                    result["reasoning"]   = llm_out.get("reasoning",  result.get("reasoning",""))
                    result["plant_mix"]   = llm_out.get("plant_mix",   result.get("plant_mix",""))
                    result["microbe_mix"] = llm_out.get("microbe_mix", result.get("microbe_mix",""))

                # ALWAYS override zone_color with the rule-based value
                result["zone_color"] = zone
                st.session_state["result"] = result

        if "result" in st.session_state:
            res = st.session_state["result"]
            res["zone_color"] = zone   # re-lock on every render

            rx_col, re_col = st.columns([1,1.2], gap="large")

            with rx_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                p_text = res.get("plant_mix",  rule_plants(zone, inputs["ec"]))
                m_text = res.get("microbe_mix", rule_microbes(zone))
                if isinstance(p_text, list): p_text = ", ".join(p_text)
                if isinstance(m_text, list): m_text = ", ".join(m_text)
                safe_f = res.get("safe_for_fodder", zone != "RED")
                f_cls  = "sb-safe" if safe_f else "sb-danger"
                f_lbl  = "Safe for fodder" if safe_f else "NOT safe for fodder"
                st.markdown(f"""
                <div class="pill-section"><h4>Recommended Plant Mix</h4>{pills(p_text)}</div>
                <div class="pill-section" style="margin-top:14px"><h4>Microbe Injection Protocol</h4>
                  {pills(m_text,"pill pill-b")}</div>
                <div style="margin-top:16px">
                  <span class="sb {f_cls}" style="font-size:.8rem;padding:5px 14px">{f_lbl}</span>
                </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with re_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">AI Decision Reasoning</div>', unsafe_allow_html=True)
                reasoning = res.get("reasoning","")
                if reasoning:
                    st.markdown(f'<div class="rx-box">{reasoning}</div>', unsafe_allow_html=True)
                else:
                    cd=inputs["cd"]; pb=inputs["pb"]; ec2=inputs["ec"]
                    cd_s = ("exceeds the EU limit" if cd>=CD_EU_LIMIT
                            else ("is above the safe threshold" if cd>=CD_SAFE_THRESHOLD else "is within safe range"))
                    pb_s = ("exceeds the high-risk threshold" if pb>=PB_HIGH_RISK
                            else ("is at moderate risk" if pb>=PB_MODERATE_RISK else "is within safe range"))
                    ec_s = ("exceeds the salinity threshold"
                            if ec2>=EC_SALINITY_STRESS else "is within acceptable range")
                    auto = (
                        f"Cadmium was measured at <strong>{cd:.3f} mg/kg</strong> — this {cd_s} "
                        f"(EU limit {CD_EU_LIMIT} mg/kg, safe below {CD_SAFE_THRESHOLD} mg/kg). "
                        f"Lead was measured at <strong>{pb:.1f} mg/kg</strong> — this {pb_s} "
                        f"(high risk above {PB_HIGH_RISK} mg/kg, moderate above {PB_MODERATE_RISK} mg/kg). "
                        f"Electrical conductivity is <strong>{ec2:.1f} dS/m</strong> — this {ec_s} "
                        f"(stress above {EC_SALINITY_STRESS} dS/m). "
                        f"Classification: <strong>{zone} ZONE</strong>."
                    )
                    st.markdown(f'<div class="rx-box">{auto}</div>', unsafe_allow_html=True)
                    if st.button("Fetch Full LLM Reasoning", key="fetch_reason"):
                        with st.spinner("Calling LLM…"):
                            out = call_llm_rx(inputs)
                            if out and "error" not in out:
                                res["reasoning"] = out.get("reasoning","No reasoning returned.")
                                st.session_state["result"] = res
                                st.rerun()
                            else:
                                st.error(f"LLM error: {out.get('error','Unknown')}")
                st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Water & Timeline
# ═══════════════════════════════════════════════════════════════════════════════
with t_p3:
    st.markdown("""
    <div class="ph">
      <div class="ph-badge">Phase 3</div>
      <h1>Water Management & Remediation Timeline</h1>
      <p>Compute precise seasonal irrigation needs for your prescribed plants
         and visualise the projected cadmium depletion curve to safety.</p>
    </div>""", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Complete Phase 2 before accessing Phase 3.")
    else:
        res    = st.session_state["result"]
        inp    = st.session_state["inputs"]
        zone   = inp["zone"]
        cd_lvl = inp.get("cd", 0)

        wi_col, wv_col = st.columns([1,1.5], gap="large")

        with wi_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Irrigation Parameters</div>', unsafe_allow_html=True)
            temp   = st.slider("Average Temperature (°C)", 10, 45, 25)
            season = st.selectbox("Season", ["Summer","Winter","Autumn","Spring"])
            pmix   = str(res.get("plant_mix",""))
            dc     = "Atriplex" if "Atriplex" in pmix else ("Pomegranate" if "Pomegranate" in pmix else "Tomato")
            crop   = st.selectbox("Primary Crop",
                                  ["Atriplex","Pomegranate","Tomato","Wheat","Barley"],
                                  index=["Atriplex","Pomegranate","Tomato","Wheat","Barley"].index(dc))
            if st.button("Calculate Water Needs", use_container_width=True, key="calc_water"):
                eng = get_water_engine()
                if eng.loaded:
                    pred, msg = eng.predict(soil_ph=inp["ph"],temperature=temp,
                                             crop_type=crop,season=season)
                    if pred: st.session_state["water_pred"] = pred
                    else:    st.error(f"Prediction error: {msg}")
                else:
                    st.error("Water ML model not loaded.")
            st.markdown("</div>", unsafe_allow_html=True)

        with wv_col:
            if "water_pred" in st.session_state:
                pred = st.session_state["water_pred"]
                gmax = max(500, int(pred*1.6))
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=pred,
                    delta={"reference":350,"valueformat":".0f"},
                    title={"text":"Required Water (mm / season cycle)",
                           "font":{"color":"#2E7D32","size":14,"family":"DM Sans"}},
                    number={"suffix":" mm","font":{"size":36,"color":"#1B5E20","family":"DM Sans"}},
                    gauge={
                        "axis":{"range":[0,gmax],"tickcolor":"#888"},
                        "bar":{"color":"#2E7D32","thickness":.25},
                        "bgcolor":"white","borderwidth":1,"bordercolor":"#E5E7EB",
                        "steps":[
                            {"range":[0,gmax*.3],"color":"#E8F5E9"},
                            {"range":[gmax*.3,gmax*.7],"color":"#C8E6C9"},
                            {"range":[gmax*.7,gmax],"color":"#A5D6A7"},
                        ],
                        "threshold":{"line":{"color":"#E65100","width":2},"thickness":.8,"value":350},
                    },
                ))
                fig_g.update_layout(height=300,margin=dict(t=60,b=0,l=20,r=20),
                                    paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"))
                st.plotly_chart(fig_g, use_container_width=True)

                # Practical irrigation guidance
                weekly_mm = round(pred / 13, 1)
                liters_ha = int(pred * 10)
                if pred < 200:
                    intensity = "low water demand"
                    schedule  = "1–2 sessions per week"
                    timing    = "early morning (06:00–08:00) to minimise evaporation"
                elif pred < 350:
                    intensity = "moderate water demand"
                    schedule  = "2–3 sessions per week"
                    timing    = "early morning or late afternoon (17:00–19:00)"
                else:
                    intensity = "high water demand"
                    schedule  = "daily or every other day"
                    timing    = "split between morning and evening to avoid heat stress"

                season_note = {
                    "Summer":  "Summer heat in Gabès can reach 40 °C — increase frequency during heat waves and monitor soil moisture daily.",
                    "Winter":  "Lower evapotranspiration in winter — reduce sessions by ~30% and avoid waterlogging.",
                    "Spring":  "Optimal planting window with moderate temperatures — stable water use across the season.",
                    "Autumn":  "Taper irrigation gradually as temperatures drop; root systems are still active.",
                }.get(season,"")

                crop_note = {
                    "Atriplex":    "Atriplex halimus tolerates brackish water (EC up to 15 dS/m) and can be irrigated with slightly saline water without yield loss.",
                    "Pomegranate": "Pomegranate requires well-drained soil — avoid standing water for more than 24 hours to prevent root rot.",
                    "Tomato":      "Maintain consistent moisture especially at flowering; water stress at this stage causes blossom drop.",
                    "Wheat":       "Highest demand during tillering (months 2–3) and grain-fill (months 5–6) — prioritise water at these stages.",
                    "Barley":      "More drought-tolerant than wheat; reduce water by 20% once established after the first 4 weeks.",
                }.get(crop,"")

                st.markdown(f"""
                <div class="winfo">
                  <strong>Irrigation Plan — {crop} · {season}</strong><br><br>
                  Your field has <strong>{intensity}</strong> this season, requiring approximately
                  <strong>{pred:.0f} mm</strong> per cycle (~{weekly_mm} mm/week · ~{liters_ha:,} L/ha/cycle).<br><br>
                  <strong>Recommended schedule:</strong> {schedule}, applied {timing}.<br><br>
                  <strong>Seasonal note:</strong> {season_note}<br><br>
                  <strong>Crop note:</strong> {crop_note}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="height:300px;display:flex;align-items:center;justify-content:center;
                            background:var(--g50);border-radius:var(--r12);border:1px dashed var(--g200)">
                  <p style="color:var(--muted);font-size:.88rem">
                    Configure parameters and click <b>Calculate Water Needs</b>
                  </p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown("### Cadmium Depletion Timeline")

        if zone == "GREEN":
            st.success("🎉 Your field is already in the safe zone — no remediation timeline needed.")
        else:
            rate   = 0.35 if zone == "RED" else 0.50
            months = rule_months(cd_lvl, zone)
            plant_src = ("Noccaea caerulescens + Sedum alfredii"
                         if zone == "RED" else "Atriplex halimus + Vetiver grass")

            st.markdown(f"""
            <div class="winfo" style="margin-bottom:16px">
              <strong>How is this calculated?</strong><br>
              The prescribed plant mix (<em>{plant_src}</em>) absorbs cadmium through phytoextraction —
              metals accumulate in the plant shoots, which are harvested and removed from the field.
              Based on field studies in Mediterranean soils, this achieves a removal rate of
              <strong>{rate} mg Cd · kg⁻¹ · month⁻¹</strong>.<br><br>
              Starting from <strong>{cd_lvl:.3f} mg/kg</strong>, the model projects
              <strong>{months} months ({months//12} yr {months%12} mo)</strong> to reach the safe
              threshold of {CD_SAFE_THRESHOLD} mg/kg. This assumes consistent planting density and
              crop rotation every 6 months.
            </div>""", unsafe_allow_html=True)

            m_axis  = list(range(0, months + 2))
            cd_traj = [max(0.0, cd_lvl - rate*m) for m in m_axis]

            fig2 = px.area(x=m_axis, y=cd_traj,
                           labels={"x":"Month","y":"Cd (mg/kg)"},
                           color_discrete_sequence=["#388E3C"])
            fig2.add_hline(y=CD_EU_LIMIT, line_dash="dash", line_color="#B71C1C",
                           annotation_text=f"EU Limit ({CD_EU_LIMIT} mg/kg)")
            fig2.add_hline(y=CD_SAFE_THRESHOLD, line_dash="dot", line_color="#1B5E20",
                           annotation_text=f"Safe Threshold ({CD_SAFE_THRESHOLD} mg/kg)")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="DM Sans"),height=320,
                               margin=dict(t=20,b=20,l=0,r=0))
            st.plotly_chart(fig2, use_container_width=True)

            m1 = next((m for m,v in zip(m_axis,cd_traj) if v<=CD_EU_LIMIT), None)
            m2 = next((m for m,v in zip(m_axis,cd_traj) if v<=CD_SAFE_THRESHOLD), None)
            ms1,ms2,ms3 = st.columns(3)
            ms1.metric("Today",          f"{cd_lvl:.3f} mg/kg", "starting point")
            ms2.metric("Below EU Limit", f"Month {m1}" if m1 else "—", f"< {CD_EU_LIMIT} mg/kg")
            ms3.metric("Fully Safe",     f"Month {m2}" if m2 else "—", f"< {CD_SAFE_THRESHOLD} mg/kg")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — SOIL PASSPORT
# ═══════════════════════════════════════════════════════════════════════════════
with t_p4:
    st.markdown("""
    <div class="ph">
      <div class="ph-badge">Phase 4</div>
      <h1>Soil Passport</h1>
      <p>A verifiable micro-zone certificate summarising contamination levels,
         remediation prescription, and market eligibility.</p>
    </div>""", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Complete Phases 1 and 2 before generating a passport.")
    else:
        res  = st.session_state["result"]
        inp  = st.session_state["inputs"]
        zone = inp["zone"]
        mons = rule_months(inp.get("cd",0), zone)

        passport_result = res.copy()
        passport_result["input"] = {
            "ec_ds_m":inp["ec"],"ph":inp["ph"],
            "cd_mg_kg":inp["cd"],"pb_mg_kg":inp["pb"],
        }
        passport_result["planting_cycles"] = mons // 6

        pc_col, pv_col = st.columns([1.2,1], gap="large")

        with pc_col:
            cert_cls = {"GREEN":"cert-ok","ORANGE":"cert-mid","RED":"cert-bad"}.get(zone,"cert-ok")
            cert_txt = {
                "GREEN":  "✅ REGENERATIVE CERTIFIED — Safe for premium market",
                "ORANGE": "⚠️ MONITORING REQUIRED — Fodder safe, metals under review",
                "RED":    "🚫 TOXIC BIOMASS — Route to biochar facility",
            }.get(zone,"")
            zone_explain = {
                "GREEN":  "All metal and salinity levels are within safe limits. This land is market-ready for food crops.",
                "ORANGE": "Salinity or moderate metal contamination detected. Remediation is underway; fodder crops are permitted under monitoring. Do not plant human food crops until next assessment.",
                "RED":    "Cadmium or lead exceeds EU safety limits. Harvested biomass from this land must NOT enter the food chain and must be routed to biochar or industrial processing.",
            }.get(zone,"")
            safe_f = res.get("safe_for_fodder", zone != "RED")
            p_text = res.get("plant_mix", rule_plants(zone, inp["ec"]))
            if isinstance(p_text,list): p_text = ", ".join(p_text)

            st.markdown(f"""
            <div class="pcard">
              <div class="pcard-hdr">
                <div>
                  <h2>Soil Passport — {inp.get("zone_id","GAB-001")}</h2>
                  <span>Gabes Regenerate AI · SoilRevive Engine v5.2</span>
                </div>
                <div>{zone_badge(zone)}</div>
              </div>

              <div class="pf"><span class="pf-k">Farmer</span>
                <span class="pf-v">{inp.get("farmer_name","—") or "—"}</span></div>
              <div class="pf"><span class="pf-k">GPS Coordinates</span>
                <span class="pf-v">{inp.get("lat",0):.5f} N, {inp.get("lon",0):.5f} E</span></div>
              <div class="pf"><span class="pf-k">Distance to SIAPE Factory</span>
                <span class="pf-v">{_dist_km(inp.get("lat",33.87),inp.get("lon",10.10)):.2f} km</span></div>

              <div class="pf"><span class="pf-k">Soil pH</span>
                <span class="pf-v">{inp["ph"]:.1f}</span></div>
              <div class="pf"><span class="pf-k">EC (Salinity)</span>
                <span class="pf-v">{inp["ec"]:.2f} dS/m &nbsp;
                  {'· above stress threshold' if inp["ec"]>=EC_SALINITY_STRESS else '· within normal range'}
                </span></div>

              <div class="pf"><span class="pf-k">Cadmium (Cd)</span>
                <span class="pf-v">{inp["cd"]:.3f} mg/kg &nbsp;
                  · EU limit {CD_EU_LIMIT} · safe &lt;{CD_SAFE_THRESHOLD}
                </span></div>
              <div class="pf"><span class="pf-k">Lead (Pb)</span>
                <span class="pf-v">{inp["pb"]:.1f} mg/kg &nbsp;
                  · high risk &gt;{PB_HIGH_RISK} · moderate &gt;{PB_MODERATE_RISK}
                </span></div>

              <div class="pf"><span class="pf-k">Remediation Time</span>
                <span class="pf-v">{mons} months ({mons//12} yr {mons%12} mo) · {mons//6} planting cycles</span></div>
              <div class="pf"><span class="pf-k">Safe for Fodder</span>
                <span class="pf-v">{"✅ Yes" if safe_f else "❌ No — biochar route required"}</span></div>
              <div class="pf"><span class="pf-k">Plant Prescription</span>
                <span class="pf-v" style="max-width:58%;text-align:right;font-size:.76rem">{p_text[:80]}</span></div>

              <div style="background:var(--g50);border-radius:var(--r8);padding:10px 14px;
                          font-size:.78rem;color:var(--muted);margin:12px 0;line-height:1.6">
                {zone_explain}
              </div>
              <div class="cert {cert_cls}">{cert_txt}</div>
            </div>""", unsafe_allow_html=True)

        if PASSPORT_OK:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            
            # Generate QR bytes
            qr_bytes = generate_qr_bytes(
                inp.get("zone_id", "GAB-001"), 
                passport_result, 
                inp.get("farmer_name", "Farmer")
            )

            # 1️⃣ Display QR prominently on screen
            qr_col, info_col = st.columns([1, 3])
            with qr_col:
                st.image(qr_bytes, caption="📱 Scan to verify passport", width=180)
            with info_col:
                st.info("Customers can scan this QR to instantly view contamination levels, zone status, and market eligibility. All data is embedded and verifiable offline.")

            # 2️⃣ Download buttons
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇ Download QR (PNG)", data=qr_bytes,
                    file_name=f"qr_{inp.get('zone_id','zone')}.png",
                    mime="image/png", use_container_width=True
                )
            with d2:
                # Pass QR bytes to embed inside PDF
                pdf = generate_passport_pdf(
                    inp.get("zone_id", "GAB-001"), 
                    passport_result, 
                    inp.get("farmer_name", "Farmer"),
                    qr_bytes=qr_bytes  # ✅ New parameter
                )
                if pdf:
                    st.download_button(
                        "📄 Download Passport (PDF)", data=pdf,
                        file_name=f"passport_{inp.get('zone_id','zone')}.pdf",
                        mime="application/pdf", use_container_width=True
                    )
        else:
                st.info("Install utils/passport_generator.py to enable PDF/QR export.")

        with pv_col:
            cd_pct = min(inp["cd"]/CD_EU_LIMIT*100, 200)
            pb_pct = min(inp["pb"]/PB_HIGH_RISK*100, 200)
            ec_pct = min(inp["ec"]/EC_SALINITY_STRESS*100, 200)

            def bar_col(pct):
                return "#B71C1C" if pct>100 else ("#E65100" if pct>60 else "#2E7D32")

            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=["Cadmium","Lead","EC"],
                y=[cd_pct,pb_pct,ec_pct],
                marker_color=[bar_col(cd_pct),bar_col(pb_pct),bar_col(ec_pct)],
                text=[f"{inp['cd']:.3f}<br>mg/kg",f"{inp['pb']:.0f}<br>mg/kg",f"{inp['ec']:.1f}<br>dS/m"],
                textposition="inside",textfont=dict(color="white",size=11,family="DM Sans"),width=0.5,
            ))
            fig_b.add_hline(y=100, line_dash="dash", line_color="#999",
                            annotation_text="Safety threshold = 100%")
            fig_b.update_layout(
                title=dict(text="Contamination Levels (% of EU limit)",
                           font=dict(size=13,color="#111827")),
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                height=280,margin=dict(t=40,b=20,l=0,r=0),
                font=dict(family="DM Sans"),showlegend=False,
                yaxis=dict(range=[0,max(170,cd_pct+20,pb_pct+20)],
                           gridcolor="#F0F0F0",title="% of threshold"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_b, use_container_width=True)

            if zone != "GREEN" and mons > 0:
                st.markdown(f"""
                <div class="card" style="margin-top:0">
                  <div class="card-label">Remediation Progress</div>
                  <div style="font-size:2rem;font-weight:800;color:var(--g800)">{mons}</div>
                  <div style="font-size:.78rem;color:var(--muted)">months remaining to safe zone</div>
                  <div style="font-size:.74rem;color:var(--muted);margin-top:6px;line-height:1.5">
                    Equivalent to <strong>{mons//6}</strong> planting cycles of the prescribed mix.
                    Each cycle, harvested biomass carrying absorbed metals is removed from the field,
                    progressively lowering soil cadmium concentration.
                  </div>
                  <div style="background:var(--border);border-radius:99px;height:6px;margin-top:14px">
                    <div style="background:var(--g800);border-radius:99px;height:6px;width:0%"></div>
                  </div>
                  <div style="font-size:.72rem;color:var(--muted);margin-top:5px">0% complete — baseline scan</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT — English & Arabic only
# ═══════════════════════════════════════════════════════════════════════════════
with t_chat:
    st.markdown("""
    <div class="ph">
      <div class="ph-badge">AI Assistant</div>
      <h1>Soil Expert Assistant</h1>
      <p>Ask anything about soil remediation, the Gabès project, or your field results —
         in <strong>English</strong> or <strong>Arabic</strong>.</p>
    </div>""", unsafe_allow_html=True)

    client = llm_client()

    ctx = ""
    if "result" in st.session_state and "inputs" in st.session_state:
        i2 = st.session_state["inputs"]
        r2 = st.session_state["result"]
        ctx = (
            f"\n\nUSER FIELD CONTEXT (reference when relevant): "
            f"Zone={i2.get('zone','?')}, Cd={i2.get('cd',0):.3f} mg/kg, "
            f"Pb={i2.get('pb',0):.1f} mg/kg, EC={i2.get('ec',0):.1f} dS/m, "
            f"pH={i2.get('ph',0):.1f}. "
            f"Prescribed plants: {str(r2.get('plant_mix',''))}."
        )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role":"system","content": CHATBOT_SYSTEM_PROMPT + ctx}
        ]

    for msg in st.session_state.chat_messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Type your question in English or Arabic…"):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role":m["role"],"content":m["content"]}
                          for m in st.session_state.chat_messages],
                stream=True, temperature=0.5,
            )
            response = st.write_stream(stream)
        st.session_state.chat_messages.append({"role":"assistant","content":response})

    if len(st.session_state.chat_messages) > 1:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_messages = [
                {"role":"system","content": CHATBOT_SYSTEM_PROMPT + ctx}
            ]
            st.rerun()