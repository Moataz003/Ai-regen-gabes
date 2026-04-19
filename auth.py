"""
auth.py ─── Gabès Regenerate AI ─── Authentication Module
"""
import hashlib
import json
import os
import streamlit as st

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _load() -> dict:
    if not os.path.exists(USERS_FILE):
        seed = {"admin": {"password": _hash("admin123"), "name": "Administrator", "role": "admin"}}
        _save(seed)
        return seed
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def authenticate(username: str, password: str):
    users = _load()
    u = users.get(username.strip().lower())
    if u and u["password"] == _hash(password):
        return u
    return None

def register(username: str, password: str, name: str) -> tuple:
    username = username.strip().lower()
    if not username or not password or not name:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load()
    if username in users:
        return False, "Username already exists."
    users[username] = {"password": _hash(password), "name": name, "role": "user"}
    _save(users)
    return True, "Account created successfully."

def is_logged_in() -> bool:
    return st.session_state.get("auth_user") is not None

def current_user() -> dict:
    return st.session_state.get("auth_user", {})

def logout():
    st.session_state.pop("auth_user", None)
    st.session_state.pop("auth_tab", None)

def show_auth_page():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }
html, body, .stApp { background: #F4FAF4 !important; }
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.main .block-container { padding: 0 !important; }

.auth-root { display: flex; min-height: 100vh; width: 100%; }
.auth-left {
    width: 48%;
    background: linear-gradient(155deg, #1B5E20 0%, #2E7D32 45%, #388E3C 100%);
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 52px 52px 44px; position: relative; overflow: hidden;
}
.auth-left::before {
    content: ''; position: absolute; width: 340px; height: 340px;
    border-radius: 50%; background: rgba(255,255,255,0.05); top: -80px; right: -100px;
}
.auth-left::after {
    content: ''; position: absolute; width: 220px; height: 220px;
    border-radius: 50%; background: rgba(255,255,255,0.04); bottom: 60px; left: -60px;
}
.auth-brand { position: relative; z-index: 1; }
.auth-brand-logo {
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 14px; padding: 10px 18px; margin-bottom: 52px;
}
.auth-brand-logo span { font-size: 1.3rem; }
.auth-brand-logo strong { font-size: 0.95rem; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.auth-headline { font-size: 2.4rem; font-weight: 800; color: #fff; line-height: 1.2; letter-spacing: -0.03em; margin: 0 0 18px; }
.auth-headline em { font-style: normal; color: #A5D6A7; }
.auth-sub { font-size: 0.97rem; color: rgba(255,255,255,0.72); line-height: 1.65; max-width: 340px; margin: 0; }
.auth-features { position: relative; z-index: 1; margin-top: 52px; }
.auth-feat-item { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.auth-feat-icon {
    width: 34px; height: 34px; border-radius: 10px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18);
    display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0;
}
.auth-feat-text { font-size: 0.85rem; color: rgba(255,255,255,0.82); font-weight: 500; }
.auth-footer-left {
    position: relative; z-index: 1; margin-top: auto; font-size: 0.75rem;
    color: rgba(255,255,255,0.4); padding-top: 32px; border-top: 1px solid rgba(255,255,255,0.1);
}
.auth-right { flex: 1; display: flex; align-items: center; justify-content: center; padding: 48px 40px; background: #F4FAF4; }
.auth-card {
    width: 100%; max-width: 420px; background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 20px; padding: 40px 36px 36px; box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.auth-card-header { margin-bottom: 28px; }
.auth-card-badge {
    display: inline-block; background: #E8F5E9; color: #2E7D32; font-size: 0.7rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    border-radius: 20px; padding: 3px 10px; margin-bottom: 10px;
}
.auth-card-title { font-size: 1.55rem; font-weight: 800; color: #111827; margin: 0 0 5px; letter-spacing: -0.025em; }
.auth-card-subtitle { font-size: 0.86rem; color: #6B7280; margin: 0; }
.auth-divider { display: flex; align-items: center; gap: 12px; margin: 20px 0 16px; font-size: 0.76rem; color: #9CA3AF; font-weight: 500; }
.auth-divider::before, .auth-divider::after { content: ''; flex: 1; height: 1px; background: #E5E7EB; }
.auth-hint { font-size: 0.76rem; color: #9CA3AF; text-align: center; margin-top: 20px; line-height: 1.6; }

div[data-testid="stTextInput"] input {
    border-radius: 10px !important; border: 1.5px solid #E5E7EB !important;
    padding: 11px 14px !important; font-size: 0.88rem !important; color: #111827 !important;
    background: #FAFAFA !important; transition: border-color 0.15s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #2E7D32 !important; background: #fff !important;
    box-shadow: 0 0 0 3px rgba(46,125,50,0.1) !important; outline: none !important;
}
div[data-testid="stTextInput"] label { font-size: 0.82rem !important; font-weight: 600 !important; color: #374151 !important; margin-bottom: 4px !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2E7D32, #388E3C) !important; color: white !important;
    border: none !important; border-radius: 11px !important; font-size: 0.9rem !important;
    font-weight: 700 !important; padding: 13px 20px !important; letter-spacing: 0.01em !important;
    box-shadow: 0 3px 12px rgba(46,125,50,0.28) !important; transition: all 0.18s !important; margin-top: 4px !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
    box-shadow: 0 5px 18px rgba(46,125,50,0.36) !important; transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0px) !important; }

div[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.84rem !important; margin-top: 8px !important; }
div[data-baseweb="tab-list"] { display: none !important; }

@media (max-width: 768px) { .auth-left { display: none; } .auth-right { padding: 32px 20px; } .auth-card { padding: 32px 24px; } }
</style>

<div class="auth-root">
  <div class="auth-left">
    <div class="auth-brand">
      <div class="auth-brand-logo"><span>🌿</span><strong>Gabès Regenerate AI</strong></div>
      <h1 class="auth-headline">Healing soil,<br><em>field by field.</em></h1>
      <p class="auth-sub">AI-powered soil remediation platform for the Gabès region. Analyze contamination, get prescriptions, and track remediation progress in real time.</p>
      <div class="auth-features" style="margin-top:40px">
        <div class="auth-feat-item"><div class="auth-feat-icon">🧪</div><span class="auth-feat-text">Heavy metal & salinity analysis</span></div>
        <div class="auth-feat-item"><div class="auth-feat-icon">🌱</div><span class="auth-feat-text">AI phytoremediation prescriptions</span></div>
        <div class="auth-feat-item"><div class="auth-feat-icon">💧</div><span class="auth-feat-text">Water-smart irrigation engine</span></div>
        <div class="auth-feat-item"><div class="auth-feat-icon">📜</div><span class="auth-feat-text">Verifiable soil passports with QR</span></div>
      </div>
    </div>
    <div class="auth-footer-left">SoilRevive Engine v5.2 · Gabès Regenerate AI · Tunisia</div>
  </div>
  <div class="auth-right">
    <div style="width:100%;max-width:420px">
      <div class="auth-card-header">
        <div class="auth-card-badge">Secure Access</div>
        <h2 class="auth-card-title">Welcome back</h2>
        <p class="auth-card-subtitle">Sign in to your account or create a new one.</p>
      </div>
""", unsafe_allow_html=True)

    # Tab state
    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "login"

    col_login, col_reg = st.columns(2)
    with col_login:
        if st.button("Sign In", key="tab_login_btn", use_container_width=True, type="primary" if st.session_state.auth_tab == "login" else "secondary"):
            st.session_state.auth_tab = "login"
            st.rerun()
    with col_reg:
        if st.button("Create Account", key="tab_reg_btn", use_container_width=True, type="primary" if st.session_state.auth_tab == "register" else "secondary"):
            st.session_state.auth_tab = "register"
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.session_state.auth_tab == "login":
        uname = st.text_input("Username", key="li_user", placeholder="your username")
        pwd = st.text_input("Password", key="li_pwd", placeholder="••••••••", type="password")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Sign In →", key="li_btn", use_container_width=True):
            if not uname or not pwd:
                st.error("Please fill in all fields.")
            else:
                user = authenticate(uname, pwd)
                if user:
                    st.session_state["auth_user"] = user
                    st.session_state["auth_username"] = uname.strip().lower()
                    st.success(f"Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
        st.markdown('<div class="auth-hint">Don\'t have an account? <span style="color:#2E7D32;font-weight:600;">Click "Create Account" above.</span></div>', unsafe_allow_html=True)
    else:
        r_name = st.text_input("Full Name", key="reg_name", placeholder="Ahmed Ben Ali")
        r_user = st.text_input("Username", key="reg_user", placeholder="choose a username")
        r_pwd = st.text_input("Password", key="reg_pwd", placeholder="min. 6 characters", type="password")
        r_pwd2 = st.text_input("Confirm Password", key="reg_pwd2", placeholder="repeat password", type="password")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Create Account →", key="reg_btn", use_container_width=True):
            if r_pwd != r_pwd2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register(r_user, r_pwd, r_name)
                if ok:
                    st.success(msg + " Please sign in.")
                    st.session_state.auth_tab = "login"
                    st.rerun()
                else:
                    st.error(msg)
        st.markdown('<div class="auth-hint">Already have an account? <span style="color:#2E7D32;font-weight:600;">Click "Sign In" above.</span></div>', unsafe_allow_html=True)

    st.markdown("</div></div></div>", unsafe_allow_html=True)