#!/usr/bin/env python3
# app_full_enhanced_v4_langchain.py

import os
import json
import time
import math
import traceback
import requests
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

# set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="Travel Disruption Advisor — Enhanced v4 (LangChain LLM)", layout="wide")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import pydeck as pdk

# Twilio optional
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# ---------------- Environment / config (LLM-focused)
# Replace previous AZURE/OpenAI env usage with these:
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")   # e.g. https://genailab.xxx.in
LLM_MODEL = os.getenv("LLM_MODEL", "")         # e.g. azure_ai/genailab-maas-DeepSeek-V3-0324
LLM_API_KEY = os.getenv("LLM_API_KEY", "")     # API key string for the model

# Keep other existing envs (flight api, email, twilio etc.)
FLIGHTAPI_KEY = os.getenv("FLIGHTAPI_KEY", "")
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
TWILIO_SID = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
DEFAULT_NOTIFY_PHONE = os.getenv("DEFAULT_NOTIFY_PHONE", "")
AUTH_USERS_JSON = os.getenv("AUTH_USERS", "")  # '{"user@example.com":"<sha256hex>"}'
AUTH_SALT = os.getenv("AUTH_SALT", "")

# ---------------- Metrics collector ----------------
from collections import defaultdict
from time import perf_counter

class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self._timers = {}

    def increment(self, name, amount=1):
        self.counters[name] += amount

    def gauge(self, name, value):
        self.gauges[name] = value

    def record_latency(self, name, seconds):
        if seconds is None:
            return
        try:
            self.histograms[name].append(float(seconds))
        except Exception:
            pass

    def start_timer(self, name):
        self._timers[name] = perf_counter()

    def stop_timer(self, name):
        start = self._timers.pop(name, None)
        if start is None:
            return None
        elapsed = perf_counter() - start
        self.record_latency(name, elapsed)
        return elapsed

    def get_metrics(self):
        hist_stats = {}
        for k, arr in self.histograms.items():
            if len(arr) == 0:
                hist_stats[k] = {"count": 0, "min": None, "max": None, "avg": None}
            else:
                hist_stats[k] = {"count": len(arr), "min": min(arr), "max": max(arr), "avg": sum(arr) / len(arr)}
        return {"counters": dict(self.counters), "gauges": dict(self.gauges), "histograms": hist_stats}

metrics = MetricsCollector()

# ---------------- Load global IATA coords (OpenFlights) ----------------
@st.cache_data(show_spinner=False)
def load_global_iata_coords():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = [
        "AirportID","Name","City","Country","IATA","ICAO",
        "Latitude","Longitude","Altitude","Timezone","DST","Tz","Type","Source"
    ]
    try:
        df = pd.read_csv(url, header=None, names=cols)
        df = df[df["IATA"].notnull() & (df["IATA"] != "\\N")]
        iata_dict = {
            row["IATA"].upper(): (float(row["Latitude"]), float(row["Longitude"]))
            for _, row in df.iterrows()
            if row["IATA"] and row["Latitude"] and row["Longitude"]
        }
        return iata_dict
    except Exception:
        # fallback minimal set
        return {
            "DEL": (28.5562, 77.1000),
            "BLR": (13.1986, 77.7066),
            "BOM": (19.0896, 72.8656),
            "MAA": (12.9959, 80.1690),
            "HYD": (17.2403, 78.4294),
            "AMD": (23.0776, 72.6347),
            "COK": (10.1520, 76.4019),
        }

IATA_TO_COORDS = load_global_iata_coords()

# ---------------- Utility functions ----------------
def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        try:
            return {'text': resp.text}
        except Exception:
            return {'error':'unknown response'}

def sha256_hash(password: str, salt: str = AUTH_SALT) -> str:
    return hashlib.sha256((salt + (password or "")).encode('utf-8')).hexdigest()

def parse_latlon_input(s: str) -> Optional[Tuple[float,float]]:
    try:
        if not s:
            return None
        if "," in s:
            a,b = s.split(",",1); a=a.strip(); b=b.strip()
            if all(c.replace(".","",1).replace("-","",1).isdigit() for c in [a,b]):
                return float(a), float(b)
    except Exception:
        pass
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(min(1, math.sqrt(a)))

# ---------------- Auth ----------------
def load_auth_users()->Dict[str,str]:
    if not AUTH_USERS_JSON:
        return {}
    try:
        return json.loads(AUTH_USERS_JSON)
    except Exception:
        return {}

AUTH_USERS = load_auth_users()
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
st.session_state["user_email"] = None
st.session_state["history"] = []
st.session_state["chat_history"] = []

def authenticate_user(email: str, password: str) -> bool:
    if not AUTH_USERS:
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email or "demo@example.com"
        return True
    if email in AUTH_USERS and sha256_hash(password) == AUTH_USERS[email]:
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email
        return True
    return False

def logout_user():
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    try:
        st.experimental_rerun()
    except Exception:
        return

# ---------------- Flight API integration ----------------
def flightapi_search_oneway(origin_iata: str, dest_iata: str, date_iso: Optional[str] = None, mock: bool = False) -> Dict[str, Any]:
    if mock or not FLIGHTAPI_KEY:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        sample = []
        for i in range(4):
            depart = now + timedelta(hours=1 + i*2)
            arrive = depart + timedelta(hours=2 + i)
            sample.append({
                "flight_iata": f"AI{201 + i}",
                "airline": "DemoAir",
                "departure": {"iata": origin_iata, "scheduled": depart.isoformat()},
                "arrival": {"iata": dest_iata, "scheduled": arrive.isoformat()},
                "departure_time": depart.isoformat(),
                "arrival_time": arrive.isoformat(),
                "duration_min": int((arrive-depart).total_seconds()//60),
                "status": "scheduled" if i%3 else "en-route" if i%2 else "landed"
            })
        return {"data": sample}
    try:
        base = "https://www.flightapi.io"
        if date_iso:
            url = f"{base}/onewaytrip/{FLIGHTAPI_KEY}/{origin_iata}/{dest_iata}/{date_iso}"
        else:
            url = f"{base}/onewaytrip/{FLIGHTAPI_KEY}/{origin_iata}/{dest_iata}"
        r = requests.get(url, timeout=12)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def flightapi_search_roundtrip(origin_iata: str, dest_iata: str, dep_date_iso: str, ret_date_iso: str, mock: bool = False) -> Dict[str, Any]:
    if mock or not FLIGHTAPI_KEY:
        out1 = flightapi_search_oneway(origin_iata, dest_iata, dep_date_iso, mock=True)
        out2 = flightapi_search_oneway(dest_iata, origin_iata, ret_date_iso, mock=True)
        for f in out1["data"]:
            f["segment"] = "outbound"
        for f in out2["data"]:
            f["segment"] = "return"
        return {"data": out1["data"] + out2["data"]}
    try:
        base = "https://www.flightapi.io"
        url = f"{base}/roundtrip/{FLIGHTAPI_KEY}/{origin_iata}/{dest_iata}/{dep_date_iso}/{ret_date_iso}"
        r = requests.get(url, timeout=14)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def classify_flight_status(flight_record: Dict[str, Any]) -> str:
    status_raw = str(flight_record.get("status") or flight_record.get("flight_status") or "").lower()
    if any(k in status_raw for k in ["en-route", "airborne", "airborn", "airbourne"]):
        return "Airborne"
    if any(k in status_raw for k in ["scheduled", "booked", "scheduled to depart"]):
        return "Scheduled"
    if any(k in status_raw for k in ["landed", "in_ground", "land", "arrived", "on ground"]):
        return "In Ground"
    try:
        arr = flight_record.get("arrival_time") or (flight_record.get("arrival") or {}).get("scheduled") or (flight_record.get("arrival") or {}).get("estimated")
        if arr:
            arr_ts = datetime.fromisoformat(arr).replace(tzinfo=timezone.utc).timestamp()
            if arr_ts < datetime.utcnow().replace(tzinfo=timezone.utc).timestamp():
                return "In Ground"
    except Exception:
        pass
    return "Scheduled"

def flights_to_dataframe(payload: Dict[str, Any], filter_iata: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for f in payload.get("data", []):
        flight_iata = f.get("flight_iata") or f.get("flight") or f.get("flight_number") or "N/A"
        airline = f.get("airline") if isinstance(f.get("airline"), str) else (f.get("airline", {}).get("name") if isinstance(f.get("airline"), dict) else "N/A")
        dep = f.get("departure") or {}
        arr = f.get("arrival") or {}
        dep_time = f.get("departure_time") or dep.get("scheduled") or dep.get("estimated") or dep.get("actual") or ""
        arr_time = f.get("arrival_time") or arr.get("scheduled") or arr.get("estimated") or arr.get("actual") or ""
        src = (dep.get("iata") or dep.get("icao") or dep.get("city") or f.get("source") or f.get("from") or "") if isinstance(dep, dict) else (f.get("source") or "")
        dst = (arr.get("iata") or arr.get("icao") or arr.get("city") or f.get("destination") or f.get("to") or "") if isinstance(arr, dict) else (f.get("destination") or "")
        duration_min = f.get("duration_min") or f.get("flight_time_minutes") or f.get("duration") or None
        if duration_min:
            try:
                duration_min_int = int(duration_min)
                duration_text = f"{duration_min_int//60}h {duration_min_int%60}m"
            except Exception:
                duration_text = str(duration_min)
        else:
            duration_text = "unknown"
        status_label = classify_flight_status(f)
        rows.append({
            "Flight": flight_iata,
            "Airline": airline,
            "Departure": dep_time,
            "Arrival": arr_time,
            "Source": src,
            "Destination": dst,
            "Duration": duration_text,
            "Duration_min": duration_min,
            "Status": status_label,
            "raw": f
        })
    cols = ["Flight","Airline","Departure","Arrival","Source","Destination","Duration","Duration_min","Status","raw"]
    if rows:
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
    else:
        df = pd.DataFrame([], columns=cols)

    if filter_iata:
        fi = filter_iata.strip().upper()
        if "Source" in df.columns and "Destination" in df.columns:
            srcs = df["Source"].fillna("").astype(str).str.upper()
            dsts = df["Destination"].fillna("").astype(str).str.upper()
            df = df[(srcs == fi) | (dsts == fi)].reset_index(drop=True)
        else:
            df = pd.DataFrame([], columns=cols)
    return df

# ---------------- OpenSky nearby aircraft ----------------
def get_nearby_aircraft(lat: float, lon: float, radius_km: float = 200.0) -> List[Dict[str, Any]]:
    try:
        url = "https://opensky-network.org/api/states/all"
        r = requests.get(url, timeout=12)
        j = safe_json(r)
        if "states" not in j:
            return []
        out = []
        for s in j["states"]:
            callsign = s[1].strip() if s[1] else ""
            lon2 = s[5]; lat2 = s[6]; vel = s[9] if len(s) > 9 else None
            if lat2 is None or lon2 is None:
                continue
            d = haversine_km(lat, lon, lat2, lon2)
            if d <= radius_km:
                out.append({"callsign": callsign or "N/A", "lat": lat2, "lon": lon2, "distance_km": round(d,1), "velocity_m_s": vel or 0.0})
        out = sorted(out, key=lambda x: x["distance_km"])
        return out[:200]
    except Exception:
        return []

def get_opensky_between(lat1, lon1, lat2, lon2):
    try:
        lamin = min(lat1, lat2) - 1.0; lamax = max(lat1, lat2) + 1.0
        lomin = min(lon1, lon2) - 1.0; lomax = max(lon1, lon2) + 1.0
        url = "https://opensky-network.org/api/states/all"
        r = requests.get(url, params={"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax}, timeout=12)
        j = safe_json(r)
        out = []
        if "states" not in j:
            return out
        for s in j["states"]:
            callsign = (s[1] or "").strip()
            lon = s[5]; lat = s[6]; vel = s[9] if len(s) > 9 else None
            if lat is None or lon is None:
                continue
            out.append({"callsign": callsign or "N/A", "lat": lat, "lon": lon, "velocity_m_s": vel or 0.0})
        return out
    except Exception:
        return []

# ---------------- LLM Chat (LangChain ChatOpenAI) ----------------
# Use the langchain_openai ChatOpenAI wrapper provided by the user:
llm_client = None
LLM_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    if LLM_BASE_URL and LLM_MODEL and LLM_API_KEY:
        try:
            # instantiate with default temperature; you can override later
            llm_client = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY, temperature=0.2)
            LLM_AVAILABLE = True
        except Exception as e:
            llm_client = None
            LLM_AVAILABLE = False
    else:
        LLM_AVAILABLE = False
except Exception:
    llm_client = None
    LLM_AVAILABLE = False

def llm_chat_reply(prompt: str, max_tokens:int=400, temperature:float=0.2) -> str:
    """
    Send prompt to the configured LangChain ChatOpenAI client.
    This function attempts a few different call styles to handle differences
    in ChatOpenAI wrapper implementations. Returns a string reply or an
    informative error message.
    """
    start = time.perf_counter()
    metrics.increment("llm_chat_calls", 1)
    if not LLM_AVAILABLE or llm_client is None:
        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
        return ("[LLM_ERROR] LLM not configured. Set LLM_BASE_URL+LLM_MODEL+LLM_API_KEY in environment.")
    try:
        # Try common call patterns and extract text robustly.
        # 1) Try __call__ (many LangChain LLM wrappers implement __call__)
        try:
            resp = llm_client(prompt)  # might return a string or a Response-like object
            # If it's string-like, return directly
            if isinstance(resp, str):
                metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                return resp
            # If it's a dict-like with text
            if isinstance(resp, dict):
                # try common keys
                for key in ("text","content","message","output"):
                    if key in resp and isinstance(resp[key], str):
                        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                        return resp[key]
            # fallthrough to other extraction below
        except Exception:
            # ignore and try other methods
            pass

        # 2) Try .generate (LangChain often has generate or generate_messages)
        try:
            if hasattr(llm_client, "generate"):
                # Some implementations accept list of messages
                try:
                    gen = llm_client.generate([{"role":"user","content":prompt}])
                except Exception:
                    # some expect raw prompt
                    gen = llm_client.generate(prompt)
                # Extract text from typical shapes
                if hasattr(gen, "generations"):  # LangChain like object
                    gens = gen.generations
                    if isinstance(gens, list) and len(gens) > 0 and isinstance(gens[0], list) and len(gens[0]) > 0:
                        text = getattr(gens[0][0], "text", None) or str(gens[0][0])
                        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                        return text
                # fallback: stringify
                metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                return str(gen)
        except Exception:
            pass

        # 3) Try .predict or .predict_messages
        try:
            if hasattr(llm_client, "predict"):
                pred = llm_client.predict(prompt)
                if isinstance(pred, str):
                    metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                    return pred
            if hasattr(llm_client, "predict_messages"):
                pm = llm_client.predict_messages([{"role":"user","content":prompt}])
                if isinstance(pm, str):
                    metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                    return pm
                # else try to parse message object
                if hasattr(pm, "content"):
                    metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                    return pm.content
        except Exception:
            pass

        # 4) As last resort, try raw request to base_url endpoint using provided api key
        # This is conservative; depends on provider accepting this route.
        try:
            # Attempt a simple POST to base_url path /openai/deployments/{model}/chat/completions (Azure-like)
            # or /v1/chat/completions (OpenAI-like). We attempt both patterns based on base URL shape.
            headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": LLM_MODEL, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens, "temperature": temperature}
            # prefer base_url as-is (user configured)
            r = requests.post(LLM_BASE_URL.rstrip("/") + "/v1/chat/completions", headers=headers, json=payload, timeout=30)
            j = safe_json(r)
            if isinstance(j, dict) and "choices" in j and len(j["choices"]) > 0:
                content = j["choices"][0].get("message", {}).get("content") or j["choices"][0].get("text")
                if content:
                    metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                    return str(content)
            # try Azure style (deployments path)
            try:
                url = LLM_BASE_URL.rstrip("/") + f"/openai/deployments/{LLM_MODEL}/chat/completions"
                headers_az = {"api-key": LLM_API_KEY, "Content-Type": "application/json"}
                payload_az = {"messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens, "temperature": temperature}
                r2 = requests.post(url, headers=headers_az, json=payload_az, timeout=30)
                j2 = safe_json(r2)
                if isinstance(j2, dict) and "choices" in j2 and len(j2["choices"]) > 0:
                    content2 = j2["choices"][0].get("message", {}).get("content") or j2["choices"][0].get("text")
                    if content2:
                        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
                        return str(content2)
            except Exception:
                pass
        except Exception:
            pass

        # If we got here, none of the methods returned a successful reply
        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
        metrics.increment("llm_chat_failures", 1)
        return "[LLM_ERROR] Could not get response from LLM client. Check LLM_BASE_URL, LLM_MODEL, LLM_API_KEY and whether langchain_openai.ChatOpenAI is compatible."

    except Exception as e:
        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start)
        metrics.increment("llm_chat_exceptions", 1)
        return f"[LLM_EXCEPTION] {str(e)}"

# ---------------- LLM guardrail & metrics ----------------
def guardrail_check(user_text: str) -> Tuple[bool,str]:
    banned_keywords = ["kill", "bomb", "attack", "murder", "suicide", "harm", "weapon", "explosive", "terror", "threat"]
    low = (user_text or "").lower()
    for kw in banned_keywords:
        if kw in low:
            return False, f"User input contains banned keyword: {kw}"
    return True, "ok"

def tokenize_simple(text: str) -> List[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in (text or "")).lower().split() if t]

def bleu_like(reference: str, candidate: str) -> float:
    r_tok = tokenize_simple(reference); c_tok = tokenize_simple(candidate)
    if not c_tok:
        return 0.0
    overlap = sum(1 for t in c_tok if t in r_tok)
    return overlap / len(c_tok)

def rouge_like(reference: str, candidate: str) -> float:
    r_tok = tokenize_simple(reference); c_tok = tokenize_simple(candidate)
    if not r_tok:
        return 0.0
    overlap = sum(1 for t in r_tok if t in c_tok)
    return overlap / len(r_tok)

def meteor_like(reference: str, candidate: str) -> float:
    r_tok = tokenize_simple(reference); c_tok = tokenize_simple(candidate)
    if not c_tok or not r_tok:
        return 0.0
    precision = sum(1 for t in c_tok if t in r_tok) / len(c_tok)
    recall = sum(1 for t in r_tok if t in c_tok) / len(r_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_llm_metrics(reference: str, candidate: str) -> Dict[str,float]:
    b = bleu_like(reference, candidate); r = rouge_like(reference, candidate); m = meteor_like(reference, candidate)
    ref_len = len(tokenize_simple(reference)); cand_len = len(tokenize_simple(candidate))
    self_check = 1.0 if (ref_len == 0 or cand_len / max(ref_len,1) >= 0.4) else 0.0
    return {"BLEU": round(b,3), "ROUGE": round(r,3), "METEOR": round(m,3), "SelfCheck": self_check}

# ---------------- Notifications ----------------
def send_email(to: str, subject: str, body: str) -> Tuple[bool,str]:
    if not GMAIL_APP_PASSWORD or not GMAIL_SENDER:
        return False, "Email config missing (GMAIL_SENDER/GMAIL_APP_PASSWORD)"
    try:
        import smtplib
        srv = smtplib.SMTP("smtp.gmail.com", 587, timeout=20); srv.starttls()
        srv.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        msg = f"Subject: {subject}\n\n{body}"
        srv.sendmail(GMAIL_SENDER, [to], msg.encode("utf-8"))
        srv.quit()
        return True, "sent"
    except Exception as e:
        return False, str(e)

def send_sms_via_twilio(to: str, body: str) -> Tuple[bool,str]:
    if not TWILIO_AVAILABLE or not TWILIO_SID or not TWILIO_TOKEN or not TWILIO_FROM:
        return False, "Twilio not configured"
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=body, from_=TWILIO_FROM, to=to)
        return True, f"sid={msg.sid}"
    except Exception as e:
        return False, str(e)

# ---------------- UI: Login Page ----------------
def show_login_page():
    st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)
    st.title("🔒 Travel & Logistics Advisor — Login")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.write("Please log in to continue. If AUTH_USERS is not configured, demo-mode will be used.")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", value=st.session_state.get("login_email",""))
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
        if submitted:
            ok = authenticate_user(email.strip(), password)
            if ok:
                st.success(f"Logged in as {st.session_state.get('user_email')}")
                try:
                    st.experimental_rerun()
                except Exception:
                    return
            else:
                st.error("Invalid credentials. If AUTH_USERS not configured, app runs in demo mode; set AUTH_USERS to enable true auth.")

# ---------------- UI: Dashboard ----------------
def show_dashboard():
    st.sidebar.header(f"User: {st.session_state.get('user_email','')}")
    if st.sidebar.button("Log out"):
        logout_user()

    st.title("🚨 Travel & Logistics Real-Time Advisor — Dashboard (v4)")

    st.sidebar.markdown("### Flight Search")
    origin = st.sidebar.text_input("Origin (IATA / city / 'lat,lon')", value="DEL")
    destination = st.sidebar.text_input("Destination (IATA / city / 'lat,lon')", value="BLR")
    trip_type = st.sidebar.selectbox("Trip type", ["One-way", "Round-trip"])
    dep_date = st.sidebar.date_input("Departure date (optional)", value=None)
    ret_date = None
    if trip_type == "Round-trip":
        ret_date = st.sidebar.date_input("Return date (optional)", value=None)
    flight_search_btn = st.sidebar.button("Search flights")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Logistics / Routing")
    route_from = st.sidebar.text_input("Pickup (address or lat,lon)", value="Indira Gandhi International Airport, Delhi")
    route_to = st.sidebar.text_input("Dropoff (address or lat,lon)", value="Kempegowda International Airport, Bangalore")
    route_btn = st.sidebar.button("Compute fastest route")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Notifications")
    notify_email = st.sidebar.text_input("Notify email", value=st.session_state.get("user_email","user@example.com"))
    notify_phone = st.sidebar.text_input("Notify phone (for SMS)", value=DEFAULT_NOTIFY_PHONE)
    send_notify_btn = st.sidebar.button("Send last summary (email+SMS)")

    # flight search processing
    selected_df = None
    flights_payload = None

    def iata_guess(s: str) -> Optional[str]:
        if not s:
            return None
        s2 = s.strip().upper()
        if len(s2) == 3 and s2.isalpha():
            return s2
        return None

    if flight_search_btn:
        metrics.start_timer("flight_search_seconds")
        mock_mode = not bool(FLIGHTAPI_KEY)
        o_iata = iata_guess(origin) or origin
        d_iata = iata_guess(destination) or destination
        dep_iso = dep_date.isoformat() if dep_date else None
        if trip_type == "One-way":
            flights_payload = flightapi_search_oneway(o_iata, d_iata, date_iso=dep_iso, mock=mock_mode)
        else:
            ret_iso = ret_date.isoformat() if ret_date else None
            flights_payload = flightapi_search_roundtrip(o_iata, d_iata, dep_date_iso=dep_iso, ret_date_iso=ret_iso, mock=mock_mode)
        metrics.stop_timer("flight_search_seconds"); metrics.increment("flight_searches", 1)

        st.sidebar.markdown("**Flight API raw payload (DEBUG)**")
        try:
            st.sidebar.write({k: (str(v)[:1000] + "..." if isinstance(v,(dict,list)) and len(str(v))>1000 else v) for k,v in (flights_payload.items() if isinstance(flights_payload, dict) else {"payload": str(flights_payload)}.items())})
        except Exception:
            st.sidebar.write("Couldn't display raw payload.")

        if not flights_payload or (isinstance(flights_payload, dict) and not flights_payload.get("data")):
            st.warning("Flight API returned no data. Showing mock sample flights for debugging (check FLIGHTAPI_KEY / network).")
            flights_payload = flightapi_search_oneway(o_iata, d_iata, date_iso=dep_iso, mock=True)

        selected_df = flights_to_dataframe(flights_payload, filter_iata=(iata_guess(origin) or iata_guess(destination)))
        st.session_state["history"].insert(0, {"ts": now_iso(), "type":"flight_search", "origin": origin, "destination": destination, "trip": trip_type, "rows": len(selected_df)})
        st.session_state["history"] = st.session_state["history"][:200]

    # Style for full-width map/results
    st.markdown("""
    <style>
      .full-width-map .stDeckGlJson { height: 650px !important; width: 100% !important; }
      .full-width-results .stDataFrame { width: 100% !important; }
      .block-container { padding-left: 1rem; padding-right: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    # Map (full width)
    st.subheader("Map (routes & flights)")
    map_container = st.container()
    with map_container:
        st.markdown('<div class="full-width-map">', unsafe_allow_html=True)
        markers = []; paths = []

        def geocode_label(label: str):
            xy = parse_latlon_input(label)
            if xy:
                return xy[0], xy[1], label
            lab = (label or "").strip().upper()
            if lab in IATA_TO_COORDS:
                return IATA_TO_COORDS[lab][0], IATA_TO_COORDS[lab][1], lab
            return None

        ocoord = None; dcoord = None
        if selected_df is not None and not selected_df.empty:
            first = selected_df.iloc[0]
            ocoord = geocode_label(first.get("Source")) or geocode_label(origin)
            dcoord = geocode_label(first.get("Destination")) or geocode_label(destination)
        else:
            ocoord = geocode_label(origin)
            dcoord = geocode_label(destination)

        if ocoord:
            markers.append({"name":"Origin","lat":ocoord[0],"lon":ocoord[1],"color":[0,200,0]})
        if dcoord:
            markers.append({"name":"Destination","lat":dcoord[0],"lon":dcoord[1],"color":[200,0,0]})
        if ocoord and dcoord:
            try:
                acs = get_opensky_between(ocoord[0], ocoord[1], dcoord[0], dcoord[1])
                for a in acs:
                    markers.append({"name": f"{a.get('callsign','N/A')} ({round(a.get('velocity_m_s',0.0),1)} m/s)","lat": a['lat'],"lon": a['lon'],"color":[255,215,0]})
                paths.append({"path":[[ocoord[1], ocoord[0]], [dcoord[1], dcoord[0]]], "name":"route_line"})
            except Exception:
                pass

        if ocoord:
            try:
                ac = get_nearby_aircraft(ocoord[0], ocoord[1], radius_km=500.0)
                for a in ac[:80]:
                    markers.append({"name": f"{a.get('callsign','N/A')} ({a.get('distance_km','?')} km)","lat": a['lat'],"lon": a['lon'],"color":[255,215,0]})
            except Exception:
                pass
        if dcoord:
            try:
                ac = get_nearby_aircraft(dcoord[0], dcoord[1], radius_km=500.0)
                for a in ac[:80]:
                    markers.append({"name": f"{a.get('callsign','N/A')} ({a.get('distance_km','?')} km)","lat": a['lat'],"lon": a['lon'],"color":[255,215,0]})
            except Exception:
                pass

        if markers or paths:
            dfm = pd.DataFrame(markers) if markers else pd.DataFrame([], columns=["name","lat","lon","color"])
            layers = []
            if not dfm.empty:
                layers.append(pdk.Layer("ScatterplotLayer", dfm, get_position=["lon","lat"], get_fill_color="color", get_radius=6000, pickable=True))
            if paths:
                dfp = pd.DataFrame(paths)
                layers.append(pdk.Layer("PathLayer", dfp, get_path="path", get_width=4, get_color=[0,128,255], pickable=False))
            center_lat = dfm["lat"].mean() if not dfm.empty else 20.5937
            center_lon = dfm["lon"].mean() if not dfm.empty else 78.9629
            deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6), tooltip={"text":"{name}"})
            st.pydeck_chart(deck)
        else:
            st.info("No map items yet. Search flights or compute a route.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Flight Results (below map)
    st.markdown("---")
    st.subheader("Flight Results")
    results_container = st.container()
    with results_container:
        st.markdown('<div class="full-width-results">', unsafe_allow_html=True)
        if selected_df is not None:
            if selected_df.empty:
                st.info("No flights returned.")
            else:
                requested_cols = ["Flight","Airline","Departure","Arrival","Source","Destination","Duration","Status"]
                show_cols = [c for c in requested_cols if c in selected_df.columns]
                st.dataframe(selected_df[show_cols].reset_index(drop=True))
                csv = selected_df[show_cols].to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="flights.csv", mime="text/csv")
                try:
                    if "Duration_min" in selected_df.columns and selected_df["Duration_min"].notnull().any():
                        best = selected_df[selected_df["Duration_min"].notnull()].sort_values("Duration_min").iloc[0]
                        st.success(f"Suggested fastest flight: {best['Flight']} ({best['Duration']}) — Status: {best['Status']}")
                        metrics.increment("recommended_flights", 1)
                    else:
                        st.info("Could not determine fastest flight.")
                except Exception:
                    st.info("Could not determine fastest flight.")
        else:
            st.info("Search for flights to see results here.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat / Q&A
    st.markdown("---")
    st.subheader("Chat / Q&A (Ask for recommendations or itineraries)")
    user_msg = st.text_area("Your question (e.g., 'Which flight is fastest? Make a 2-day itinerary for BLR trip')", key="chat_input", height=120)
    if st.button("Ask LLM"):
        if not user_msg or user_msg.strip() == "":
            st.warning("Please write a question.")
        else:
            allowed, reason = guardrail_check(user_msg)
            if not allowed:
                st.error(f"Guardrail blocked the request: {reason}")
            else:
                context = {
                    "user": st.session_state.get("user_email"),
                    "last_search": {
                        "origin": origin,
                        "destination": destination,
                        "trip_type": trip_type
                    },
                    "note": "Do not provide illegal or violent advice."
                }

                prompt = f"""
                You are a Travel and Logistics Disruption Advisor.

                Context:
                {json.dumps(context)}

                User Disruption Query:
                {user_msg}

                Your role:
                - Identify, assess, and explain disruptions in travel or supply-chain logistics
                - Evaluate risks, operational impact, and cost/time consequences
                - Monitor real-time events and specify required data inputs or alerts
                - Recommend mitigation steps including rerouting, rebooking, alternative carriers, or contingency workflows
                - Coordinate responses to minimize delays, service interruption, and financial impact
                - Provide clear, actionable guidance and communication plans for customers, partners, and internal stakeholders
                - Help design resilient travel/logistics networks and contingency procedures

                When responding to a disruption scenario or user query, always include:

                1. Summary — what is happening + likely operational impact  
                2. Real-time Data Needs — what should be monitored  
                3. Mitigation Strategies — rerouting, rebooking, alternatives  
                4. Cost / Time / Operational Trade-offs — pros/cons of each path  
                5. Communication Plan — messaging for customers and partners  
                6. Assumptions — state explicitly  
                7. Clarifying Questions — only if essential  

                Be concise, operational, and highly actionable.
                """

                reply = llm_chat_reply(prompt)
                st.markdown("**LLM reply:**")
                st.write(reply)
                if selected_df is not None and not selected_df.empty:
                    ref_text = "\n".join(selected_df.head(5).apply(lambda r: f"{r['Flight']} from {r['Source']} to {r['Destination']} dep {r['Departure']}", axis=1).tolist())
                    m = compute_llm_metrics(ref_text, reply)
                    metrics_table = pd.DataFrame([m])
                    st.markdown("**LLM metrics (approx)**")
                    st.table(metrics_table)
                st.session_state["chat_history"].insert(0, {"ts": now_iso(), "q": user_msg, "a": reply})
                st.session_state["chat_history"] = st.session_state["chat_history"][:200]

    if st.checkbox("Show recent chat history"):
        for ch in st.session_state["chat_history"][:20]:
            st.write(f"{ch['ts']}: Q: {ch['q']}")
            st.write(f"A: {ch['a']}")
    st.markdown("---")

    # Notifications
    st.markdown("---")
    st.subheader("Notifications")
    if send_notify_btn:
        body = "Travel Disruption Advisor — Summary\n\n"
        if selected_df is not None and not selected_df.empty:
            body += "Top flights:\n" + selected_df.head(5).to_string() + "\n\n"
        ok, msg = send_email(notify_email, f"Advisor summary {origin}->{destination}", body)
        if ok:
            st.success("Email sent"); metrics.increment("emails_sent", 1)
        else:
            st.error("Email failed: " + msg)
        if notify_phone:
            ok2, m2 = send_sms_via_twilio(notify_phone, f"Advisor summary {origin}->{destination}")
            if ok2:
                st.success("SMS sent"); metrics.increment("sms_sent", 1)
            else:
                st.error("SMS failed: " + m2)

    # Bottom: metrics & history
    st.markdown("---")
    st.subheader("App Metrics & History")
    st.json(metrics.get_metrics())
    if st.checkbox("Show history (last 20)"):
        st.subheader("History")
        for h in st.session_state["history"][:20]:
            st.write(h)

# ---------------- Main ----------------
def main():
    if not st.session_state.get("authenticated", False):
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
