"""
AI-Based Smart Energy Management System (ESP32 + Streamlit + LangChain/Ollama)

Run:  streamlit run app.py

ESP32 should print one JSON object per line over serial (115200 baud), e.g.:
  {"voltage":229.8,"current":2.31,"power":512.4,"pf":0.93,"temp":29.5}
(Optional) The app can send "RELAY_ON" / "RELAY_OFF" lines to switch a relay.

Arduino sketch outline:
  void loop(){ 
    Serial.printf("{\"voltage\":%.1f,\"current\":%.2f,\"power\":%.1f,\"pf\":%.2f,\"temp\":%.1f}\n", v,i,p,pf,t);
    if(Serial.available()){ String c=Serial.readStringUntil('\n'); c.trim();
      if(c=="RELAY_ON") digitalWrite(RELAY_PIN,HIGH); if(c=="RELAY_OFF") digitalWrite(RELAY_PIN,LOW);} 
    delay(1000);
  }
"""
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    import serial  # pyserial
    import serial.tools.list_ports as list_ports
except Exception:  # pyserial missing -> simulation only
    serial = None
    list_ports = None

st.set_page_config(page_title="Smart Energy AI", page_icon="⚡", layout="wide")

COLS = ["timestamp", "voltage", "current", "power", "pf", "temp"]
MAX_ROWS = 3000
PEAK_HOURS = range(18, 23)
rng = np.random.default_rng()


# ----------------------------------------------------------------- data layer
def simulate_reading(ts: datetime) -> dict:
    """Realistic household load: base + morning/evening peaks + random spikes."""
    h = ts.hour + ts.minute / 60
    base = 250
    morning = 450 * np.exp(-((h - 7.5) ** 2) / 2)
    evening = 900 * np.exp(-((h - 20) ** 2) / 4)
    ac = 700 if 13 <= h <= 17 else 0
    spike = 1500 if rng.random() < 0.01 else 0  # occasional anomaly
    power = max(40, base + morning + evening + ac + rng.normal(0, 40) + spike)
    voltage = 230 + rng.normal(0, 2.5)
    pf = float(np.clip(rng.normal(0.92, 0.03), 0.6, 1.0))
    current = power / (voltage * pf)
    temp = 27 + 3 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.3)
    return dict(timestamp=ts, voltage=voltage, current=current, power=power, pf=pf, temp=temp)


def seed_history(minutes=240) -> pd.DataFrame:
    now = datetime.now()
    rows = [simulate_reading(now - timedelta(minutes=m)) for m in range(minutes, 0, -1)]
    return pd.DataFrame(rows, columns=COLS)


@st.cache_resource
def get_serial(port: str, baud: int):
    return serial.Serial(port, baud, timeout=0.5)


def read_serial(port: str, baud: int) -> list[dict]:
    ser = get_serial(port, baud)
    out = []
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        try:
            d = json.loads(line)
            v, i = float(d.get("voltage", 230)), float(d.get("current", 0))
            p = float(d.get("power", v * i))
            out.append(dict(timestamp=datetime.now(), voltage=v, current=i, power=p,
                            pf=float(d.get("pf", 0.9)), temp=float(d.get("temp", np.nan))))
        except (ValueError, json.JSONDecodeError):
            continue
    return out


def send_command(port: str, baud: int, cmd: str):
    get_serial(port, baud).write((cmd + "\n").encode())


def ingest(source: str, port: str, baud: int):
    if "df" not in st.session_state:
        st.session_state.df = seed_history() if source == "Simulated" else pd.DataFrame(columns=COLS)
    new = [simulate_reading(datetime.now())] if source == "Simulated" else read_serial(port, baud)
    if new:
        df = pd.concat([st.session_state.df, pd.DataFrame(new, columns=COLS)], ignore_index=True)
        st.session_state.df = df.tail(MAX_ROWS)


# ------------------------------------------------------------------ analytics
def enrich(df: pd.DataFrame, tariff: float, peak_mult: float) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    dt_h = d["timestamp"].diff().dt.total_seconds().fillna(0).clip(0, 600) / 3600
    d["kwh"] = d["power"] * dt_h / 1000
    d["rate"] = np.where(d["timestamp"].dt.hour.isin(PEAK_HOURS), tariff * peak_mult, tariff)
    d["cost"] = d["kwh"] * d["rate"]
    mu = d["power"].rolling(30, min_periods=10).mean()
    sd = d["power"].rolling(30, min_periods=10).std().replace(0, np.nan)
    d["z"] = ((d["power"] - mu) / sd).fillna(0)
    d["anomaly"] = d["z"].abs() > 3
    return d


def forecast(d: pd.DataFrame, steps=30) -> pd.DataFrame:
    """Short-term forecast: EWMA level + damped linear trend (numpy)."""
    y = d["power"].tail(60).to_numpy()
    if len(y) < 10:
        return pd.DataFrame()
    level = pd.Series(y).ewm(span=10).mean().iloc[-1]
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    step = (d["timestamp"].diff().dt.total_seconds().tail(30).median() or 60)
    fut = [level + slope * k * 0.5 ** (k / 15) for k in range(1, steps + 1)]
    ts = [d["timestamp"].iloc[-1] + timedelta(seconds=step * k) for k in range(1, steps + 1)]
    return pd.DataFrame({"timestamp": ts, "power": np.maximum(fut, 0)})


def summarize(d: pd.DataFrame, budget_kwh: float) -> dict:
    last = d.iloc[-1]
    peak_share = d.loc[d["timestamp"].dt.hour.isin(PEAK_HOURS), "kwh"].sum() / max(d["kwh"].sum(), 1e-9)
    return {
        "current_power_w": round(last.power, 1),
        "avg_power_w": round(d.power.mean(), 1),
        "max_power_w": round(d.power.max(), 1),
        "energy_kwh": round(d.kwh.sum(), 3),
        "cost": round(d.cost.sum(), 2),
        "avg_voltage": round(d.voltage.mean(), 1),
        "avg_power_factor": round(d.pf.mean(), 2),
        "avg_temp_c": round(d.temp.mean(), 1) if d.temp.notna().any() else None,
        "peak_hour_share_pct": round(100 * peak_share, 1),
        "anomalies": int(d.anomaly.sum()),
        "budget_used_pct": round(100 * d.kwh.sum() / budget_kwh, 1),
    }


def rule_based_tips(s: dict) -> list[str]:
    tips = []
    if s["peak_hour_share_pct"] > 35:
        tips.append(f"{s['peak_hour_share_pct']}% of energy is used in peak hours (6-11 PM). Shift laundry, "
                    "water heating and EV charging to off-peak times.")
    if s["avg_power_factor"] < 0.9:
        tips.append("Low power factor detected. Consider capacitor correction or replacing inductive loads.")
    if s["anomalies"] > 0:
        tips.append(f"{s['anomalies']} abnormal power spikes detected. Check for faulty or always-on appliances.")
    if s["budget_used_pct"] > 80:
        tips.append("You have used over 80% of your energy budget. Reduce AC/heater usage.")
    if s["avg_voltage"] > 245 or s["avg_voltage"] < 215:
        tips.append("Voltage is outside the safe range. Consider a stabilizer or surge protection.")
    return tips or ["Consumption looks healthy. Keep it up!"]


# --------------------------------------------------------------------- AI/LLM
def ollama_up(url: str) -> bool:
    try:
        return requests.get(url, timeout=1.5).ok
    except requests.RequestException:
        return False


def ask_ai(question: str, summary: dict, model: str, url: str) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert energy management assistant for a smart home monitored by an ESP32. "
                   "Use ONLY this live data summary and give concise, practical, numeric advice.\n"
                   "DATA: {summary}"),
        ("human", "{question}"),
    ])
    chain = prompt | ChatOllama(model=model, base_url=url, temperature=0.3) | StrOutputParser()
    return chain.invoke({"summary": json.dumps(summary), "question": question})


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚡ Settings")
    source = st.radio("Data source", ["Simulated", "ESP32 (Serial)"])
    port, baud = None, 115200
    if source.startswith("ESP32"):
        if serial is None:
            st.error("pyserial not installed.")
        ports = [p.device for p in list_ports.comports()] if list_ports else []
        port = st.selectbox("COM port", ports) if ports else st.text_input("COM port", "COM3")
        baud = st.selectbox("Baud rate", [9600, 115200, 921600], index=1)
        c1, c2 = st.columns(2)
        if c1.button("Relay ON"):
            send_command(port, baud, "RELAY_ON")
        if c2.button("Relay OFF"):
            send_command(port, baud, "RELAY_OFF")
    refresh = st.slider("Refresh (sec)", 1, 10, 2)
    live = st.toggle("Live updates", True)
    st.subheader("Tariff")
    tariff = st.number_input("Rate per kWh", 0.0, 100.0, 8.0, 0.5)
    peak_mult = st.slider("Peak-hour multiplier", 1.0, 3.0, 1.5, 0.1)
    budget = st.number_input("Energy budget (kWh)", 0.1, 1000.0, 10.0)
    st.subheader("AI (Ollama)")
    model = st.text_input("Model", "llama3.2")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
    if st.button("Reset data"):
        st.session_state.pop("df", None)
        st.rerun()
    if "df" in st.session_state:
        st.download_button("Download CSV", st.session_state.df.to_csv(index=False), "energy_data.csv")

st.title("⚡ AI Smart Energy Management System")
st.caption(f"Source: {source}")


# ------------------------------------------------------------------ dashboard
@st.fragment(run_every=refresh if live else None)
def dashboard():
    try:
        ingest(source, port, baud)
    except Exception as e:
        st.error(f"Data error: {e}")
        return
    raw = st.session_state.df
    if len(raw) < 3:
        st.info("Waiting for data from ESP32…")
        return
    d = enrich(raw, tariff, peak_mult)
    s = summarize(d, budget)
    st.session_state.summary = s

    k = st.columns(5)
    k[0].metric("Power", f"{s['current_power_w']} W", f"{s['current_power_w'] - s['avg_power_w']:+.0f} vs avg")
    k[1].metric("Energy", f"{s['energy_kwh']} kWh")
    k[2].metric("Cost", f"{s['cost']}")
    k[3].metric("Voltage", f"{d.voltage.iloc[-1]:.1f} V")
    k[4].metric("Power factor", f"{d.pf.iloc[-1]:.2f}")
    st.progress(min(s["budget_used_pct"] / 100, 1.0), text=f"Budget used: {s['budget_used_pct']}%")

    t1, t2, t3, t4 = st.tabs(["📈 Live", "🔍 Analysis", "💡 Insights", "🤖 AI Assistant"])

    with t1:
        fig = go.Figure()
        fig.add_scatter(x=d.timestamp, y=d.power, name="Power (W)", line=dict(color="#1f77b4"))
        an = d[d.anomaly]
        fig.add_scatter(x=an.timestamp, y=an.power, mode="markers", name="Anomaly",
                        marker=dict(color="red", size=9, symbol="x"))
        fc = forecast(d)
        if not fc.empty:
            fig.add_scatter(x=fc.timestamp, y=fc.power, name="Forecast",
                            line=dict(color="orange", dash="dash"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title="Power & forecast")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(d, x="timestamp", y="voltage", title="Voltage (V)").update_layout(height=260),
                        use_container_width=True)
        c2.plotly_chart(px.line(d, x="timestamp", y="current", title="Current (A)").update_layout(height=260),
                        use_container_width=True)

    with t2:
        hourly = d.set_index("timestamp").resample("1h").agg({"kwh": "sum", "cost": "sum"}).reset_index()
        st.plotly_chart(px.bar(hourly, x="timestamp", y="kwh", color="cost", title="Energy per hour (kWh)"),
                        use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(d, x="power", nbins=30, title="Load distribution"),
                        use_container_width=True)
        c2.plotly_chart(px.scatter(d, x="temp", y="power", trendline="ols", title="Temperature vs power")
                        if d.temp.notna().sum() > 5 and _has_statsmodels() else
                        px.scatter(d, x="temp", y="power", title="Temperature vs power"),
                        use_container_width=True)

    with t3:
        st.subheader("Recommendations")
        for tip in rule_based_tips(s):
            st.write("• " + tip)
        peak_kwh = d.loc[d.timestamp.dt.hour.isin(PEAK_HOURS), "kwh"].sum()
        saving = peak_kwh * 0.3 * tariff * (peak_mult - 1)
        st.success(f"Shifting 30% of peak-hour load to off-peak could save about {saving:.2f} per cycle.")
        st.json(s, expanded=False)

    with t4:
        render_chat(model, ollama_url)


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


def render_chat(model: str, url: str):
    st.caption("Ask about your usage. Needs Ollama running (`ollama serve`, `ollama pull " + model + "`).")
    hist = st.session_state.setdefault("chat", [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    q = st.text_input("Your question", key="q", placeholder="How can I cut my evening bill?")
    if st.button("Ask AI") and q:
        s = st.session_state.get("summary", {})
        with st.spinner("Thinking…"):
            if ollama_up(url):
                try:
                    ans = ask_ai(q, s, model, url)
                except Exception as e:
                    ans = f"LLM error: {e}\n\n" + "\n".join(rule_based_tips(s))
            else:
                ans = "Ollama is not reachable. Rule-based advice:\n\n" + "\n".join(rule_based_tips(s))
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        st.rerun(scope="app")


dashboard()
