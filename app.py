import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import os
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Paals diett APP")

DATA_FILE = "data.csv"

# ------------------------
# LOAD / SAVE
# ------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    return pd.DataFrame(columns=["date", "weight", "calories"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# ------------------------
# BEREGNING
# ------------------------
def process(df):
    df = df.copy()
    df["weight_smooth"] = df["weight"].rolling(7).mean()
    return df

def calculate_tdee(df):
    df = df.dropna()

    if len(df) < 7:
        return None

    window = min(14, len(df))

    start = df["weight_smooth"].iloc[-window]
    end = df["weight_smooth"].iloc[-1]

    if pd.isna(start) or pd.isna(end):
        return None

    delta = end - start
    kcal = delta * 7700
    avg = df["calories"].iloc[-window:].mean()

    return round(avg - (kcal / window), 0)

# ------------------------
# 🔮 PREDIKSJON
# ------------------------
def predict_weight(df, days=14):
    df = df.dropna()

    if len(df) < 7:
        return None

    y = df["weight_smooth"].dropna().values
    x = np.arange(len(y))

    coef = np.polyfit(x, y, 1)

    future_x = np.arange(len(y) + days)
    pred = coef[0] * future_x + coef[1]

    future_dates = pd.date_range(
        start=df["date"].iloc[-1],
        periods=days + 1
    )

    return future_dates, pred[-(days+1):]

# ------------------------
# 🎯 MÅLVEKT
# ------------------------
def predict_goal_date(df, target_weight):
    df = df.dropna()

    if len(df) < 7:
        return None

    y = df["weight_smooth"].dropna().values
    x = np.arange(len(y))

    coef = np.polyfit(x, y, 1)
    slope = coef[0]

    if slope >= 0:
        return None

    current_weight = y[-1]
    days_needed = (target_weight - current_weight) / slope

    if days_needed < 0:
        return "Allerede nådd"

    return df["date"].iloc[-1] + pd.Timedelta(days=int(days_needed))

# ------------------------
# 📉 PLATEAU
# ------------------------
def detect_plateau(df):
    df = df.dropna()

    if len(df) < 10:
        return False

    recent = df["weight_smooth"].iloc[-7:]
    change = recent.iloc[-1] - recent.iloc[0]

    return abs(change) < 0.2

# ------------------------
# 🔄 KALORIER
# ------------------------
def suggest_calories(tdee, goal):
    if not tdee:
        return None

    if goal == "cut":
        return round(tdee - 500)
    elif goal == "maintain":
        return round(tdee)
    elif goal == "bulk":
        return round(tdee + 300)

# ------------------------
# 📊 FETTPROSENT
# ------------------------
def estimate_bodyfat(df):
    df = df.dropna()

    if len(df) < 14:
        return None

    delta = df["weight_smooth"].iloc[-14] - df["weight_smooth"].iloc[-1]
    fat_loss = delta * 0.8

    bf = 20 - (fat_loss * 2)
    return round(max(5, bf), 1)

# ------------------------
# ⚠️ MUSKEL RISIKO
# ------------------------
def muscle_loss_risk(df, tdee):
    df = df.dropna()

    if len(df) < 7 or not tdee:
        return False

    avg_cal = df["calories"].iloc[-7:].mean()
    deficit = tdee - avg_cal

    return deficit > 800

# ------------------------
# ⚖️ BMI
# ------------------------
def calculate_bmi(weight, height_cm):
    return round(weight / ((height_cm / 100) ** 2), 1)

# ------------------------
# LOAD DATA
# ------------------------
df = load_data()

st.header("📊 Status")

if len(df) > 0:
    df_proc = process(df)
    tdee = calculate_tdee(df_proc)

    col1, col2 = st.columns(2)

    if tdee:
        col1.metric("TDEE", tdee)
    else:
        col1.warning("For lite data")

    plateau = detect_plateau(df_proc)

    if plateau:
        col2.warning("📉 Plateau")
    else:
        col2.success("📈 Progresjon")

    # 🔄 Kalorimål
    goal = st.selectbox("Mål", ["cut", "maintain", "bulk"])
    suggested = suggest_calories(tdee, goal)

    if suggested:
        st.metric("🔥 Kalorimål", suggested)

    # 🎯 Målvekt
    target_weight = st.number_input("🎯 Målvekt", value=80.0)
    goal_date = predict_goal_date(df_proc, target_weight)

    if isinstance(goal_date, str):
        st.success(goal_date)
    elif goal_date:
        st.success(f"📅 {goal_date.date()}")

    # 📊 Fettprosent
    bf = estimate_bodyfat(df_proc)
    if bf:
        st.metric("Fettprosent", f"{bf}%")

    # ⚠️ Muskel
    if muscle_loss_risk(df_proc, tdee):
        st.error("⚠️ Risiko for muskeltap")

    # ⚖️ BMI
    height = st.number_input("Høyde (cm)", value=180)
    current_weight = df_proc["weight"].dropna().iloc[-1]
    bmi = calculate_bmi(current_weight, height)
    st.metric("BMI", bmi)

    # 📈 GRAF
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["weight"],
        name="Vekt",
        mode="lines+markers",
        opacity=0.4
    ))

    fig.add_trace(go.Scatter(
        x=df_proc["date"],
        y=df_proc["weight_smooth"],
        name="Trend",
        line=dict(width=3)
    ))

    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["calories"],
        name="Kalorier",
        yaxis="y2",
        opacity=0.3
    ))

    pred = predict_weight(df_proc)
    if pred:
        fig.add_trace(go.Scatter(
            x=pred[0],
            y=pred[1],
            name="Prediksjon",
            line=dict(dash="dash")
        ))

    fig.update_layout(
        height=420,
        template="plotly_dark",
        yaxis=dict(title="Vekt"),
        yaxis2=dict(overlaying="y", side="right", title="Kalorier")
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.info("Ingen data enda")

# ------------------------
# ⚡ AUTO-FILL
# ------------------------
last_weight = df["weight"].dropna().iloc[-1] if len(df) > 0 and df["weight"].notna().any() else 0

st.header("⚖️ Logg vekt")

date_w = st.date_input("Dato", datetime.now())
weight = st.number_input("Vekt", value=float(last_weight), step=0.1)

if st.button("Lagre vekt"):
    if (df["date"] == pd.to_datetime(date_w)).any():
        df.loc[df["date"] == pd.to_datetime(date_w), "weight"] = weight
    else:
        df = pd.concat([df, pd.DataFrame([{
            "date": pd.to_datetime(date_w),
            "weight": weight,
            "calories": None
        }])])
    save_data(df)
    st.success("Lagret")

st.header("🔥 Logg kalorier")

date_c = st.date_input("Dato kalorier", datetime.now(), key="c")
cal = st.number_input("Kalorier", min_value=0)

if st.button("Lagre kalorier"):
    if (df["date"] == pd.to_datetime(date_c)).any():
        df.loc[df["date"] == pd.to_datetime(date_c), "calories"] = cal
    else:
        df = pd.concat([df, pd.DataFrame([{
            "date": pd.to_datetime(date_c),
            "weight": None,
            "calories": cal
        }])])
    save_data(df)
    st.success("Lagret")

st.header("✏️ Rediger")
if len(df) > 0:
    df_edit = st.data_editor(df.copy())
    if st.button("Lagre endringer"):
        df_edit["date"] = pd.to_datetime(df_edit["date"])
        save_data(df_edit)
        st.success("Oppdatert")

st.header("🗑️ Slett")
if len(df) > 0:
    selected = st.selectbox("Dato", df["date"].dt.strftime("%Y-%m-%d"))
    if st.button("Slett"):
        df = df[df["date"].dt.strftime("%Y-%m-%d") != selected]
        save_data(df)
        st.success("Slettet")
