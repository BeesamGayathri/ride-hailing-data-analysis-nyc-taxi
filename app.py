import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")

st.title("🚖 Ride-Hailing Data Analysis: NYC Taxi Insights")
st.markdown("### 📊 Explore trip behavior, pricing, and demand trends")

# -------------------- LOAD DATA (FINAL SAFE VERSION) --------------------
@st.cache_data
def load_data():
    possible_paths = [
        "data/yellow_tripdata_2020-06.csv",
        "./data/yellow_tripdata_2020-06.csv",
        "yellow_tripdata_2020-06.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)

                # Handle empty or corrupted file
                if df.empty:
                    st.error("❌ CSV file is empty or corrupted")
                    return pd.DataFrame()

                # 🔥 PERFORMANCE: sample if too large
                if len(df) > 100000:
                    df = df.sample(100000, random_state=42)

                st.success(f"✅ Loaded dataset from: {path}")
                return df

            except Exception as e:
                st.error("❌ Error reading CSV file")
                st.exception(e)
                return pd.DataFrame()

    st.error("❌ CSV file not found")
    return pd.DataFrame()


df = load_data()

if df.empty:
    st.stop()

# -------------------- CLEAN COLUMN NAMES --------------------
df.columns = df.columns.str.strip()

# -------------------- REQUIRED COLUMNS CHECK --------------------
required_cols = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"❌ Missing columns: {missing_cols}")
    st.stop()

# -------------------- DATA CLEANING --------------------
df["passenger_count"] = df["passenger_count"].fillna(df["passenger_count"].median())
df["trip_distance"] = df["trip_distance"].fillna(df["trip_distance"].median())
df["tip_amount"] = df["tip_amount"].fillna(0)

df = df[
    (df["passenger_count"] > 0) &
    (df["trip_distance"] > 0) &
    (df["fare_amount"] > 0) &
    (df["total_amount"] > 0)
].copy()

df = df.drop_duplicates()

# -------------------- DATETIME FEATURES --------------------
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()

df["trip_duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

df = df[(df["trip_duration"] > 0) & (df["trip_duration"] < 300)]

# -------------------- FEATURE ENGINEERING --------------------
df["fare_per_km"] = df["fare_amount"] / df["trip_distance"]
df["tip_percentage"] = (df["tip_amount"] / df["fare_amount"]) * 100
df["trip_speed"] = df["trip_distance"] / (df["trip_duration"] / 60)

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔍 Filters")

hour = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))

day = st.sidebar.multiselect(
    "Select Day of Week",
    options=df["day_of_week"].unique(),
    default=df["day_of_week"].unique()
)

filtered_df = df[
    (df["pickup_hour"] >= hour[0]) &
    (df["pickup_hour"] <= hour[1]) &
    (df["day_of_week"].isin(day))
]

# -------------------- KPI METRICS --------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("🚖 Total Trips", len(filtered_df))
col2.metric("💰 Avg Fare ($)", round(filtered_df["fare_amount"].mean(), 2))
col3.metric("📏 Avg Distance (km)", round(filtered_df["trip_distance"].mean(), 2))

# -------------------- VISUALIZATIONS --------------------

st.subheader("⏱️ Trips by Hour")
fig1, ax1 = plt.subplots()
sns.countplot(x="pickup_hour", data=filtered_df, ax=ax1)
st.pyplot(fig1)

st.subheader("💰 Fare vs Distance")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="trip_distance", y="fare_amount", data=filtered_df, ax=ax2)
st.pyplot(fig2)

st.subheader("💸 Tip vs Fare")
fig3, ax3 = plt.subplots()
sns.scatterplot(x="fare_amount", y="tip_amount", data=filtered_df, ax=ax3)
st.pyplot(fig3)

st.subheader("⏱️ Duration vs Distance")
fig4, ax4 = plt.subplots()
sns.scatterplot(x="trip_distance", y="trip_duration", data=filtered_df, ax=ax4)
st.pyplot(fig4)

st.subheader("🔥 Correlation Heatmap")
corr = filtered_df[
    ["fare_amount", "trip_distance", "trip_duration", "tip_amount"]
].corr()

fig5, ax5 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
st.pyplot(fig5)

# -------------------- DATA VIEW --------------------
st.subheader("📄 Sample Data")
st.dataframe(filtered_df.head(100))

# -------------------- DOWNLOAD OPTION --------------------
st.subheader("⬇️ Export Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="nyc_taxi_filtered.csv",
    mime="text/csv"
)
