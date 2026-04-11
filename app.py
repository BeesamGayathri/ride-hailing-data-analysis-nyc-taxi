import streamlit as st
st.cache_data.clear()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NYC Taxi Analysis", layout="wide")

st.title("🚖 Ride-Hailing Data Analysis: NYC Taxi Insights")
st.markdown("### 📊 Explore trip behavior, pricing, and demand trends")

# -------------------- CUSTOM UI --------------------
st.markdown("""
<style>
.metric-card {
    background-color: #111;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    csv_path = "yellow_tripdata_2020-06.csv"

    if not os.path.exists(csv_path):
        st.error("❌ CSV file not found in root folder")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        st.success(f"✅ Loaded dataset from: {csv_path}")
        return df
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# -------------------- DATA CLEANING --------------------
df.columns = df.columns.str.strip()

df['passenger_count'] = df['passenger_count'].fillna(df['passenger_count'].median())
df['trip_distance'] = df['trip_distance'].fillna(df['trip_distance'].median())
df['tip_amount'] = df['tip_amount'].fillna(0)
df['payment_type'] = df['payment_type'].fillna(df['payment_type'].mode()[0])

df = df[
    (df['passenger_count'] > 0) &
    (df['trip_distance'] > 0) &
    (df['fare_amount'] > 0) &
    (df['total_amount'] > 0)
].copy()

df = df.drop_duplicates()

# -------------------- DATETIME FEATURES --------------------
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()

df['trip_duration'] = (
    df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
).dt.total_seconds() / 60

df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 300)]

# -------------------- FEATURE ENGINEERING --------------------
df['fare_per_km'] = df['fare_amount'] / df['trip_distance']
df['tip_percentage'] = (df['tip_amount'] / df['fare_amount']) * 100
df['trip_speed'] = df['trip_distance'] / (df['trip_duration'] / 60)

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔍 Filters")

hour = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))

day = st.sidebar.multiselect(
    "Select Day of Week",
    options=df['day_of_week'].dropna().unique(),
    default=df['day_of_week'].dropna().unique()
)

filtered_df = df[
    (df['pickup_hour'] >= hour[0]) &
    (df['pickup_hour'] <= hour[1]) &
    (df['day_of_week'].isin(day))
]

# -------------------- KPI CARDS --------------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="metric-card">
<h3>🚖 Total Trips</h3>
<h2>{len(filtered_df)}</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-card">
<h3>💰 Avg Fare</h3>
<h2>${round(filtered_df['fare_amount'].mean(),2)}</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="metric-card">
<h3>📏 Avg Distance</h3>
<h2>{round(filtered_df['trip_distance'].mean(),2)} km</h2>
</div>
""", unsafe_allow_html=True)

# -------------------- CHARTS --------------------
st.subheader("⏱️ Trips by Hour")
st.bar_chart(filtered_df['pickup_hour'].value_counts().sort_index())

st.subheader("💰 Fare vs Distance")
fig1, ax1 = plt.subplots()
sns.scatterplot(x='trip_distance', y='fare_amount', data=filtered_df, ax=ax1)
st.pyplot(fig1)

# -------------------- MAP --------------------
st.subheader("🗺️ Pickup Locations (Demo Map)")

sample_map = filtered_df.copy().head(1000)
sample_map['lat'] = np.random.uniform(40.6, 40.9, size=len(sample_map))
sample_map['lon'] = np.random.uniform(-74.05, -73.7, size=len(sample_map))

st.map(sample_map[['lat', 'lon']])

# -------------------- ML MODEL --------------------
st.subheader("🤖 Fare Prediction Model")

model_df = filtered_df[['trip_distance', 'trip_duration', 'fare_amount']].dropna()

X = model_df[['trip_distance', 'trip_duration']]
y = model_df['fare_amount']

model = LinearRegression()
model.fit(X, y)

dist = st.slider("Trip Distance (km)", 1.0, 20.0, 5.0)
dur = st.slider("Trip Duration (min)", 5.0, 60.0, 20.0)

prediction = model.predict([[dist, dur]])

st.success(f"💰 Predicted Fare: ${round(prediction[0],2)}")

# -------------------- TIME SERIES --------------------
st.subheader("📈 Trips Over Time")

time_df = filtered_df.copy()
time_df['date'] = time_df['tpep_pickup_datetime'].dt.date

trend = time_df.groupby('date').size()

st.line_chart(trend)

# -------------------- DATA VIEW --------------------
st.subheader("📄 Sample Data")
st.dataframe(filtered_df.head(100))

# -------------------- DOWNLOAD --------------------
st.subheader("⬇️ Export Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="nyc_taxi_filtered.csv",
    mime="text/csv"
)
