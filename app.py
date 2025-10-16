
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import math
from difflib import SequenceMatcher

st.set_page_config(layout="wide", page_title="SampleMetrics Demo")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def compare_strings(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Agents sample (id, name, lat, lon, specialization)
AGENTS = [
    {"agent_id":"A001", "name":"Kamau", "lat":51.5074, "lon":-0.1278, "specialization":"Apartment", "active_listings":5},
    {"agent_id":"A002", "name":"Aisha", "lat":53.4808, "lon":-2.2426, "specialization":"House", "active_listings":3},
    {"agent_id":"A003", "name":"Gareth", "lat":51.4816, "lon":-3.1791, "specialization":"Commercial", "active_listings":4},
    {"agent_id":"A004", "name":"Sophie", "lat":52.4862, "lon":-1.8904, "specialization":"House", "active_listings":6},
    {"agent_id":"A005", "name":"Liam", "lat":51.5090, "lon":-0.1340, "specialization":"Apartment", "active_listings":2},
]

st.title("SampleMetrics — UK Portfolio Analytics (Demo)")
st.markdown("**Demo account:** demo@samplemetrics.com | **Password:** password123 (mock login)")

# Sidebar: upload or use sample
st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use sample UK dataset", value=True)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if use_sample or uploaded is None:
    df = pd.read_csv("sample_properties_uk.csv")
else:
    df = pd.read_csv(uploaded)

st.sidebar.markdown("---")
if st.sidebar.button("Run Analysis"):
    st.session_state['run'] = True

if 'run' not in st.session_state:
    st.session_state['run'] = False

col1, col2 = st.columns([1,3])

with col1:
    st.subheader("Upload & Preview")
    st.write("Rows:", len(df))
    st.dataframe(df.head(8), height=240)
    st.download_button("Download sample CSV", df.to_csv(index=False).encode('utf-8'), "sample_properties_uk.csv")
    st.markdown("**Processing toggles**")
    rec = st.checkbox("Enable Reconciliation", value=True)
    route = st.checkbox("Enable Routing", value=True)
    risk = st.checkbox("Enable Risk Scoring", value=True)
    lcm = st.checkbox("Enable LCM Portfolio Analysis", value=True)

with col2:
    st.subheader("Processing & Results")
    if not st.session_state['run']:
        st.info("Click **Run Analysis** in the sidebar to process data (demo).")
        st.stop()
    st.info("Running: reconciling duplicates, routing agents, risk scoring, and LCM analysis...")

    # 1. Reconciliation (simple fuzzy address matching)
    df = df.copy()
    df['is_duplicate'] = False
    if rec:
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                if compare_strings(df.loc[i,'address'], df.loc[j,'address']) > 0.90 and abs(df.loc[i,'price'] - df.loc[j,'price'])/df.loc[i,'price'] < 0.06 and df.loc[i,'bedrooms'] == df.loc[j,'bedrooms']:
                    df.loc[j,'is_duplicate'] = True
        cleaned = df[~df['is_duplicate']].reset_index(drop=True)
    else:
        cleaned = df.copy()

    st.markdown("### Reconciliation Summary")
    st.metric("Total uploaded", len(df))
    st.metric("Unique after reconciliation", len(cleaned))
    dup_count = df['is_duplicate'].sum()
    st.write(f"Duplicates flagged: {dup_count}")

    # 2. Routing logic (assign best agent)
    def assign_agent(row):
        best_score = -1
        best_agent = None
        for agent in AGENTS:
            distance_km = haversine(row['lat'], row['lon'], agent['lat'], agent['lon'])
            distance_score = max(0, 100 - distance_km*2)
            type_score = 100 if agent['specialization'] == row['property_type'] else 60
            workload_score = max(0, 100 - agent['active_listings']*5)
            total_score = distance_score*0.4 + type_score*0.4 + workload_score*0.2
            if total_score > best_score:
                best_score = total_score
                best_agent = agent['name']
        return best_agent, round(distance_km,2)

    if route:
        routing_results = []
        for idx, r in cleaned.iterrows():
            agent, dist = assign_agent(r)
            routing_results.append({"property_id": r['property_id'], "assigned_agent": agent, "distance_km": dist})
        routing_df = pd.DataFrame(routing_results)
    else:
        routing_df = pd.DataFrame(columns=["property_id","assigned_agent","distance_km"])

    st.markdown("### Routing Summary")
    st.dataframe(routing_df.head(8))

    # 3. Risk scoring
    if risk:
        def compute_risk(row):
            price_score = 100 - (abs(row['price'] - row['avg_area_price']) / row['avg_area_price'] * 100)
            safety_score = 100 - (row['crime_rate'] * 100)
            occupancy_score = 100 - (row['vacancy_rate'] * 100)
            growth_score = row['market_growth_rate'] * 100 * 0.2  # scaled
            risk_score = price_score*0.3 + safety_score*0.3 + occupancy_score*0.2 + growth_score*0.2
            return max(0, min(100, risk_score))
        cleaned['risk_score'] = cleaned.apply(compute_risk, axis=1)
        cleaned['risk_label'] = cleaned['risk_score'].apply(lambda x: "Low" if x>=70 else ("Medium" if x>=50 else "High"))
    else:
        cleaned['risk_score'] = np.nan
        cleaned['risk_label'] = ""

    st.markdown("### Risk Scoring")
    st.bar_chart(cleaned.set_index('property_id')['risk_score'])
    st.dataframe(cleaned[['property_id','location','price','risk_score','risk_label']])

    # 4. LCM Portfolio Analysis
    st.markdown("### LCM Portfolio Analysis")
    if lcm:
        records = []
        for idx, p in cleaned.iterrows():
            net_income = p['annual_rent_income'] - p['maintenance_cost_annual']
            net_yield = (net_income / p['price']) * 100 if p['price']>0 else 0
            age_years = p['age_years']
            if age_years <= 5:
                age_factor = 1.0
            elif age_years <= 15:
                age_factor = 0.85
            else:
                age_factor = 0.7
            vacancy_penalty = max(0, 1 - p['vacancy_rate'])
            lifecycle_index = (net_yield * 0.4) + ((100 - p['risk_score']) * 0.3) + (p['market_growth_rate']*100*0.2) + (age_factor*10*0.1)
            lifecycle_index = max(0, min(100, lifecycle_index))
            if lifecycle_index >= 70 and vacancy_penalty >= 0.9:
                lifecycle_stage = "Hold / Invest"
            elif 50 <= lifecycle_index < 70:
                lifecycle_stage = "Maintain"
            else:
                lifecycle_stage = "Divest"
            maintenance_priority_score = (p['maintenance_cost_annual'] / p['price']) * 1000 + (100 - lifecycle_index)
            if maintenance_priority_score > 50:
                maintenance_priority = "High"
            elif maintenance_priority_score > 25:
                maintenance_priority = "Medium"
            else:
                maintenance_priority = "Low"
            records.append({
                "property_id": p['property_id'],
                "location": p['location'],
                "price": p['price'],
                "net_yield": round(net_yield,2),
                "lifecycle_index": round(lifecycle_index,2),
                "lifecycle_stage": lifecycle_stage,
                "maintenance_priority": maintenance_priority
            })
        lcm_df = pd.DataFrame(records)
        st.dataframe(lcm_df)
        # Portfolio-level aggregates
        total_value = cleaned['price'].sum()
        weighted_avg_yield = (lcm_df['net_yield'] * cleaned['price']).sum() / total_value
        avg_lifecycle_index = lcm_df['lifecycle_index'].mean()
        st.metric("Total portfolio value (GBP)", f"£{int(total_value):,}")
        st.metric("Weighted avg net yield (%)", f"{round(weighted_avg_yield,2)}%")
        st.metric("Avg lifecycle index", f"{round(avg_lifecycle_index,2)}")
    else:
        st.write("LCM analysis disabled.")

    # Offer download of results (combined)
    result = cleaned.merge(routing_df, on='property_id', how='left')
    result = result.merge(lcm_df[['property_id','lifecycle_stage','maintenance_priority','net_yield','lifecycle_index']], on='property_id', how='left')
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Download analysis results (CSV)", csv, "samplemetrics_results.csv")
