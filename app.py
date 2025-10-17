import pandas as pd
import streamlit as st
from difflib import SequenceMatcher
import altair as alt

# --- App Title ---
st.set_page_config(page_title="SampleMetrics", layout="wide")
st.title("🏠 SampleMetrics Real Estate Analytics (UK)")

# --- Sidebar ---
st.sidebar.header("Upload / Select Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
sample_option = st.sidebar.selectbox("Or choose sample data", 
                                     ["None", "Balanced", "High Risk", "Premium", "Regional", "Low Yield"])

# --- Load Data ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded file: {uploaded_file.name}")
elif sample_option != "None":
    sample_files = {
        "Balanced": "balanced_portfolio.csv",
        "High Risk": "high_risk_portfolio.csv",
        "Premium": "premium_portfolio.csv",
        "Regional": "regional_portfolio.csv",
        "Low Yield": "low_yield_portfolio.csv"
    }
    df = pd.read_csv(sample_files[sample_option])
    st.success(f"Loaded {sample_option} sample data")
else:
    st.warning("Please upload a CSV file or select a sample dataset from the sidebar.")
    st.stop()

# --- Required columns check ---
required_columns = ['PropertyID','Location','PurchasePrice','CurrentValue','AnnualRent','VacancyRate']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# --- Add missing columns ---
if 'address' not in df.columns:
    df['address'] = df['Location'].astype(str) + " " + df['PropertyID'].astype(str)
if 'price' not in df.columns:
    df['price'] = df['PurchasePrice']
if 'bedrooms' not in df.columns:
    df['bedrooms'] = [(i % 5) + 1 for i in range(len(df))]

# --- Duplicate / high-risk detection ---
def compare_strings(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

duplicates = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if compare_strings(df.loc[i,'address'], df.loc[j,'address']) > 0.9 \
            and abs(df.loc[i,'price'] - df.loc[j,'price'])/df.loc[i,'price'] < 0.06 \
            and df.loc[i,'bedrooms'] == df.loc[j,'bedrooms']:
            duplicates.append((i,j))

# --- Portfolio Metrics ---
df['ROI'] = (df['CurrentValue'] - df['PurchasePrice']) / df['PurchasePrice']
df['RentalYield'] = df['AnnualRent'] / df['PurchasePrice']
df['RiskScore'] = (df['VacancyRate'] * 0.5 + (1 - df['ROI'].clip(0,1)) * 0.5)  # Simple risk metric 0–1

# --- Layout: Metrics ---
st.subheader("📊 Portfolio Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Average ROI", f"{df['ROI'].mean()*100:.2f}%")
col2.metric("Average Rental Yield", f"{df['RentalYield'].mean()*100:.2f}%")
col3.metric("Average Vacancy Rate", f"{df['VacancyRate'].mean()*100:.2f}%")

# --- Risk Distribution Chart ---
st.subheader("⚠️ Risk Distribution")
risk_chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('RiskScore', bin=alt.Bin(maxbins=10), title='Risk Score'),
    y='count()',
    tooltip=['count()']
).properties(width=700, height=300)
st.altair_chart(risk_chart)

# --- ROI vs Rental Yield Chart ---
st.subheader("📈 ROI vs Rental Yield")
scatter_chart = alt.Chart(df).mark_circle(size=80).encode(
    x='ROI',
    y='RentalYield',
    color=alt.Color('RiskScore', scale=alt.Scale(scheme='redyellowgreen')),
    tooltip=['PropertyID', 'Location', 'ROI', 'RentalYield', 'VacancyRate']
).interactive()
st.altair_chart(scatter_chart, use_container_width=True)

# --- Duplicate / High-Risk Pairs ---
st.subheader("🛑 Duplicate / High-Risk Pairs Detected")
st.write(duplicates)

# --- Raw Data ---
with st.expander("View Full Data"):
    st.dataframe(df)
