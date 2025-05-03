import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("Max Showroom Data.csv")
    df = df.dropna(subset=["Available Stock", "Sold Stock", "Total Items", "Target", "Price", "Restock Needed"])
    return df

df = load_data()

# Train model
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

# Title
st.title("🛍️ Chennai Max Inventory Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].dropna().unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].dropna().unique())

# Apply filters
filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# Predict restocking status
if not filtered_df.empty:
    pred_X = filtered_df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
    filtered_df["Restocking Status"] = model.predict(pred_X)
    filtered_df["Restocking Status"] = filtered_df["Restocking Status"].map({1: "Restock Needed", 0: "Sufficient"})
else:
    filtered_df["Restocking Status"] = []

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Products", len(filtered_df))
col2.metric("Total Sold", filtered_df["Sold Stock"].sum())
col3.metric("Average Price", f"₹{filtered_df['Price'].mean():.2f}")

# Visualization
st.subheader("📊 Sold Stock by Size")
plot_df = filtered_df.groupby("Size")["Sold Stock"].sum()
fig, ax = plt.subplots()
plot_df.plot(kind="bar", ax=ax, color='orange')
ax.set_xlabel("Size")
ax.set_ylabel("Sold Stock")
st.pyplot(fig)

# Show filtered data with restocking status
st.subheader("📋 Filtered Inventory Data")

def highlight_status(val):
    color = '#ffcccc' if val == "Restock Needed" else '#ccffcc'
    return f'background-color: {color}'

if not filtered_df.empty:
    styled = filtered_df.style.applymap(highlight_status, subset=["Restocking Status"])
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No data to display based on selected filters.")
