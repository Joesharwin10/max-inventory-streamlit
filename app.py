import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Max Showroom Data.csv")

df = load_data()

# Drop rows with missing values in required columns
df = df.dropna(subset=["Available Stock", "Sold Stock", "Total Items", "Target", "Price", "Restock Needed"])

# Train logistic regression model
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

# Title
st.title("🛍️ Chennai Max Inventory Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique())

# Apply filters
filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# Predict Restocking Status for filtered data
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

# ----------------------------
# Restocking Prediction Section
# ----------------------------
st.markdown("---")
st.header("📦 Restocking Predictor")

col_a, col_b = st.columns(2)

with col_a:
    available_stock = st.number_input("Enter Available Stock", min_value=0, step=1)
with col_b:
    sold_stock = st.number_input("Enter Sold Stock", min_value=0, step=1)

if available_stock or sold_stock:
    restock_needed = "Yes" if available_stock < sold_stock else "No"
    status_color = "green" if restock_needed == "Yes" else "red"
    st.markdown(f"### ✅ Restocking Status: <span style='color:{status_color}'>{restock_needed}</span>", unsafe_allow_html=True)

# Plot
st.subheader("📊 Sold Stock by Size")
plot_df = filtered_df.groupby("Size")["Sold Stock"].sum()
fig, ax = plt.subplots()
plot_df.plot(kind="bar", ax=ax, color='skyblue')
ax.set_xlabel("Size")
ax.set_ylabel("Sold Stock")
st.pyplot(fig)

# Show data with highlighting
st.subheader("📋 Filtered Inventory Data")

def highlight_status(val):
    color = '#ffcccc' if val == "Restock Needed" else '#ccffcc'
    return f'background-color: {color}'

if not filtered_df.empty:
    styled = filtered_df.style.applymap(highlight_status, subset=["Restocking Status"])
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No data to display based on selected filters.")
