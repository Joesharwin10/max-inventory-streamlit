import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Max Showroom Data.csv")

df = load_data()
df = df.dropna(subset=["Available Stock", "Sold Stock", "Total Items", "Target", "Price", "Restock Needed"])

# Train logistic regression model
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

# --------------------------
# Sidebar: Filter Options
# --------------------------
st.sidebar.header("🔍 Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].dropna().unique(), default=df["Category"].unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].dropna().unique(), default=df["Gender"].unique())

# Filter data
filtered_df = df[df["Category"].isin(category) & df["Gender"].isin(gender)]

# --------------------------
# Page Title & Metrics
# --------------------------
st.title("🛍️ Max Inventory Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Products", len(filtered_df))
col2.metric("Total Sold", filtered_df["Sold Stock"].sum())
col3.metric("Avg. Price", f"₹{filtered_df['Price'].mean():.2f}")

# --------------------------
# Visualization
# --------------------------
st.subheader("📊 Sold Stock by Size")
plot_df = filtered_df.groupby("Size")["Sold Stock"].sum()
fig, ax = plt.subplots()
plot_df.plot(kind="bar", ax=ax, color="skyblue")
st.pyplot(fig)

# --------------------------
# ML Prediction Section
# --------------------------
st.markdown("---")
st.header("🔁 Restocking Predictor")

if not filtered_df.empty:
    pred_X = filtered_df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
    pred_y = model.predict(pred_X)
    filtered_df["Restocking Status"] = pred_y
    filtered_df["Restocking Status"] = filtered_df["Restocking Status"].map({1: "Restock Needed", 0: "Sufficient"})

    def highlight_status(val):
        color = '#ffcccc' if val == "Restock Needed" else '#ccffcc'
        return f'background-color: {color}'

    st.dataframe(filtered_df.style.applymap(highlight_status, subset=["Restocking Status"]), use_container_width=True)
else:
    st.warning("No data matches your selected filters.")
