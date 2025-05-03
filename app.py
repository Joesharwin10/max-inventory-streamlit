import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Streamlit page config
st.set_page_config(page_title="Max Inventory Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Max Showroom Data.csv")
    df = df.dropna(subset=["Available Stock", "Sold Stock", "Total Items", "Target", "Price", "Restock Needed"])
    return df

df = load_data()

# Title
st.title("🛍️ Max Inventory Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].dropna().unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].dropna().unique())

# Apply filters
filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Products", len(filtered_df))
col2.metric("Total Sold", int(filtered_df["Sold Stock"].sum()))
col3.metric("Average Price", f"₹{filtered_df['Price'].mean():.2f}")

# ----------------------------
# Restocking Prediction Section
# ----------------------------
st.markdown("---")
st.header("📦 Restocking Predictor")

# Train model
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X, y)

# Predict on filtered data
if not filtered_df.empty:
    pred_X = filtered_df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
    filtered_df["Restocking Status"] = model.predict(pred_X)
    filtered_df["Restocking Status"] = filtered_df["Restocking Status"].map({1: "Restock Needed", 0: "Sufficient"})

    # Highlight restocking status
    def highlight_status(val):
        color = '#ffcccc' if val == "Restock Needed" else '#ccffcc'
        return f'background-color: {color}'

    st.subheader("🔁 Predicted Restocking Status")
    styled_df = filtered_df[["Brand", "Category", "Available Stock", "Sold Stock", "Target", "Price", "Restocking Status"]]
    
    try:
        # Use to_html + markdown for consistent styling on Streamlit Cloud
        st.markdown(styled_df.style.applymap(highlight_status, subset=["Restocking Status"]).to_html(), unsafe_allow_html=True)
    except:
        st.dataframe(styled_df)  # fallback if styling fails

# Plot
st.markdown("---")
st.subheader("📊 Sold Stock by Size")
if "Size" in filtered_df.columns:
    plot_df = filtered_df.groupby("Size")["Sold Stock"].sum()
    fig, ax = plt.subplots()
    plot_df.plot(kind="bar", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Size column is missing in data.")

# Show data
st.subheader("📋 Filtered Data")
st.dataframe(filtered_df)
