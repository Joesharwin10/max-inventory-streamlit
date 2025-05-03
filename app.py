import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Max Showroom Data.csv")
    df.fillna({
        "Available Stock": 0,
        "Sold Stock": 0,
        "Total Items": 0,
        "Target": 0,
        "Price": 0,
        "Restock Needed": 0
    }, inplace=True)
    return df

df = load_data()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].dropna().unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].dropna().unique())

filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# ----------------------------
# Title & Metrics
# ----------------------------
st.title("🛍️ Chennai Max Inventory Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Products", len(filtered_df))
col2.metric("Total Sold", int(filtered_df["Sold Stock"].sum()))
col3.metric("Average Price", f"₹{filtered_df['Price'].mean():.2f}")

# ----------------------------
# Train Logistic Regression Model
# ----------------------------
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

# ----------------------------
# Restocking Predictor (Manual Input)
# ----------------------------
st.markdown("---")
st.header("🔁 Restocking Predictor (Manual Input)")

col_a, col_b = st.columns(2)
with col_a:
    input_available = st.number_input("Available Stock", min_value=0, step=1)
with col_b:
    input_sold = st.number_input("Sold Stock", min_value=0, step=1)

# Auto-fill other fields for simplicity
input_total = input_available + input_sold
input_target = st.slider("Target", min_value=0, max_value=200, value=50)
input_price = st.slider("Price (₹)", min_value=100, max_value=10000, value=1000)

if input_available or input_sold:
    pred = model.predict([[input_available, input_sold, input_total, input_target, input_price]])[0]
    label = "Restock Needed" if pred == 1 else "Sufficient"
    color = "green" if pred == 1 else "red"
    st.markdown(f"### ✅ Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

# ----------------------------
# Plot - Sold Stock by Size
# ----------------------------
st.markdown("---")
st.subheader("📊 Sold Stock by Size")
plot_df = filtered_df.groupby("Size")["Sold Stock"].sum().sort_index()
fig, ax = plt.subplots()
plot_df.plot(kind="bar", ax=ax, color="#007acc")
ax.set_ylabel("Sold Stock")
st.pyplot(fig)

# ----------------------------
# Show Filtered Data
# ----------------------------
st.markdown("---")
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
