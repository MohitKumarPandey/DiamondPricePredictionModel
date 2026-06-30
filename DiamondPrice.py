import streamlit as st
import pandas as pd
import joblib

# ------------------ Page Config ------------------

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# ------------------ Load Model ------------------

model = joblib.load("DiamondPricePredictionMl (1).pkl")
cut_encoder = joblib.load("cut_encoder (1).pkl")
color_encoder = joblib.load("color_encoder (1).pkl")
clarity_encoder = joblib.load("clarity_encoder.pkl")

# ------------------ Title ------------------

st.title("💎 Diamond Price Prediction")
st.write("Predict the estimated price of a diamond using Machine Learning.")

st.divider()

# ------------------ Sidebar ------------------

st.sidebar.header("About")

st.sidebar.info("""
This project predicts the estimated price of a diamond based on:

• Carat
• Cut
• Color
• Clarity
• Depth
• Table
• Length (x)
• Width (y)
• Height (z)

Model Used:
Linear Regression
""")

# ------------------ Inputs ------------------

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input(
        "Carat",
        min_value=0.10,
        max_value=5.50,
        value=0.50,
        step=0.01
    )

    cut = st.selectbox(
        "Cut",
        ["ideal", "premium", "very good", "good", "fair"]
    )

    color = st.selectbox(
        "Color",
        ["D", "E", "F", "G", "H", "I", "J"]
    )

    clarity = st.selectbox(
        "Clarity",
        ["SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"]
    )

with col2:
    depth = st.number_input(
        "Depth (%)",
        min_value=40.0,
        max_value=80.0,
        value=61.5
    )

    table = st.number_input(
        "Table (%)",
        min_value=40.0,
        max_value=80.0,
        value=55.0
    )

    x = st.number_input(
        "Length (mm)",
        min_value=0.0,
        value=5.0
    )

    y = st.number_input(
        "Width (mm)",
        min_value=0.0,
        value=5.0
    )

    z = st.number_input(
        "Height (mm)",
        min_value=0.0,
        value=3.0
    )

st.divider()

predict = st.button("🔮 Predict Price", use_container_width=True)
if predict:

    try:

        # Encode categorical values
        cut_value = cut_encoder.transform([cut])[0]
        color_value = color_encoder.transform([color])[0]
        clarity_value = clarity_encoder.transform([clarity])[0]

        # Create DataFrame
        new_data = pd.DataFrame({
            "carat": [carat],
            "cut": [cut_value],
            "color": [color_value],
            "clarity": [clarity_value],
            "depth": [depth],
            "table": [table],
            "x": [x],
            "y": [y],
            "z": [z]
        })

        # Prediction
        prediction = model.predict(new_data)[0]

        # Avoid negative values
        prediction = max(0, prediction)

        st.success("Prediction Successful ✅")

        st.metric(
            label="Estimated Diamond Price",
            value=f"${prediction:,.2f}"
        )

        with st.expander("View Entered Details"):
            st.dataframe(new_data, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

st.caption("💎 Diamond Price Prediction using Machine Learning")
st.caption("Developed by Mohit Pandey")
