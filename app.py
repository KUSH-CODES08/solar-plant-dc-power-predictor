import streamlit as st
import joblib
import pandas as pd

# ====================================
# LOAD MODEL & SCALER
# ====================================
model = joblib.load("solar_Power_generation_model.pkl")
scaler = joblib.load("scaler.pkl")

# ====================================
# APP UI
# ====================================
st.set_page_config(page_title="Solar DC Power Predictor", page_icon="☀️")

st.title("☀️ Solar DC Power Prediction App")
st.markdown(
    """
    This app predicts **plant-level DC Power output**  
    based on **solar irradiation and temperature conditions**.
    """
)

st.divider()

# ====================================
# USER INPUTS
# ====================================
irradiation = st.number_input(
    "🌞 IRRADIATION (W/m²)",
    min_value=0.0,
    step=10.0
)

module_temp = st.number_input(
    "🔧 MODULE TEMPERATURE (°C)",
    step=1.0
)

ambient_temp = st.number_input(
    "🌡️ AMBIENT TEMPERATURE (°C)",
    step=1.0
)

# ====================================
# PREDICTION
# ====================================
if st.button("🔮 Predict DC Power"):

    if irradiation <= 0:
        st.warning("No sunlight detected → DC Power = 0 kW")
    else:
        input_df = pd.DataFrame([{
            "IRRADIATION": irradiation,
            "MODULE_TEMPERATURE": module_temp,
            "AMBIENT_TEMPERATURE": ambient_temp
        }])

        # Scale input
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(
            input_scaled,
            columns=input_df.columns
        )

        # Predict
        prediction = model.predict(input_scaled_df)

        # Convert W → kW
        dc_power_kw = prediction[0] / 1000

        st.success(f"⚡ Predicted DC Power: **{dc_power_kw:,.2f} kW**")

        st.caption(
            "ℹ️ This is total DC power output of the entire solar plant."
        )

st.divider()

st.caption("Built using Machine Learning | Linear Regression | Streamlit")
