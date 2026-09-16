import streamlit as st
import pandas as pd
import joblib

# Load your trained model (must be in the same folder as this file)
model = joblib.load('house_price_model.pkl')

st.title("California House Price Predictor")
st.write("Enter the details below to predict your house value in California!")

# --- User inputs ---
median_income = st.number_input(
    "Income (in tens of thousands, e.g. 5.0 = $50,000)",
    min_value=0.0, max_value=20.0, value=5.0, step=0.1
)

rooms_per_household = st.number_input(
    "Number of rooms in your household",
    min_value=0.0, max_value=20.0, value=5.0, step=0.1
)

ocean_proximity = st.selectbox(
    "Ocean proximity",
    ["<1H to the ocean", "Inland", "Near the ocean", "Near the bay"]
)

# --- Convert ocean_proximity into the one-hot columns your model expects ---
ocean_proximity_INLAND = 1 if ocean_proximity == "Inland" else 0
ocean_proximity_NEAR_OCEAN = 1 if ocean_proximity == "Near the ocean" else 0
ocean_proximity_NEAR_BAY = 1 if ocean_proximity == "Near the bay" else 0
# "<1H OCEAN" is the baseline category, so all three stay 0 for it

# --- Build input row in the exact column order your model was trained on ---
input_df = pd.DataFrame([{
    "median_income": median_income,
    "rooms_per_household": rooms_per_household,
    "ocean_proximity_INLAND": ocean_proximity_INLAND,
    "ocean_proximity_NEAR OCEAN": ocean_proximity_NEAR_OCEAN,
    "ocean_proximity_NEAR BAY": ocean_proximity_NEAR_BAY,
}])

if st.button("Predict house value"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted median house value: ${prediction:,.2f}")
