import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("🚀 Exoplanet Habitability Predictor")

radius = st.number_input(
    "Planet Radius",
    value=1.0
)

period = st.number_input(
    "Orbital Period",
    value=365
)

temp = st.number_input(
    "Star Temperature",
    value=5778
)

distance = st.number_input(
    "Distance From Star",
    value=1.0
)

if st.button("Predict"):

    prediction = model.predict(
        [[radius, period, temp, distance]]
    )[0]

    probability = model.predict_proba(
        [[radius, period, temp, distance]]
    )[0][1]

    st.write(
        f"Habitability Score: {probability*100:.1f}%"
    )

    if prediction == 1:
        st.success(
            "🌍 Potentially Habitable"
        )
    else:
        st.error(
            "☄️ Not Habitable"
        )
        