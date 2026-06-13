import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NASA Exoplanet Explorer",
    page_icon="🚀",
    layout="centered"
)

# ---------------- LOAD DATA ----------------

model = joblib.load("model.pkl")

df = pd.read_csv("planets.csv")

# ---------------- TITLE ----------------

st.title("🚀 NASA Exoplanet Habitability Explorer")

st.markdown("""
Explore known exoplanets and discover whether they may support life.
""")

# ---------------- PLANET SELECT ----------------

planet = st.selectbox(
    "🌌 Choose a Planet",
    df["planet"]
)

selected = df[df["planet"] == planet].iloc[0]

radius = selected["radius"]
temp = selected["temp"]
distance = selected["distance"]
period = selected["period"]

# ---------------- BUTTON ----------------

if st.button("🔍 Analyze Planet"):

    prediction = model.predict(
        [[radius, period, temp, distance]]
    )[0]

    probability = model.predict_proba(
        [[radius, period, temp, distance]]
    )[0][1]

    score = int(probability * 100)

    st.header(f"🪐 {planet}")

    st.metric(
        "Habitability Score",
        f"{score}%"
    )

    st.progress(score)

    # ---------------- RESULT ----------------

    if prediction == 1:

        st.success(
            "🌍 Potentially Habitable"
        )

        st.balloons()

    else:

        st.error(
            "☄️ Not Habitable"
        )

    # ---------------- WHY SECTION ----------------

    st.subheader("🔬 Habitability Analysis")

    # Water

    if 0.8 <= distance <= 1.5:
        st.success(
            "💧 Possible Liquid Water"
        )
    else:
        st.warning(
            "💧 Water Unlikely"
        )

    # Planet Size

    if radius <= 2:
        st.success(
            "🌍 Earth-like Size"
        )
    else:
        st.warning(
            "🌍 Larger than Earth"
        )

    # Temperature

    if 4500 <= temp <= 6500:
        st.success(
            "☀ Suitable Star Temperature"
        )
    else:
        st.warning(
            "☀ Extreme Star Temperature"
        )

    # Atmosphere

    st.info(
        "🌫 Atmosphere Data Unknown"
    )

    # Life Probability

    if score >= 80:

        st.success(
            "👽 High Probability of Supporting Life"
        )

    elif score >= 50:

        st.info(
            "👽 Moderate Possibility of Life"
        )

    else:

        st.warning(
            "👽 Low Probability of Life"
        )

    # Planet Facts

    st.subheader("📊 Planet Facts")

    st.write(f"**Radius:** {radius} Earth Radii")
    st.write(f"**Star Temperature:** {temp} K")
    st.write(f"**Distance From Star:** {distance} AU")
    st.write(f"**Orbital Period:** {period} Days")

    # Summary

    st.subheader("📝 AI Summary")

    if prediction == 1:

        st.write(
            f"""
            {planet} appears to have several characteristics
            associated with potentially habitable planets.
            
            The planet is located near the habitable zone,
            may support liquid water, and has an Earth-like size.
            """
        )

    else:

        st.write(
            f"""
            {planet} has characteristics that make
            habitability less likely.
            
            Factors such as temperature, size,
            or orbital distance may reduce the chance
            of supporting life.
            """
        )
