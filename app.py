import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
planet_images = {
    "Earth":"https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg",
    "Mars":"https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg",
    "Venus":"https://upload.wikimedia.org/wikipedia/commons/e/e5/Venus-real_color.jpg",
    "Kepler-442 b":"https://upload.wikimedia.org/wikipedia/commons/5/58/Kepler-442b_artist_impression.jpg",
    "Kepler-186 f":"https://upload.wikimedia.org/wikipedia/commons/4/46/PIA18034-Kepler186f.jpg",
    "TRAPPIST-1 e":"https://upload.wikimedia.org/wikipedia/commons/d/d9/TRAPPIST-1e_artist_concept.jpg"
}
st.set_page_config(
    page_title="NASA Exoplanet Intelligence System",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------

st.markdown("""
<style>

.stApp {
    background-color: #050816;
    color: white;
}

h1,h2,h3 {
    color: #4FC3F7;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

model = joblib.load("model.pkl")

df = pd.read_csv("nasa_clean.csv")

# ---------------- TITLE ----------------

st.title("🚀 NASA Exoplanet Intelligence System")

st.write(
    "Analyze real exoplanets from NASA's Exoplanet Archive using Machine Learning."
)

# ---------------- SEARCH ----------------

planet_names = sorted(df["pl_name"].unique())

planet = st.selectbox(
    "🔍 Search Planet",
    planet_names
)

selected = df[df["pl_name"] == planet].iloc[0]

radius = selected["pl_rade"]
period = selected["pl_orbper"]
distance = selected["pl_orbsmax"]
temp = selected["st_teff"]

# ---------------- ANALYZE BUTTON ----------------

if st.button("🛰 Analyze Planet"):

    prediction = model.predict(
        [[radius, period, distance, temp]]
    )[0]

    probability = model.predict_proba(
        [[radius, period, distance, temp]]
    )[0][1]

    score = int(probability * 100)

    # ---------------- HEADER ----------------

    st.header(f"🪐 {planet}")
    if planet in planet_images:
    st.image(
        planet_images[planet],
        width=500
    )

    # ---------------- METRICS ----------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Habitability Score",
            f"{score}%"
        )

    # Earth Similarity Index

    esi = max(
        0,
        round(
            1 - abs(radius - 1) / 4,
            2
        )
    )

    with col2:
        st.metric(
            "🌍 Earth Similarity",
            esi
        )

    life_probability = int(
        ((score + esi * 100) / 2)
    )

    with col3:
        st.metric(
            "👽 Life Probability",
            f"{life_probability}%"
        )

    st.progress(life_probability)

    # ---------------- RESULT ----------------

    if prediction == 1:

        st.success(
            "🌍 Potentially Habitable"
        )

        st.balloons()

    else:

        st.error(
            "☄ Not Habitable"
        )

    # ---------------- PLANET DATA ----------------

    st.subheader("📊 NASA Planet Facts")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"⭐ Host Star: {selected['hostname']}"
        )

        st.write(
            f"📅 Discovery Year: {selected['disc_year']}"
        )

        st.write(
            f"🌍 Radius: {radius:.2f} Earth Radii"
        )

    with col2:

        st.write(
            f"🛰 Distance From Star: {distance:.2f} AU"
        )

        st.write(
            f"☀ Star Temperature: {temp:.0f} K"
        )

        st.write(
            f"🕒 Orbital Period: {period:.1f} Days"
        )
        st.subheader("🌍 Comparison With Earth")
        comparison = pd.DataFrame({
    "Property":[
        "Radius",
        "Distance From Star",
        "Orbital Period"
    ],
    "Earth":[
        1.0,
        1.0,
        365
    ],
    planet:[
        round(radius,2),
        round(distance,2),
        round(period,1)
    ]
})

st.dataframe(
    comparison,
    use_container_width=True
)
st.subheader("📈 Planet Radius Comparison")
fig, ax = plt.subplots()

ax.bar(
    ["Earth", planet],
    [1.0, radius]
)

ax.set_ylabel(
    "Earth Radii"
)

st.pyplot(fig)
        

    # ---------------- HABITABILITY REASONS ----------------

    st.subheader("🔬 Habitability Analysis")

    if 0.8 <= distance <= 1.5:

        st.success(
            "💧 Planet lies inside the Habitable Zone. Liquid water may exist."
        )

    else:

        st.warning(
            "💧 Distance from host star may reduce chances of liquid water."
        )

    if radius <= 2:

        st.success(
            "🌍 Planet is roughly Earth-sized."
        )

    else:

        st.warning(
            "🌍 Planet is significantly larger than Earth."
        )

    if 4500 <= temp <= 6500:

        st.success(
            "☀ Host star temperature is favorable."
        )

    else:

        st.warning(
            "☀ Host star temperature may be extreme."
        )

    st.info(
        "🌫 Atmospheric composition data unavailable."
    )

    # ---------------- AI REPORT ----------------

    st.subheader("🧠 AI Planet Report")

    if life_probability >= 80:

        st.success(
            f"""
            {planet} is one of the strongest candidates
            for habitability in the current dataset.

            The planet possesses Earth-like properties
            and may support liquid water.
            """
        )

    elif life_probability >= 50:

        st.info(
            f"""
            {planet} exhibits several potentially
            habitable characteristics.

            Further atmospheric observations are required.
            """
        )

    else:

        st.warning(
            f"""
            {planet} is unlikely to support life
            due to environmental conditions.
            """
        )

    # ---------------- LIFE FORMS ----------------

    st.subheader("👾 Hypothetical Life Forms")

    if life_probability >= 80:

        st.success(
            """
            Possible Life Forms:

            🦠 Microbial Life

            🌱 Primitive Plant Life

            🐟 Aquatic Ecosystems
            """
        )

    elif life_probability >= 50:

        st.info(
            """
            Possible Life Forms:

            🦠 Extremophile Microorganisms
            """
        )

    else:

        st.warning(
            """
            No known life forms are likely.
            """
        )

st.markdown("---")

st.caption(
    "Powered by Machine Learning and NASA Exoplanet Archive Data 🚀"
)
