import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NASA Exoplanet Intelligence System",
    page_icon="🚀",
    layout="wide"
)

# ---------------- PLANET IMAGES ----------------

planet_images = {
    "Earth":"https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg",
    "Mars":"https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg",
    "Venus":"https://upload.wikimedia.org/wikipedia/commons/e/e5/Venus-real_color.jpg",
    "Kepler-442 b":"https://upload.wikimedia.org/wikipedia/commons/5/58/Kepler-442b_artist_impression.jpg",
    "Kepler-186 f":"https://upload.wikimedia.org/wikipedia/commons/4/46/PIA18034-Kepler186f.jpg",
    "TRAPPIST-1 e":"https://upload.wikimedia.org/wikipedia/commons/d/d9/TRAPPIST-1e_artist_concept.jpg"
}

# ---------------- STYLING ----------------

st.markdown("""
<style>

.stApp{
    background:
    radial-gradient(circle at top,#001219,#000814,#000000);
    color:#00ffff;
}

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:#00ffff;
    text-shadow:0px 0px 20px #00ffff;
}

.scanner-box{
    background:#001d3d;
    border:2px solid #00ffff;
    border-radius:15px;
    padding:20px;
    margin-bottom:20px;
    box-shadow:0px 0px 20px #00ffff;
}

.scanner-text{
    color:#00ffff;
    font-family:monospace;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL & DATA ----------------

model = joblib.load("model.pkl")
df = pd.read_csv("nasa_clean.csv")

# ---------------- TITLE ----------------

st.markdown("""
<div class="main-title">
👽 EXOPLANET SCANNER SYSTEM
</div>
""", unsafe_allow_html=True)

st.caption(
"INTERSTELLAR RECONNAISSANCE NETWORK • PLANET ANALYSIS ENGINE"
)

st.write(
    "Analyze real exoplanets from NASA's Exoplanet Archive using Machine Learning."
)

# ---------------- PLANET SELECTION ----------------

planet_names = sorted(df["pl_name"].unique())

planet = st.selectbox(
    "🔍 Select a Planet",
    planet_names
)

selected = df[df["pl_name"] == planet].iloc[0]

radius = selected["pl_rade"]
period = selected["pl_orbper"]
distance = selected["pl_orbsmax"]
temp = selected["st_teff"]

# ---------------- ANALYZE ----------------

if st.button("🛰 INITIATE PLANET SCAN"):
    scan = st.progress(0)

    for i in range(100):
        scan.progress(i+1)

    st.success("🎯 TARGET LOCK ACQUIRED")

    prediction = model.predict(
        [[radius, period, distance, temp]]
    )[0]

    probability = model.predict_proba(
        [[radius, period, distance, temp]]
    )[0][1]

    score = int(probability * 100)

    st.header(f"🪐 {planet}")

    # Planet Image

    if planet in planet_images:
        st.image(
            planet_images[planet],
            width=500
        )

    # Earth Similarity Index

    esi = max(
        0,
        round(
            1 - abs(radius - 1) / 4,
            2
        )
    )

    life_probability = int(
        (score + esi * 100) / 2
    )

    # ---------------- METRICS ----------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Habitability Score",
            f"{score}%"
        )

    with col2:
        st.metric(
            "🌍 Earth Similarity",
            f"{esi}"
        )

    with col3:
        st.metric(
            "👽 Life Probability",
            f"{life_probability}%"
        )

    st.progress(life_probability)

    if life_probability >= 80:
    
        st.success(
            "👽 THREAT LEVEL: POSSIBLE ADVANCED LIFE"
        )
    
    elif life_probability >= 50:
    
        st.warning(
            "🦠 THREAT LEVEL: MICROBIAL LIFE POSSIBLE"
        )
    
    else:
    
        st.error(
            "☠ NO LIFE SIGNATURES DETECTED"
        )
    

    # ---------------- RESULT ----------------
    if prediction == 1:

        st.markdown(f"""
        <div class="scanner-box">
        <h1>🟢 HABITABLE WORLD DETECTED</h1>
        <h2>{planet}</h2>
        </div>
        """, unsafe_allow_html=True)

    else:
    
        st.markdown(f"""
        <div class="scanner-box">
        <h1>🔴 HOSTILE WORLD</h1>
        <h2>{planet}</h2>
        </div>
        """, unsafe_allow_html=True)
    # ---------------- PLANET FACTS ----------------

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

    # ---------------- EARTH COMPARISON ----------------

    st.subheader("🌍 Comparison With Earth")

    comparison = pd.DataFrame({
        "Property": [
            "Radius",
            "Distance From Star",
            "Orbital Period"
        ],
        "Earth": [
            1.0,
            1.0,
            365
        ],
        planet: [
            round(radius, 2),
            round(distance, 2),
            round(period, 1)
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    # ---------------- CHART ----------------

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

    # ---------------- HABITABLE ZONE ----------------

    st.subheader("🌞 Habitable Zone Visualization")

    fig2, ax2 = plt.subplots()

    ax2.axvspan(
        0.8,
        1.5,
        alpha=0.3,
        label="Habitable Zone"
    )

    ax2.scatter(
        distance,
        1,
        s=200,
        label=planet
    )

    ax2.set_xlabel(
        "Distance From Star (AU)"
    )

    ax2.set_yticks([])

    ax2.legend()

    st.pyplot(fig2)

    # ---------------- ANALYSIS ----------------

    st.subheader("🔬 Habitability Analysis")

    if 0.8 <= distance <= 1.5:
        st.success(
            "💧 Possible Liquid Water"
        )
    else:
        st.warning(
            "💧 Water Unlikely"
        )

    if radius <= 2:
        st.success(
            "🌍 Earth-like Size"
        )
    else:
        st.warning(
            "🌍 Larger than Earth"
        )

    if 4500 <= temp <= 6500:
        st.success(
            "☀ Suitable Star Temperature"
        )
    else:
        st.warning(
            "☀ Extreme Star Temperature"
        )

    st.info(
        "🌫 Atmosphere Data Unknown"
    )

    # ---------------- AI REPORT ----------------

        # ---------------- AI REPORT ----------------

    report = f"""
TARGET: {planet}

HOST STAR: {selected['hostname']}

MISSION STATUS:
{'HABITABLE' if prediction == 1 else 'HOSTILE'}

LIFE PROBABILITY:
{life_probability}%

SCANNER RECOMMENDATION:

{'Priority candidate for future exploration missions.' if prediction == 1 else 'Low-priority exploration target.'}
"""

    st.code(report)    
    # ---------------- LIFE FORMS ----------------

    st.subheader("👾 Hypothetical Life Forms")

    if life_probability >= 80:

        st.success("""
🦠 Microbial Life

🌱 Primitive Plant Life

🐟 Aquatic Ecosystems
""")

    elif life_probability >= 50:

        st.info("""
🦠 Extremophile Microorganisms
""")

    else:

        st.warning("""
No known life forms are likely.
""")

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Powered by Machine Learning and NASA Exoplanet Archive Data 🚀"
)
