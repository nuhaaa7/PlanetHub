import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("exoplanets.csv")

df["habitable"] = (
    (df["pl_orbsmax"] >= 0.8)
    & (df["pl_orbsmax"] <= 1.5)
    & (df["pl_rade"] <= 2)
    & (df["st_teff"] >= 4500)
    & (df["st_teff"] <= 6500)
).astype(int)

X = df[
    [
        "pl_rade",
        "pl_orbper",
        "st_teff",
        "pl_orbsmax"
    ]
]


y = df["habitable"]

model = RandomForestClassifier()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("exoplanets.csv")

df["habitable"] = (
    (df["pl_orbsmax"] >= 0.8)
    & (df["pl_orbsmax"] <= 1.5)
    & (df["pl_rade"] <= 2)
    & (df["st_teff"] >= 4500)
    & (df["st_teff"] <= 6500)
).astype(int)

X = df[
    [
        "pl_rade",
        "pl_orbper",
        "st_teff",
        "pl_orbsmax"
    ]
]

y = df["habitable"]

model = RandomForestClassifier()

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model Trained!")