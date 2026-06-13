import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("nasa_clean.csv")

X = df[
    [
        "pl_rade",
        "pl_orbper",
        "pl_orbsmax",
        "st_teff"
    ]
]

y = df["habitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(
    f"Accuracy: {accuracy:.2f}"
)

joblib.dump(
    model,
    "model.pkl"
)

print("Model saved!")
