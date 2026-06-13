import pandas as pd

df = pd.read_csv(
    "PS_2026.06.13_01.19.38.csv",
    comment="#"
)

df = df[
    [
        "pl_name",
        "hostname",
        "disc_year",
        "pl_rade",
        "pl_orbper",
        "pl_orbsmax",
        "st_teff"
    ]
]

df = df.dropna()

df["habitable"] = (
    (df["pl_orbsmax"] >= 0.8)
    & (df["pl_orbsmax"] <= 1.5)
    & (df["pl_rade"] <= 2)
    & (df["st_teff"] >= 4500)
    & (df["st_teff"] <= 6500)
).astype(int)

df.to_csv(
    "nasa_clean.csv",
    index=False
)

print("Planets:", len(df))
print("Saved nasa_clean.csv")