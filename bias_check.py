import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df = pd.read_csv("data/raw/heart.csv")

features = ["age","sex","chest_pain_type","resting_bp","cholestoral",
            "fasting_blood_sugar","restecg","max_hr","exang",
            "oldpeak","slope","num_major_vessels","thal"]

X = scaler.transform(df[features].values)
y = df["target"].values
preds = model.predict(X)

print("=== GENDER BIAS CHECK ===")
for sex, label in [(1,"Male"), (0,"Female")]:
    mask = df["sex"] == sex
    acc = (preds[mask] == y[mask]).mean()
    print(f"{label} ({mask.sum()} patients): Accuracy = {acc:.2%}")

print("\n=== AGE GROUP BIAS CHECK ===")
df["age_group"] = pd.cut(df["age"], bins=[20,40,55,70,90],
                          labels=["20-40","40-55","55-70","70+"])
for grp in ["20-40","40-55","55-70","70+"]:
    mask = df["age_group"] == grp
    if mask.sum() > 5:
        acc = (preds[mask] == y[mask]).mean()
        print(f"Age {grp} ({mask.sum()} patients): Accuracy = {acc:.2%}")

print("\n=== CLASS DISTRIBUTION ===")
print(df["target"].value_counts())
