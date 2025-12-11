import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Use safe backend for Render
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the saved model bundle
bundle = joblib.load("placement_model.pkl")
model = bundle["model"]
enc = bundle["enc"]
num_imp = bundle["num_imp"]
feature_names = bundle["feature_names"]

# Exact training order of features
numeric_cols = [
    "Xth", "XIIth", "BACKLOG", "Btech",
    "Training Status", "Campus Drive attended",
    "Placement Year"
]

categorical_cols = [
    "SEX(M/F)", "Stream", "Board of Xth ",
    "Board of XII", "Origin"
]

all_features = numeric_cols + categorical_cols


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Map HTML → Dataset column names
        data = {
            "Xth": float(form.get("Xth")),
            "XIIth": float(form.get("XIIth")),
            "BACKLOG": float(form.get("Backlog")),
            "Btech": float(form.get("Graduation")),
            "Training Status": float(form.get("Training")),
            "Campus Drive attended": float(form.get("CampusDrive")),
            "Placement Year": float(form.get("Year")),

            "SEX(M/F)": form.get("Gender"),
            "Stream": form.get("Stream"),
            "Board of Xth ": form.get("XthBoard"),
            "Board of XII": form.get("XIIthBoard"),
            "Origin": form.get("Origin"),
        }

        # Create DataFrame in correct column order
        df = pd.DataFrame([data])[all_features]

        # Apply preprocessing: Imputation + Encoding
        df[numeric_cols] = num_imp.transform(df[numeric_cols])
        df[categorical_cols] = enc.transform(df[categorical_cols].astype(str))

        # Prediction
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        # Graph: Feature Importance
        importances = model.feature_importances_
        plt.figure(figsize=(8, 4))
        plt.bar(feature_names, importances)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if not os.path.exists("static"):
            os.makedirs("static")

        plt.savefig("static/feature_importance.png")
        plt.close()

        # Result text
        result_msg = "Student is LIKELY to be PLACED 🎉" if pred == 1 else "Student is NOT likely to be placed ❌"

        return render_template("index.html",
                               prediction=result_msg,
                               probability=f"{prob:.2f}%")

    except Exception as e:
        return render_template("index.html",
                               prediction="ERROR OCCURRED",
                               probability=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
