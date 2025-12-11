import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Use safe backend for Render
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model bundle
bundle = joblib.load("placement_model.pkl")
model = bundle["model"]
enc = bundle["enc"]
num_imp = bundle["num_imp"]
feature_names = bundle["feature_names"]

# Dataset columns
numeric_cols = [
    "Xth", "XIIth", "BACKLOG", "Btech",
    "Training Status", "Campus Drive attended",
    "Placement Year"
]

categorical_cols = [
    "SEX(M/F)", "Stream", "Board of Xth ",
    "Board of XII", "Origin"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Map HTML → Dataset columns
        data = {
            "SEX(M/F)": form.get("Gender"),
            "Stream": form.get("Stream"),
            "Xth": float(form.get("Xth")),
            "Board of Xth ": form.get("XthBoard"),
            "XIIth": float(form.get("XIIth")),
            "Board of XII": form.get("XIIthBoard"),
            "BACKLOG": float(form.get("Backlog")),
            "Btech": float(form.get("Graduation")),
            "Training Status": float(form.get("Training")),
            "Campus Drive attended": float(form.get("CampusDrive")),
            "Placement Year": float(form.get("Year")),
            "Origin": form.get("Origin")
        }

        df = pd.DataFrame([data])

        df[numeric_cols] = num_imp.transform(df[numeric_cols])
        df[categorical_cols] = enc.transform(df[categorical_cols].astype(str))

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100

        # Feature importance
        importances = model.feature_importances_
        plt.figure(figsize=(8, 4))
        plt.bar(feature_names, importances)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if not os.path.exists("static"):
            os.makedirs("static")

        plt.savefig("static/feature_importance.png")
        plt.close()

        if prediction == 1:
            result = "Student is LIKELY to be PLACED 🎉"
        else:
            result = "Student is NOT likely to be placed ❌"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability:.2f}%"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="ERROR OCCURRED",
            probability=str(e)
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
