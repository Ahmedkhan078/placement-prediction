import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Disable GUI backend for Matplotlib (IMPORTANT for Flask)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Flask App
app = Flask(__name__)

# Load trained model bundle
bundle = joblib.load("placement_model.pkl")
model = bundle["model"]
enc = bundle["enc"]           # categorical encoder
feature_names = bundle["feature_names"]
num_imp = bundle["num_imp"]   # numeric imputer

# Define columns
numeric_cols = ["Xth", "XIIth", "Backlog", "Graduation", "Training", "CampusDrive", "Year"]
categorical_cols = ["Gender", "Stream", "XthBoard", "XIIthBoard", "Origin"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Collect user input into dictionary
        data = {
            "Gender": form.get("Gender"),
            "Stream": form.get("Stream"),
            "Xth": float(form.get("Xth")),
            "XthBoard": form.get("XthBoard"),
            "XIIth": float(form.get("XIIth")),
            "XIIthBoard": form.get("XIIthBoard"),
            "Backlog": float(form.get("Backlog")),
            "Graduation": float(form.get("Graduation")),
            "Training": float(form.get("Training")),
            "CampusDrive": float(form.get("CampusDrive")),
            "Year": float(form.get("Year")),
            "Origin": form.get("Origin")
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Handle missing values for numeric columns (using saved imputer)
        input_df[numeric_cols] = num_imp.transform(input_df[numeric_cols])

        # Encode categorical features using saved encoder
        input_df[categorical_cols] = enc.transform(input_df[categorical_cols].astype(str))

        # Predict placement
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        # Create feature importance graph
        importances = model.feature_importances_
        plt.figure(figsize=(8, 4))
        plt.bar(feature_names, importances)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save static image
        if not os.path.exists("static"):
            os.makedirs("static")

        plt.savefig("static/feature_importance.png")
        plt.close()

        # Convert prediction to message
        if prediction == 1:
            result_text = "Student is LIKELY to be PLACED 🎉"
        else:
            result_text = "Student is NOT LIKELY to be PLACED ❌"

        return render_template("index.html",
                               prediction=result_text,
                               probability=f"{probability:.2f}%")

    except Exception as e:
        return render_template("index.html",
                               prediction="Error occurred",
                               probability=str(e))


# For deployment: host must be 0.0.0.0
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
