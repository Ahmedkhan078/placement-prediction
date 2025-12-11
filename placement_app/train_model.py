# train_model file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_excel("AU_Student_Placement_Data.xlsx")


df = df.rename(columns={
    "SEX(M/F)": "Gender",
    "Board of Xth ": "XthBoard",
    "Board of XII": "XIIthBoard",
    "BACKLOG": "Backlog",
    "Btech": "Graduation",
    "Training Status": "Training",
    "Placement Status": "Placement",
    "Campus Drive attended": "CampusDrive",
    "Placement Year": "Year"
})


df = df.drop(columns=["Salary"])

df = df.dropna(subset=["Placement"])


categorical_cols = ["Gender", "Stream", "XthBoard", "XIIthBoard", "Origin"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le


X = df.drop(columns=["Placement"])
y = df["Placement"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc:.3f}")


bundle = {
    "model": model,
    "encoders": encoders,
    "feature_names": feature_names
}

joblib.dump(bundle, "placement_model.pkl")
print("Saved trained model to placement_model.pkl")
