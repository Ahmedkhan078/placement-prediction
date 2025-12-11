import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_excel("AU_Student_Placement_Data.xlsx")

# Remove Salary column (not needed)
df = df.drop(columns=["Salary"])

# Dataset column names (REAL from Excel)
numeric_cols = [
    "Xth", "XIIth", "BACKLOG", "Btech",
    "Training Status", "Campus Drive attended",
    "Placement Year"
]

categorical_cols = [
    "SEX(M/F)", "Stream", "Board of Xth ",
    "Board of XII", "Origin"
]

target_col = "Placement Status"

X = df[numeric_cols + categorical_cols]
y = df[target_col]

# Numeric imputer
num_imp = SimpleImputer(strategy="median")
X[numeric_cols] = num_imp.fit_transform(X[numeric_cols])

# Categorical encoder
enc = OrdinalEncoder()
X[categorical_cols] = enc.fit_transform(X[categorical_cols].astype(str))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save everything
bundle = {
    "model": model,
    "enc": enc,
    "num_imp": num_imp,
    "feature_names": X.columns.tolist()
}

joblib.dump(bundle, "placement_model.pkl")

print("🎉 Model trained and saved successfully!")
