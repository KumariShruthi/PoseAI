import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load features
features = pd.read_csv("../pose_features.csv")

# Load labels
labels = pd.read_csv("../pose_labels.csv")

# Create matching key
features["key"] = features["category"] + "\\" + features["image"]
labels["key"] = labels["category"] + "\\" + labels["image"]

# Merge features and labels
data = features.merge(
    labels[["key", "pose_label"]],
    on="key",
    how="inner"
)

# Features used by the ML model
feature_columns = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "shoulder_angle",
    "hip_angle",
    "left_wrist_shoulder_distance",
    "right_wrist_shoulder_distance",
    "left_ankle_hip_distance",
    "right_ankle_hip_distance"
]

X = data[feature_columns]
y = data["pose_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
joblib.dump(model, "../pose_model.pkl")

print("\nModel saved to: ../pose_model.pkl")