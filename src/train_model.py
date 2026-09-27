import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# -----------------------------------------
# 1. Load training dataset
# -----------------------------------------

df = pd.read_csv("data/pair_training_data.csv")

print("Training dataset loaded!")
print("Number of records:", len(df))


# -----------------------------------------
# 2. Select input features
# -----------------------------------------

features = [
    "skill_similarity",
    "skill_complementarity",
    "domain_match",
    "experience_difference",
    "experience_balance"
]

X = df[features]

# Target
y = df["compatible"]


# -----------------------------------------
# 3. Split data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------------
# 4. Create Random Forest model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------------------
# 5. Train model
# -----------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# -----------------------------------------
# 6. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test)


# -----------------------------------------
# 7. Calculate accuracy
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


# -----------------------------------------
# 8. Classification report
# -----------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------------
# 9. Confusion Matrix
# -----------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------------
# 10. Save the model
# -----------------------------------------

joblib.dump(
    model,
    "models/team_compatibility_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Location: models/team_compatibility_model.pkl"
)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Predict on the test dataset
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
