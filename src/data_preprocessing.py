import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# 1. Load the dataset
# -----------------------------------

df = pd.read_csv("data/students.csv")

print("Dataset loaded successfully!")
print("Number of students:", len(df))

# -----------------------------------
# 2. Display basic information
# -----------------------------------

print("\nFirst 5 students:")
print(df.head())

print("\nDataset information:")
print(df.info())

# -----------------------------------
# 3. Check missing values
# -----------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------------
# 4. Remove duplicate students
# -----------------------------------

df = df.drop_duplicates(subset="student_id")

# -----------------------------------
# 5. Convert domain into numbers
# -----------------------------------

domain_mapping = {
    "AI": 0,
    "Web": 1,
    "Cloud": 2,
    "Cybersecurity": 3,
    "Data Science": 4,
    "IoT": 5,
    "App Development": 6
}

df["domain_encoded"] = df["domain"].map(domain_mapping)

# -----------------------------------
# 6. Select numerical features
# -----------------------------------

features = [
    "python",
    "java",
    "cpp",
    "machine_learning",
    "frontend",
    "backend",
    "database",
    "cloud",
    "ui_ux",
    "leadership",
    "communication",
    "problem_solving",
    "experience",
    "domain_encoded"
]

X = df[features]

# -----------------------------------
# 7. Scale the features
# -----------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

# Keep student ID and name
X_scaled.insert(0, "student_id", df["student_id"].values)
X_scaled.insert(1, "name", df["name"].values)

# -----------------------------------
# 8. Save processed dataset
# -----------------------------------

X_scaled.to_csv(
    "data/processed_students.csv",
    index=False
)

print("\nPreprocessing completed!")
print("Processed dataset saved successfully.")

print("\nProcessed data:")
print(X_scaled.head())