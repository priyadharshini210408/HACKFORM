import pandas as pd
from itertools import combinations

# Load student dataset
df = pd.read_csv("data/students.csv")

# Skills used for team compatibility
skill_columns = [
    "python",
    "java",
    "cpp",
    "machine_learning",
    "frontend",
    "backend",
    "database",
    "cloud",
    "ui_ux"
]

pair_data = []

# Generate every possible pair of students
for i, j in combinations(range(len(df)), 2):

    student1 = df.iloc[i]
    student2 = df.iloc[j]

    # ---------------------------------------
    # 1. Skill Similarity
    # ---------------------------------------

    differences = []

    for skill in skill_columns:
        difference = abs(student1[skill] - student2[skill])
        differences.append(difference)

    average_difference = sum(differences) / len(differences)

    skill_similarity = 1 - (average_difference / 9)

    # ---------------------------------------
    # 2. Skill Complementarity
    # ---------------------------------------

    complementary_skills = 0

    for skill in skill_columns:

        # One student is strong while
        # the other student is relatively weak
        if (
            (student1[skill] >= 7 and student2[skill] <= 5)
            or
            (student2[skill] >= 7 and student1[skill] <= 5)
        ):
            complementary_skills += 1

    skill_complementarity = (
        complementary_skills / len(skill_columns)
    )

    # ---------------------------------------
    # 3. Domain Match
    # ---------------------------------------

    if student1["domain"] == student2["domain"]:
        domain_match = 1
    else:
        domain_match = 0

    # ---------------------------------------
    # 4. Experience Balance
    # ---------------------------------------

    experience_difference = abs(
        student1["experience"] -
        student2["experience"]
    )

    experience_balance = 1 - (
        experience_difference / 4
    )

    # ---------------------------------------
    # 5. Compatibility Score
    # ---------------------------------------

    compatibility_score = (
        0.50 * skill_complementarity
        + 0.30 * domain_match
        + 0.20 * experience_balance
    )

    # ---------------------------------------
    # 6. Create Label
    # ---------------------------------------

    if compatibility_score >= 0.50:
        compatible = 1
    else:
        compatible = 0

    # ---------------------------------------
    # Save pair information
    # ---------------------------------------

    pair_data.append({
        "student1_id": student1["student_id"],
        "student2_id": student2["student_id"],

        "skill_similarity": round(
            skill_similarity, 3
        ),

        "skill_complementarity": round(
            skill_complementarity, 3
        ),

        "domain_match": domain_match,

        "experience_difference":
            experience_difference,

        "experience_balance": round(
            experience_balance, 3
        ),

        "compatibility_score": round(
            compatibility_score, 3
        ),

        "compatible": compatible
    })


# Convert to DataFrame
pairs_df = pd.DataFrame(pair_data)

# Save training data
pairs_df.to_csv(
    "data/pair_training_data.csv",
    index=False
)

print("Pair training dataset created successfully!")

print("Total student pairs:", len(pairs_df))

print("\nFirst 5 pairs:")
print(pairs_df.head())

print("\nCompatible distribution:")
print(pairs_df["compatible"].value_counts())