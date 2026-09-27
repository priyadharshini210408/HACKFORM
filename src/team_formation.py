
import pandas as pd
import numpy as np
import joblib
from itertools import combinations


# ==================================================
# 1. LOAD DATA AND MODEL
# ==================================================

students = pd.read_csv("data/students.csv")

model = joblib.load(
    "models/team_compatibility_model.pkl"
)

print("Students loaded:", len(students))
print("Model loaded successfully!")


# ==================================================
# 2. SKILL COLUMNS
# ==================================================

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


# ==================================================
# 3. CALCULATE PAIR FEATURES
# ==================================================

def calculate_pair_features(student1, student2):

    differences = []

    for skill in skill_columns:
        difference = abs(
            float(student1[skill]) -
            float(student2[skill])
        )
        differences.append(difference)

    # Skill similarity
    average_difference = np.mean(differences)

    skill_similarity = (
        1 - average_difference / 9
    )

    # Skill complementarity
    complementary_skills = 0

    for skill in skill_columns:
        s1 = float(student1[skill])
        s2 = float(student2[skill])

        if (
            (s1 >= 7 and s2 <= 5)
            or
            (s2 >= 7 and s1 <= 5)
        ):
            complementary_skills += 1

    skill_complementarity = (
        complementary_skills / len(skill_columns)
    )

    # Domain match
    domain_match = int(
        student1["domain"] == student2["domain"]
    )

    # Experience balance
    experience_difference = abs(
        float(student1["experience"]) -
        float(student2["experience"])
    )

    experience_balance = max(
        0,
        1 - experience_difference / 4
    )

    return [
        skill_similarity,
        skill_complementarity,
        domain_match,
        experience_difference,
        experience_balance
    ]


# ==================================================
# 4. CREATE COMPATIBILITY MATRIX
# ==================================================

def create_compatibility_matrix():

    n = len(students)
    matrix = np.zeros((n, n))

    feature_rows = []
    pair_indices = []

    print("\nCalculating student compatibility...")

    # Generate every unique student pair
    for i in range(n):
        for j in range(i + 1, n):

            student1 = students.iloc[i]
            student2 = students.iloc[j]

            features = calculate_pair_features(
                student1, student2
            )

            feature_rows.append(features)
            pair_indices.append((i, j))

    # Predict all pairs in one batch
    if feature_rows:
        feature_df = pd.DataFrame(
            feature_rows,
            columns=[
                "skill_similarity",
                "skill_complementarity",
                "domain_match",
                "experience_difference",
                "experience_balance"
            ]
        )

        probabilities = model.predict_proba(
            feature_df
        )[:, 1]

        # Fill the symmetric matrix
        for probability, (i, j) in zip(
            probabilities, pair_indices
        ):
            matrix[i, j] = probability
            matrix[j, i] = probability

    print(
        f"Compatibility calculated for "
        f"{len(pair_indices)} pairs."
    )

    return matrix


# ==================================================
# 5. CALCULATE TEAM SCORE
# ==================================================

def calculate_team_score(
    team_indices,
    compatibility_matrix
):

    # Average pairwise ML compatibility
    pair_scores = [
        compatibility_matrix[i, j]
        for i, j in combinations(team_indices, 2)
    ]

    compatibility = (
        float(np.mean(pair_scores))
        if pair_scores else 0.0
    )

    # Skill diversity: proportion of skill areas
    # in which at least one member scores 7 or more
    diversity_values = []

    for skill in skill_columns:
        has_strong_member = any(
            float(students.iloc[index][skill]) >= 7
            for index in team_indices
        )

        diversity_values.append(
            int(has_strong_member)
        )

    skill_diversity = (
        sum(diversity_values) / len(skill_columns)
    )

    # Domain alignment: proportion belonging
    # to the team's most common domain
    domains = [
        students.iloc[index]["domain"]
        for index in team_indices
    ]

    domain_counts = pd.Series(domains).value_counts()

    domain_alignment = (
        float(domain_counts.max()) / len(team_indices)
    )

    # Overall team compatibility score
    final_score = (
        0.50 * compatibility
        + 0.30 * skill_diversity
        + 0.20 * domain_alignment
    )

    return (
        compatibility,
        skill_diversity,
        domain_alignment,
        final_score
    )


# ==================================================
# 6. GENERATE CANDIDATE TEAMS
# ==================================================

def generate_teams(
    team_size,
    compatibility_matrix,
    number_of_teams=10
):

    n = len(students)

    if team_size < 2 or team_size > n:
        return []

    if number_of_teams < 1:
        return []

    rng = np.random.default_rng(42)

    candidate_teams = {}

    # Generate different candidate teams
    attempts = min(5000, max(500, n * 50))

    # Calculate average compatibility for each student
    average_compatibility = (
        compatibility_matrix.sum(axis=1)
        / max(n - 1, 1)
    )

    # Use a mixture of strong and random starting students
    top_count = min(n, max(1, n // 2))

    top_students = np.argsort(
        average_compatibility
    )[::-1][:top_count]

    print("\nGenerating candidate teams...")

    for attempt in range(attempts):

        if attempt % 2 == 0:
            start = int(rng.choice(top_students))
        else:
            start = int(rng.integers(n))

        team = [start]

        available = [
            i for i in range(n) if i != start
        ]

        # Greedily add compatible members,
        # choosing randomly among the top five
        while len(team) < team_size:

            scores = []

            for candidate in available:

                score = float(np.mean([
                    compatibility_matrix[candidate, member]
                    for member in team
                ]))

                scores.append((candidate, score))

            if not scores:
                break

            scores.sort(
                key=lambda item: item[1],
                reverse=True
            )

            top_k = min(5, len(scores))

            selected_position = int(
                rng.integers(top_k)
            )

            best_student = scores[selected_position][0]

            team.append(best_student)
            available.remove(best_student)

        if len(team) != team_size:
            continue

        # Calculate the complete team score
        (
            compatibility,
            diversity,
            domain_alignment,
            final_score
        ) = calculate_team_score(
            team, compatibility_matrix
        )

        # Sorted IDs uniquely identify a team
        team_key = tuple(sorted(team))

        candidate_teams[team_key] = (
            team,
            compatibility,
            diversity,
            domain_alignment,
            final_score
        )

    # ==================================================
    # 7. SORT CANDIDATE TEAMS
    # ==================================================

    sorted_teams = sorted(
        candidate_teams.values(),
        key=lambda x: x[4],
        reverse=True
    )

    # ==================================================
    # 8. SELECT DISTINCT TEAMS
    # ==================================================

    selected_teams = []
    selected_keys = set()

    # First prefer teams with no shared members.
    # Gradually allow more overlap if necessary.
    for max_overlap in range(team_size):

        for candidate in sorted_teams:

            team = candidate[0]
            team_key = tuple(sorted(team))

            if team_key in selected_keys:
                continue

            # Check overlap with already selected teams
            acceptable = all(
                len(set(team) & set(existing[0]))
                <= max_overlap
                for existing in selected_teams
            )

            if acceptable:
                selected_teams.append(candidate)
                selected_keys.add(team_key)

            if len(selected_teams) >= number_of_teams:
                break

        if len(selected_teams) >= number_of_teams:
            break

    # Always return teams in descending score order
    selected_teams.sort(
        key=lambda x: x[4],
        reverse=True
    )

    print(
        f"\nGenerated {len(selected_teams)} "
        f"distinct teams."
    )

    return selected_teams


# ==================================================
# 9. TERMINAL TEST
# ==================================================

if __name__ == "__main__":

    print("\n================================")
    print(" HACKATHON TEAM FORMATION")
    print("================================")

    try:
        team_size = int(
            input("\nEnter desired team size: ")
        )

        number_of_teams = int(
            input("Enter number of teams: ")
        )

        if team_size < 2 or team_size > len(students):
            print("Invalid team size.")
            raise SystemExit

        if number_of_teams < 1:
            print("Number of teams must be positive.")
            raise SystemExit

        matrix = create_compatibility_matrix()

        teams = generate_teams(
            team_size,
            matrix,
            number_of_teams
        )

        print("\n================================")
        print(" RECOMMENDED TEAMS")
        print("================================")

        for i, team_info in enumerate(teams, start=1):

            (
                team,
                compatibility,
                diversity,
                domain_alignment,
                final_score
            ) = team_info

            print(f"\nTEAM {i}")
            print(
                f"Team Compatibility: "
                f"{final_score * 100:.2f}%"
            )
            print("-------------------------------")

            for index in team:
                student = students.iloc[index]

                print(
                    f"{student['student_id']} - "
                    f"{student['name']} - "
                    f"{student['domain']}"
                )

        if not teams:
            print("No teams could be generated.")

    except ValueError:
        print("Please enter valid whole numbers.")
