
from src.team_formation import (
    create_compatibility_matrix,
    generate_teams
)

def evaluate_teams(team_size=4, number_of_teams=10):
    matrix = create_compatibility_matrix()

    teams = generate_teams(
        team_size,
        matrix,
        number_of_teams
    )

    if not teams:
        print("No teams generated.")
        return

    print("\nTEAM EVALUATION")
    print("----------------")

    scores = []
    team_sets = set()

    for i, team_info in enumerate(teams, start=1):
        team, compatibility, diversity, domain, final = team_info

        scores.append(final)
        team_sets.add(tuple(sorted(team)))

        print(f"\nTeam {i}")
        print(f"Compatibility: {compatibility:.2f}%")
        print(f"Skill Diversity: {diversity:.2f}%")
        print(f"Domain Alignment: {domain:.2f}%")
        print(f"Final Score: {final:.2f}%")

    print("\nOVERALL RESULTS")
    print("----------------")
    print("Teams generated:", len(teams))
    print("Unique teams:", len(team_sets))
    print(f"Average score: {sum(scores)/len(scores):.2f}%")
    print(f"Highest score: {max(scores):.2f}%")
    print(f"Lowest score: {min(scores):.2f}%")

if __name__ == "__main__":
    evaluate_teams()
