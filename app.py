import streamlit as st
import pandas as pd
import src.team_formation as team_module

st.set_page_config(
    page_title="Hackathon Team Formation",
    layout="wide"
)

st.title("Hackathon Team Formation using Machine Learning")
st.write("Generate compatible teams for hackathons.")

# Load student dataset
st.header("1. Student Dataset")

uploaded_file = st.file_uploader(
    "Upload Student CSV",
    type=["csv"]
)

if uploaded_file is not None:
    students = pd.read_csv(uploaded_file)
else:
    students = pd.read_csv("data/students.csv")

# Validate dataset
required_columns = [
    "student_id",
    "experience",
    "domain"
]

skill_columns = [
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
    "problem_solving"
]

missing = [
    col for col in required_columns
    if col not in students.columns
]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

if len(students) < 2:
    st.error("At least two students are required.")
    st.stop()

if students["student_id"].isna().any():
    st.error("Student IDs cannot be empty.")
    st.stop()

if students["student_id"].duplicated().any():
    st.error("Duplicate student IDs found.")
    st.stop()

try:
    students["experience"] = pd.to_numeric(
        students["experience"]
    )

    for col in skill_columns:
        if col in students.columns:
            students[col] = pd.to_numeric(students[col])

except (ValueError, TypeError):
    st.error("Experience and skill values must be numeric.")
    st.stop()

if students["experience"].isna().any():
    st.error("Experience cannot be empty.")
    st.stop()

if (students["experience"] < 0).any():
    st.error("Experience cannot be negative.")
    st.stop()

if students["domain"].isna().any():
    st.error("Domain cannot be empty.")
    st.stop()

available_skills = [
    col for col in skill_columns
    if col in students.columns
]

if not available_skills:
    st.error("No skill columns found in the dataset.")
    st.stop()

for col in available_skills:
    if students[col].isna().any():
        st.error(f"Empty values found in {col}.")
        st.stop()

    if ((students[col] < 0) | (students[col] > 10)).any():
        st.error(f"Skill values in {col} must be between 0 and 10.")
        st.stop()

students = students.reset_index(drop=True)
team_module.students = students

st.success("Dataset loaded successfully!")

st.subheader("Student Details")
st.dataframe(students, use_container_width=True)

st.write("Total Students:", len(students))

# Team configuration
st.header("2. Team Configuration")

team_size = st.number_input(
    "Select Team Size",
    min_value=2,
    max_value=len(students),
    value=min(4, len(students)),
    step=1
)

number_of_teams = st.number_input(
    "Number of Teams",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

# Generate teams
st.header("3. Generate Teams")

if st.button("Generate Teams"):

    try:
        with st.spinner("Generating teams..."):

            team_module.students = students.copy()

            compatibility_matrix = (
                team_module.create_compatibility_matrix()
            )

            generated_teams = team_module.generate_teams(
                team_size=int(team_size),
                compatibility_matrix=compatibility_matrix,
                number_of_teams=int(number_of_teams)
            )

            if generated_teams:
                st.session_state["generated_teams"] = generated_teams
                st.session_state["team_students"] = students.copy()
                st.success("Teams generated successfully!")
            else:
                st.warning("No teams could be generated.")

    except Exception as e:
        st.error(f"Error generating teams: {e}")

# Display teams
if "generated_teams" in st.session_state:

    st.header("4. Recommended Teams")

    generated_teams = st.session_state["generated_teams"]
    result_students = st.session_state["team_students"]

    def get_student_name(student):
        if isinstance(student, (dict, pd.Series)):
            return str(
                student.get(
                    "name",
                    student.get("student_id", "Student")
                )
            )

        if isinstance(student, int):
            if 0 <= student < len(result_students):
                row = result_students.iloc[student]
                return str(row.get("name", row["student_id"]))

        matches = result_students[
            result_students["student_id"].astype(str)
            == str(student)
        ]

        if not matches.empty:
            return str(
                matches.iloc[0].get("name", student)
            )

        return str(student)

    def to_percentage(value):
        value = float(value)

        if 0 <= value <= 1:
            return value * 100

        return value

    team_summary = []
    team_details = []

    for i, result in enumerate(generated_teams, start=1):

        team, compatibility, diversity, domain_alignment, final_score = result

        if isinstance(team, pd.DataFrame):
            if "name" in team.columns:
                members = team["name"].astype(str).tolist()
            else:
                members = team["student_id"].astype(str).tolist()
        else:
            members = [
                get_student_name(student)
                for student in team
            ]

        compatibility = to_percentage(compatibility)
        diversity = to_percentage(diversity)
        domain_alignment = to_percentage(domain_alignment)
        final_score = to_percentage(final_score)

        with st.expander(
            f"Team {i} - {final_score:.2f}%",
            expanded=(i == 1)
        ):
            st.subheader(f"Team {i}")

            st.write("Team Members:")

            for member in members:
                st.write(f"- {member}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "ML Compatibility",
                    f"{compatibility:.2f}%"
                )

                st.metric(
                    "Skill Diversity",
                    f"{diversity:.2f}%"
                )

            with col2:
                st.metric(
                    "Domain Alignment",
                    f"{domain_alignment:.2f}%"
                )

                st.metric(
                    "Final Team Score",
                    f"{final_score:.2f}%"
                )

        team_summary.append({
            "Team": i,
            "Members": ", ".join(members),
            "Team Size": len(members),
            "ML Compatibility (%)": round(compatibility, 2),
            "Skill Diversity (%)": round(diversity, 2),
            "Domain Alignment (%)": round(domain_alignment, 2),
            "Final Team Score (%)": round(final_score, 2)
        })

        for member in members:
            team_details.append({
                "Team": i,
                "Student": member,
                "Final Team Score (%)": round(final_score, 2)
            })

    # Team summary
    st.header("5. Team Summary")

    summary_df = pd.DataFrame(team_summary)
    st.dataframe(summary_df, use_container_width=True)

    # Download results
    st.header("6. Download Results")

    details_df = pd.DataFrame(team_details)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Team Details",
            data=details_df.to_csv(index=False),
            file_name="hackathon_teams.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="Download Team Summary",
            data=summary_df.to_csv(index=False),
            file_name="team_summary.csv",
            mime="text/csv"
        )

st.caption(
    "Team scores represent estimated compatibility, "
    "not actual hackathon winning probabilities."
)