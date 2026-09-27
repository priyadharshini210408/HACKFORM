# Hackathon Team Formation Using Machine Learning

## Project Overview
This project uses Machine Learning to form compatible teams for hackathons based on students' technical skills, experience and domain interests.

The application uses a Random Forest Classifier to estimate compatibility between students and generates multiple possible teams based on their compatibility scores.

## Features
- Upload student data using CSV files.
- Validate student details and skill levels.
- Select the required team size.
- Generate multiple compatible teams.
- Rank generated teams by their overall team score.
- Display ML compatibility, skill diversity and domain alignment.
- Download team details and team summaries as CSV files.

## Technologies Used
- Python
- Machine Learning
- Random Forest Classifier
- Pandas
- NumPy
- Streamlit
- Scikit-learn
- Joblib

## Project Structure

```text
hackathon_team_formation/
│
├── data/
│   └── students.csv
│
├── models/
│   └── team_compatibility_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── generate_pairs.py
│   ├── train_model.py
│   └── team_formation.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project folder.

```bash
cd hackathon_team_formation
```

3. Install the required libraries.

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application.

```bash
python -m streamlit run app.py
```

## How to Use

1. Run the Streamlit application.
2. Upload the student dataset or use the default CSV file.
3. Select the desired team size.
4. Choose the number of teams to generate.
5. Click the Generate Teams button.
6. View the generated teams and their compatibility scores.
7. Download the team details and summary as CSV files.

## Machine Learning Approach

The project uses a Random Forest Classifier to estimate pairwise student compatibility.

The team formation process considers:
- Technical skill complementarity
- Skill similarity
- Domain matching
- Experience balance

The final team score combines ML compatibility, skill diversity and domain alignment.

The displayed score is a heuristic compatibility measure, not a validated probability of winning a hackathon.

## Future Enhancements
- Improve team formation using optimization algorithms.
- Support more skill categories.
- Add user authentication.
- Deploy the application online.
- Evaluate the model using real hackathon team outcomes.

## Author
Priyadharshini

## License
This project is available for educational and learning purposes.
