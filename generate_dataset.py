import pandas as pd
import random

names = [
    "Arun", "Priya", "Kavin", "Divya", "Rahul",
    "Anjali", "Vijay", "Keerthi", "Sanjay", "Harini",
    "Aditya", "Swetha", "Gokul", "Nithya", "Vishnu",
    "Aishwarya", "Dinesh", "Deepika", "Surya", "Pavithra"
]

domains = [
    "AI",
    "Web",
    "Cloud",
    "Cybersecurity",
    "Data Science",
    "IoT",
    "App Development"
]

students = []

for i in range(100):

    student = {
        "student_id": f"S{i+1:03d}",
        "name": random.choice(names) + str(i + 1),

        "python": random.randint(1, 10),
        "java": random.randint(1, 10),
        "cpp": random.randint(1, 10),

        "machine_learning": random.randint(1, 10),
        "frontend": random.randint(1, 10),
        "backend": random.randint(1, 10),
        "database": random.randint(1, 10),
        "cloud": random.randint(1, 10),
        "ui_ux": random.randint(1, 10),

        "leadership": random.randint(1, 10),
        "communication": random.randint(1, 10),
        "problem_solving": random.randint(1, 10),

        "experience": random.randint(1, 5),

        "domain": random.choice(domains)
    }

    students.append(student)

df = pd.DataFrame(students)

df.to_csv("data/students.csv", index=False)

print("Dataset created successfully!")
print("Total students:", len(df))