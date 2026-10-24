"""
Generate a realistic (and deliberately MESSY) synthetic dataset of CS students.

The data is split across three files in three formats, just like real life:
    data/students.csv      -> habits & background   (CSV)
    data/coursework.json   -> coursework activity   (JSON)
    data/exams.xlsx        -> exam results          (Excel)

Mess we inject on purpose (so we can practice cleaning it):
    * missing values (NaN) in several columns
    * duplicate rows
    * inconsistent category spelling ("ai/ml", "  AI/ML ")
    * numbers stored as text ("87.5%")
    * impossible outliers (studying 150 hours a week, 0 hours of sleep)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRACKS = ["AI/ML", "Systems", "Web Dev", "Security", "Theory"]
TRACK_DIFFICULTY = {"AI/ML": -2.0, "Systems": -4.0, "Web Dev": 2.0, "Security": 0.0, "Theory": -5.0}


def generate(n: int = 1500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # ---------- independent features (vectorized, no loops) ----------
    student_id = np.arange(1001, 1001 + n)
    track = rng.choice(TRACKS, size=n, p=[0.30, 0.20, 0.25, 0.15, 0.10])
    year = rng.integers(1, 5, size=n)
    enrollment_date = pd.Timestamp("2022-08-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")

    study_hours = np.clip(rng.normal(14, 6, n), 0, 45)
    sleep_hours = np.clip(rng.normal(6.8, 1.2, n), 3.5, 10)
    screen_time = np.clip(rng.normal(5, 2, n), 0.5, 14)
    part_time_job = rng.random(n) < 0.35
    prior_gpa = np.clip(rng.normal(3.0, 0.5, n), 1.5, 4.0)
    coffee_cups = rng.poisson(2, n)

    # ---------- correlated features ----------
    attendance = np.clip(rng.normal(78, 12, n) + (study_hours - 14) * 0.6 - part_time_job * 5, 20, 100)
    leetcode_solved = rng.poisson(np.maximum(study_hours * 4, 1))
    github_commits = rng.poisson(np.maximum(15 + study_hours * 1.5, 1))
    assignment_avg = np.clip(45 + study_hours * 1.1 + prior_gpa * 6 + rng.normal(0, 8, n), 0, 100)

    # ---------- the hidden "truth" that decides the final score ----------
    sleep_effect = -1.8 * (sleep_hours - 7.5) ** 2                 # inverted U: too little OR too much sleep hurts
    coffee_effect = np.where(coffee_cups <= 3, coffee_cups, 3 - (coffee_cups - 3) * 2.5)  # a little helps, a lot hurts
    track_effect = np.vectorize(TRACK_DIFFICULTY.get)(track)
    final_score = (
        12
        + 0.9 * study_hours
        + 0.22 * attendance
        + 6.0 * prior_gpa
        + 0.25 * assignment_avg
        + 0.03 * leetcode_solved
        + sleep_effect
        - 1.1 * screen_time
        - 4.0 * part_time_job
        + coffee_effect
        + track_effect
        + rng.normal(0, 5, n)
    )
    # shift so that ~65% of students pass (score >= 60)
    final_score = np.clip(final_score + (60 - np.quantile(final_score, 0.35)), 0, 100)
    midterm = np.clip(0.75 * final_score + 0.25 * rng.normal(65, 15, n), 0, 100)
    exam_date = pd.Timestamp("2025-12-10") + pd.to_timedelta(rng.integers(0, 10, n), unit="D")

    students = pd.DataFrame({
        "student_id": student_id,
        "track": track,
        "year": year,
        "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
        "study_hours_week": study_hours.round(1),
        "sleep_hours": sleep_hours.round(1),
        "screen_time_hours": screen_time.round(1),
        "part_time_job": np.where(part_time_job, "Yes", "No"),
        "prior_gpa": prior_gpa.round(2),
        "coffee_cups_day": coffee_cups.astype(float),
        "attendance": [f"{a:.1f}%" for a in attendance],     # numbers stored as text!
    })
    coursework = pd.DataFrame({
        "student_id": student_id,
        "assignment_avg": assignment_avg.round(1),
        "leetcode_solved": leetcode_solved,
        "github_commits_month": github_commits,
    })
    exams = pd.DataFrame({
        "student_id": student_id,
        "midterm": midterm.round(1),
        "final_score": final_score.round(1),
        "exam_date": exam_date,
    })

    students = _make_messy(students, rng)
    return students, coursework, exams


def _make_messy(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1) missing values
    for col, frac in [("sleep_hours", 0.04), ("study_hours_week", 0.03), ("prior_gpa", 0.05), ("coffee_cups_day", 0.03)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2) inconsistent category spelling
    idx = rng.choice(n, size=int(n * 0.06), replace=False)
    df.loc[idx, "track"] = [rng.choice([t.lower(), f"  {t} ", t.upper()]) for t in df.loc[idx, "track"]]

    # 3) impossible outliers
    df.loc[rng.choice(n, 8, replace=False), "study_hours_week"] = rng.uniform(120, 168, 8).round(1)
    df.loc[rng.choice(n, 5, replace=False), "sleep_hours"] = 0.0

    # 4) duplicate rows
    dupes = df.sample(25, random_state=7)
    return pd.concat([df, dupes], ignore_index=True)


def save(students: pd.DataFrame, coursework: pd.DataFrame, exams: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    students.to_csv(DATA_DIR / "students.csv", index=False)
    coursework.to_json(DATA_DIR / "coursework.json", orient="records", indent=1)
    exams.to_excel(DATA_DIR / "exams.xlsx", index=False)


if __name__ == "__main__":
    save(*generate())
    print(f"Wrote students.csv, coursework.json, exams.xlsx to {DATA_DIR}")
