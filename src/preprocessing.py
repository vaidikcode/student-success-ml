"""
Pandas pipeline: load -> inspect -> clean -> merge -> outliers -> features -> encode.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALID_TRACKS = ["AI/ML", "Systems", "Web Dev", "Security", "Theory"]


# ----------------------------------------------------------------------------
# Loading (CSV / JSON / Excel)
# ----------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    students = pd.read_csv(DATA_DIR / "students.csv")
    coursework = pd.read_json(DATA_DIR / "coursework.json")
    exams = pd.read_excel(DATA_DIR / "exams.xlsx")
    return students, coursework, exams


def inspect(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== Inspecting {name}: shape={df.shape} ===")
    print(df.head(3).to_string())
    print("\ndtypes:\n", df.dtypes.to_string())
    print("\nmissing values per column:\n", df.isnull().sum()[lambda s: s > 0].to_string() or "none")
    print("\nduplicate rows:", df.duplicated().sum())


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    print(f"\n[clean] dropped {before - len(df)} duplicate rows")

    # fix inconsistent category spelling: "  ai/ml " -> "AI/ML"
    lookup = {t.lower(): t for t in VALID_TRACKS}
    df["track"] = df["track"].str.strip().str.lower().map(lookup)
    print("[clean] track value_counts after normalizing:\n", df["track"].value_counts().to_string())

    # fix data types
    df["attendance"] = df["attendance"].str.rstrip("%").astype(float)        # "87.5%" -> 87.5
    df["enrollment_date"] = pd.to_datetime(df["enrollment_date"])
    df["part_time_job"] = df["part_time_job"].map({"Yes": 1, "No": 0})

    # fill missing values: median *within each track* (more precise than a global median)
    for col in ["sleep_hours", "study_hours_week", "prior_gpa"]:
        n_missing = df[col].isnull().sum()
        df[col] = df[col].fillna(df.groupby("track")[col].transform("median"))
        print(f"[clean] filled {n_missing} missing values in {col} with per-track median")
    df["coffee_cups_day"] = df["coffee_cups_day"].fillna(df["coffee_cups_day"].mode()[0]).astype(int)
    return df


def merge_all(students, coursework, exams) -> pd.DataFrame:
    df = students.merge(coursework, on="student_id", how="inner").merge(exams, on="student_id", how="inner")
    print(f"\n[merge] students {students.shape} + coursework {coursework.shape} + exams {exams.shape} -> {df.shape}")
    return df


# ----------------------------------------------------------------------------
# Outliers
# ----------------------------------------------------------------------------
def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def zscore_outlier_mask(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    return ((s - s.mean()) / s.std()).abs() > threshold


def remove_outliers(df: pd.DataFrame, cols=("study_hours_week", "sleep_hours")) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for col in cols:
        iqr_m, z_m = iqr_outlier_mask(df[col]), zscore_outlier_mask(df[col])
        print(f"[outliers] {col:<18} IQR flags {iqr_m.sum():>3} | z-score flags {z_m.sum():>3}")
        mask |= z_m      # z-score: removes the impossible values without throwing away legit extremes
    print(f"[outliers] removing {mask.sum()} rows")
    return df.loc[~mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def _wellbeing(row) -> float:
    """Row-wise apply(): a 0-10 'wellbeing' score from sleep, screen time and caffeine."""
    score = 10 - abs(row["sleep_hours"] - 7.5) * 1.5 - max(row["screen_time_hours"] - 4, 0) * 0.5
    score -= max(row["coffee_cups_day"] - 3, 0)
    return float(np.clip(score, 0, 10))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["enroll_month"] = df["enrollment_date"].dt.month
    df["enroll_year"] = df["enrollment_date"].dt.year
    df["enroll_dayofweek"] = df["enrollment_date"].dt.dayofweek
    df["days_enrolled"] = (df["exam_date"] - df["enrollment_date"]).dt.days
    df["sleep_deficit"] = (7.5 - df["sleep_hours"]).clip(lower=0)
    df["problems_per_study_hour"] = df["leetcode_solved"] / (df["study_hours_week"] + 1)
    df["wellbeing"] = df.apply(_wellbeing, axis=1)
    df["year_label"] = df["year"].map({1: "Freshman", 2: "Sophomore", 3: "Junior", 4: "Senior"})

    # targets
    df["passed"] = (df["final_score"] >= 60).astype(int)
    df["grade"] = pd.cut(df["final_score"], bins=[-1, 59.9, 69.9, 79.9, 89.9, 100], labels=["F", "D", "C", "B", "A"])
    return df.sort_values("student_id").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Grouping, aggregation & querying
# ----------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> None:
    print("\n=== groupby + agg: performance by track ===")
    print(df.groupby("track").agg(
        students=("student_id", "count"),
        avg_score=("final_score", "mean"),
        pass_rate=("passed", "mean"),
        avg_study=("study_hours_week", "mean"),
    ).round(2).sort_values("pass_rate", ascending=False).to_string())

    print("\n=== pivot_table: mean score by track x year ===")
    print(df.pivot_table(values="final_score", index="track", columns="year_label", aggfunc="mean")
          [["Freshman", "Sophomore", "Junior", "Senior"]].round(1).to_string())

    print("\n=== crosstab: grade distribution by part-time job (row %) ===")
    print((pd.crosstab(df["part_time_job"].map({0: "No job", 1: "Has job"}), df["grade"], normalize="index") * 100).round(1).to_string())

    print("\n=== query(): sleep-deprived students (< 5.5h) who still scored a B or better ===")
    heroes = df.query("sleep_hours < 5.5 and grade in ['A', 'B']")
    print(heroes[["student_id", "sleep_hours", "study_hours_week", "final_score"]].to_string(index=False) if len(heroes) else "none — sleep matters!")

    print("\n=== value_counts: grades ===")
    print(df["grade"].value_counts().sort_index().to_string())


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "year", "study_hours_week", "sleep_hours", "screen_time_hours", "part_time_job", "prior_gpa",
    "coffee_cups_day", "attendance", "assignment_avg", "leetcode_solved", "github_commits_month",
    "sleep_deficit", "problems_per_study_hour", "wellbeing", "days_enrolled",
]


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """One-hot encode `track` (nominal) and label-encode `grade` (ordinal)."""
    df = pd.get_dummies(df, columns=["track"], prefix="track", dtype=int)
    le = LabelEncoder()
    df["grade_encoded"] = le.fit_transform(df["grade"].astype(str))
    return df, le


def feature_columns(df: pd.DataFrame) -> list[str]:
    return NUMERIC_FEATURES + [c for c in df.columns if c.startswith("track_")]


def build_dataset(verbose: bool = True) -> pd.DataFrame:
    students, coursework, exams = load_raw()
    if verbose:
        inspect(students, "students.csv")
        print("\ndescribe():\n", students.describe().round(2).to_string())
    df = merge_all(clean_students(students), coursework, exams)
    df = remove_outliers(df)
    df = engineer_features(df)
    if verbose:
        summarize(df)
    return df
