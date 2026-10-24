"""
Will YOU pass? Enter your habits and the trained neural network predicts it.

    python predict.py                                  # interactive
    python predict.py --study 20 --sleep 7.5 --track "AI/ML"
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import VALID_TRACKS, _wellbeing

MODEL_PATH = Path(__file__).resolve().parent / "results" / "pass_predictor.joblib"

QUESTIONS = [  # (arg name, prompt, default, type)
    ("track", f"Track {VALID_TRACKS}", "AI/ML", str),
    ("year", "Year (1-4)", 2, int),
    ("study", "Study hours per week", 14, float),
    ("sleep", "Sleep hours per night", 7, float),
    ("screen", "Social-media/screen hours per day", 5, float),
    ("job", "Part-time job? (1 = yes, 0 = no)", 0, int),
    ("gpa", "Prior GPA (0-4)", 3.0, float),
    ("coffee", "Cups of coffee per day", 2, int),
    ("attendance", "Lecture attendance %", 80, float),
    ("assignments", "Assignment average %", 70, float),
    ("leetcode", "LeetCode problems solved this semester", 50, int),
    ("commits", "GitHub commits per month", 35, int),
]


def build_row(a) -> dict:
    row = {
        "year": a.year, "study_hours_week": a.study, "sleep_hours": a.sleep, "screen_time_hours": a.screen,
        "part_time_job": a.job, "prior_gpa": a.gpa, "coffee_cups_day": a.coffee, "attendance": a.attendance,
        "assignment_avg": a.assignments, "leetcode_solved": a.leetcode, "github_commits_month": a.commits,
        "sleep_deficit": max(7.5 - a.sleep, 0), "problems_per_study_hour": a.leetcode / (a.study + 1),
        "days_enrolled": 800,
    }
    row["wellbeing"] = _wellbeing(pd.Series(row))
    for t in VALID_TRACKS:
        row[f"track_{t}"] = int(a.track.strip().lower() == t.lower())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, prompt, default, typ in QUESTIONS:
        parser.add_argument(f"--{name}", type=typ, default=None, help=f"{prompt} (default {default})".replace("%", "%%"))
    args = parser.parse_args()

    interactive = all(getattr(args, q[0]) is None for q in QUESTIONS)
    for name, prompt, default, typ in QUESTIONS:
        if getattr(args, name) is None:
            raw = input(f"{prompt} [{default}]: ").strip() if interactive else ""
            setattr(args, name, typ(raw) if raw else default)

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model yet — run `python main.py` first.")
    bundle = joblib.load(MODEL_PATH)
    row = build_row(args)
    x = np.array([[row[f] for f in bundle["features"]]], dtype=float)
    p = bundle["model"].predict_proba(bundle["scaler"].transform(x))[0, 1]

    bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
    print(f"\nProbability of passing: {bar} {p:.1%}")
    print("Verdict:", "🎉 You're on track!" if p >= 0.7 else "😬 Borderline — small changes matter." if p >= 0.4
          else "🚨 At risk. Time to change some habits.")
    if args.sleep < 6.5:
        print("  💤 Tip: more sleep is one of the strongest levers in this dataset.")
    if args.study < 12:
        print("  📚 Tip: students who study 15+ hrs/week pass far more often.")
    if args.coffee > 4:
        print("  ☕ Tip: past ~3 cups a day, scores start dropping.")


if __name__ == "__main__":
    main()
