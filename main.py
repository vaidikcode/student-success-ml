"""
Run the whole project end to end:

    python main.py

1. Generate messy data (CSV + JSON + Excel)      -> src/data_generation.py
2. NumPy foundations + from-scratch neural net   -> src/numpy_foundations.py
3. Pandas cleaning & feature engineering         -> src/preprocessing.py
4. EDA plots with Matplotlib & Seaborn           -> src/eda.py
5. scikit-learn MLPs: classify & regress         -> src/models.py
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src import data_generation, eda, models
from src import numpy_foundations as npf
from src.preprocessing import build_dataset, encode, feature_columns


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78)


def main() -> None:
    banner("STEP 1 — Generating the (messy) dataset")
    data_generation.save(*data_generation.generate())
    print("Wrote data/students.csv, data/coursework.json, data/exams.xlsx")

    banner("STEP 2 — Pandas: load, inspect, clean, merge, engineer features")
    df = build_dataset(verbose=True)
    df_enc, grade_encoder = encode(df)
    features = feature_columns(df_enc)
    print(f"\n[encode] one-hot tracks + label-encoded grades {dict(enumerate(grade_encoder.classes_))}")
    print(f"[encode] {len(features)} model features: {features}")

    X = df_enc[features].to_numpy(dtype=float)
    y_cls = df_enc["passed"].to_numpy()
    y_reg = df_enc["final_score"].to_numpy()

    banner("STEP 3 — NumPy foundations")
    npf.array_basics_demo()
    npf.vectorization_demo()
    npf.describe(X[:, :8], features[:8])
    npf.correlation_with_target(X[:, :15], y_reg, features[:15])

    X_tr, X_te, y_tr, y_te = npf.train_test_split_np(X, y_reg)
    X_tr_s, X_te_s = npf.standardize(X_tr, X_te)
    X_tr_mm, _ = npf.min_max_normalize(X_tr, X_te)
    print(f"\nstandardized train: mean ≈ {X_tr_s.mean():.3f}, std ≈ {X_tr_s.std():.3f}"
          f" | min-max train range: [{X_tr_mm.min():.1f}, {X_tr_mm.max():.1f}]")

    w = npf.normal_equation(X_tr_s, y_tr)
    pred = npf.predict_linear(w, X_te_s)
    linear = {"rmse": float(np.sqrt(mean_squared_error(y_te, pred))), "r2": r2_score(y_te, pred)}
    print(f"Normal-equation linear regression: RMSE = {linear['rmse']:.2f}, R² = {linear['r2']:.3f}")

    print("\nTraining a neural network written from scratch in NumPy...")
    Xc_tr, Xc_te, yc_tr, yc_te = npf.train_test_split_np(X, y_cls)
    Xc_tr_s, Xc_te_s = npf.standardize(Xc_tr, Xc_te)
    Xa, Xv, ya, yv = npf.train_test_split_np(Xc_tr_s, yc_tr, test_size=0.2, seed=1)
    scratch = npf.NumpyMLP(layer_sizes=(16, 8), lr=0.02, alpha=1.0, epochs=150).fit(Xa, ya, Xv, yv)
    scratch_acc = (scratch.predict(Xc_te_s) == yc_te).mean()
    print(f"From-scratch NumPy MLP test accuracy: {scratch_acc:.3f}")
    models.plot_curves({"NumPy MLP (from scratch)": scratch.history}, "07_numpy_mlp_training.png",
                       "Our hand-written NumPy neural network learning")

    banner("STEP 4 — Exploratory Data Analysis (Matplotlib + Seaborn)")
    eda.run_eda(df_enc.assign(track=df["track"]), features)

    banner("STEP 5 — scikit-learn MLPClassifier: will the student pass?")
    models.scaler_comparison(X, y_cls)
    X_train, X_test, y_train, y_test, scaler = models.split_and_scale(X, y_cls)
    models.overfitting_experiment(X_train, y_train)
    models.hyperparameter_search(X_train, y_train)

    clf = models.MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=1.0,
                               learning_rate_init=0.001, max_iter=1000, early_stopping=True,
                               validation_fraction=0.15, n_iter_no_change=25, random_state=models.SEED)
    clf.fit(X_train, y_train)
    print(f"\nFinal model stopped early after {clf.n_iter_} epochs "
          f"(best validation accuracy {max(clf.validation_scores_):.3f})")
    # the NumPy MLP used a different split, so retrain it on this one for a fair ROC comparison
    scratch_fair = npf.NumpyMLP(layer_sizes=(16, 8), lr=0.02, alpha=1.0, epochs=150).fit(X_train, y_train)
    cls_metrics = models.evaluate_classifier(clf, X_test, y_test, scratch_fair, X_test)
    models.plot_decision_boundary(df)

    banner("STEP 6 — scikit-learn MLPRegressor: predict the exact score")
    reg_metrics = models.run_regression(X, y_reg, linear_baseline=linear)

    models.save_model(clf, scaler, features, {
        "classifier": cls_metrics,
        "regressor": reg_metrics,
        "numpy_mlp_from_scratch_accuracy": scratch_acc,
        "numpy_linear_regression": linear,
    })
    banner("DONE ✅  — check the figures/ folder, then try:  python predict.py")


if __name__ == "__main__":
    main()
