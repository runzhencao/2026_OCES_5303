"""Reproducible bacteria-abundance regression for OCES 5303 Assignment 1.

The scientific task is contemporaneous estimation: environmental measurements
from a sampling occasion are used to estimate the bacteria-abundance value from
that same occasion.  The final chronological block is held out to test whether
the fitted relationship generalizes to a later sampling period.  This is not a
forecast using unknown future covariates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_URL = (
    "https://raw.githubusercontent.com/julianmak/"
    "OCES4303_ML_ocean/refs/heads/main/PIDweekly_env_data.txt"
)
TARGET = "Bacteria_abundance"
TARGET_UNITS = "cells mL^-1"
TIME_COLUMN = "YearDay"
ID_COLUMN = "SampleID"
MAIN_FEATURES = [
    "Temp",
    "Salinity",
    "Oxygen_Saturation",
    "pH",
    "DIC",
    "Chlorophyll",
    "NH4",
    "NO2.NO3",
    "PO4",
    "SiO4",
]
EXCLUDED_COLUMNS = [
    ID_COLUMN,
    TIME_COLUMN,
    "Projected_Daily_Insolation",
    "MLLW",
]
RANDOM_SEED = 5303
TEST_FRACTION = 0.20
CV_SPLITS = 5
CV_GAP = 1


def _json_ready(value: Any) -> Any:
    """Convert numpy/pandas scalar containers into strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if pd.isna(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def _validate_data(df: pd.DataFrame) -> None:
    required = {TARGET, TIME_COLUMN, ID_COLUMN, *MAIN_FEATURES, *EXCLUDED_COLUMNS}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if len(df) < 50:
        raise ValueError("Too few rows for the prespecified chronological analysis.")
    if df[[TARGET, TIME_COLUMN]].isna().any().any():
        raise ValueError("Target and time columns must be complete.")
    numeric = df[[TARGET, TIME_COLUMN, *MAIN_FEATURES]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Target, time, and main predictors must be finite.")
    if df[TIME_COLUMN].duplicated().any():
        raise ValueError("YearDay values must be unique for an unambiguous time order.")


def _model_definitions() -> tuple[dict[str, Pipeline], dict[str, dict[str, list[Any]]]]:
    imputer = SimpleImputer(strategy="median")
    models = {
        "mean_baseline": Pipeline(
            [("imputer", clone(imputer)), ("model", DummyRegressor(strategy="mean"))]
        ),
        "temperature_linear": Pipeline(
            [
                ("imputer", clone(imputer)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            [
                ("imputer", clone(imputer)),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", clone(imputer)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=RANDOM_SEED,
                        # A single worker keeps floating-point reduction order
                        # identical across repeated runs; GridSearchCV still
                        # evaluates independent candidates in parallel.
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    grids = {
        "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        "random_forest": {
            "model__max_depth": [3, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": [0.7, 1.0],
        },
    }
    return models, grids


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_overview(df: pd.DataFrame, split_index: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    axes[0].plot(train[TIME_COLUMN], train[TARGET] / 1e6, color="#28628b", lw=1.4)
    axes[0].plot(test[TIME_COLUMN], test[TARGET] / 1e6, color="#d1495b", lw=1.4)
    axes[0].scatter(train[TIME_COLUMN], train[TARGET] / 1e6, color="#28628b", s=14, label="Training")
    axes[0].scatter(test[TIME_COLUMN], test[TARGET] / 1e6, color="#d1495b", s=18, label="Held-out test")
    axes[0].axvline(
        (float(train[TIME_COLUMN].iloc[-1]) + float(test[TIME_COLUMN].iloc[0])) / 2,
        color="0.25",
        ls="--",
        lw=1,
    )
    axes[0].set(xlabel="YearDay", ylabel=r"Bacteria abundance ($10^6$ cells mL$^{-1}$)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        train["Temp"], train[TARGET] / 1e6, color="#28628b", alpha=0.76, s=25, label="Training"
    )
    axes[1].scatter(
        test["Temp"], test[TARGET] / 1e6, color="#d1495b", alpha=0.85, s=27, label="Held-out test"
    )
    axes[1].set(
        xlabel="Temperature (°C)",
        ylabel=r"Bacteria abundance ($10^6$ cells mL$^{-1}$)",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Chronological evaluation design and temperature relationship", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_training_diagnostics(
    train: pd.DataFrame, fitted: dict[str, Pipeline], path: Path
) -> pd.DataFrame:
    correlation = train[[*MAIN_FEATURES, TARGET]].corr(method="spearman")
    ridge_coef = fitted["ridge"].named_steps["model"].coef_
    forest_importance = fitted["random_forest"].named_steps["model"].feature_importances_
    diagnostic = pd.DataFrame(
        {
            "feature": MAIN_FEATURES,
            "spearman_with_target_train": correlation.loc[MAIN_FEATURES, TARGET].to_numpy(),
            "ridge_standardized_coefficient": ridge_coef,
            "random_forest_impurity_importance": forest_importance,
        }
    )
    diagnostic = diagnostic.sort_values("spearman_with_target_train")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    colors = np.where(diagnostic["spearman_with_target_train"] >= 0, "#28628b", "#d1495b")
    axes[0].barh(
        diagnostic["feature"], diagnostic["spearman_with_target_train"], color=colors, alpha=0.88
    )
    axes[0].axvline(0, color="0.2", lw=0.8)
    axes[0].set(xlabel="Spearman correlation with target (training only)", ylabel="")
    axes[0].grid(axis="x", alpha=0.25)

    importance_order = diagnostic.sort_values("random_forest_impurity_importance")
    axes[1].barh(
        importance_order["feature"],
        importance_order["random_forest_impurity_importance"],
        color="#4f772d",
        alpha=0.88,
    )
    axes[1].set(xlabel="Random-forest impurity importance (training fit)", ylabel="")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Training-only feature diagnostics (associations, not causal effects)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return diagnostic.sort_values("feature").reset_index(drop=True)


def _plot_performance(metrics: pd.DataFrame, path: Path) -> None:
    ordered = metrics.sort_values("rmse")
    labels = [name.replace("_", " ").title() for name in ordered["model"]]
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.bar(x - width / 2, ordered["mae"] / 1e6, width, label="MAE", color="#28628b")
    ax.bar(x + width / 2, ordered["rmse"] / 1e6, width, label="RMSE", color="#d1495b")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(r"Error ($10^6$ cells mL$^{-1}$)")
    ax.set_title("Performance on the untouched later-period test block")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_predictions(predictions: pd.DataFrame, path: Path) -> None:
    palette = {
        "mean_baseline": "#7f7f7f",
        "temperature_linear": "#f28e2b",
        "ridge": "#28628b",
        "random_forest": "#4f772d",
    }
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.plot(
        predictions[TIME_COLUMN],
        predictions["observed"] / 1e6,
        color="black",
        marker="o",
        ms=4.5,
        lw=1.5,
        label="Observed",
        zorder=5,
    )
    for model, color in palette.items():
        ax.plot(
            predictions[TIME_COLUMN],
            predictions[f"pred_{model}"] / 1e6,
            color=color,
            lw=1.45,
            label=model.replace("_", " ").title(),
        )
    ax.set(xlabel="YearDay", ylabel=r"Bacteria abundance ($10^6$ cells mL$^{-1}$)")
    ax.set_title("Observed and predicted values in the held-out later period")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the prespecified analysis and save all report-ready artifacts."""
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path, sep=r"\s+")
    _validate_data(raw)
    original_monotonic = bool(raw[TIME_COLUMN].is_monotonic_increasing)
    df = raw.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)

    n_total = len(df)
    n_test = int(np.ceil(TEST_FRACTION * n_total))
    split_index = n_total - n_test
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    X_train = train[MAIN_FEATURES]
    X_test = test[MAIN_FEATURES]
    y_train = train[TARGET]
    y_test = test[TARGET]

    split_membership = df[[ID_COLUMN, TIME_COLUMN]].copy()
    split_membership["row_after_time_sort"] = np.arange(n_total)
    split_membership["split"] = np.where(
        split_membership["row_after_time_sort"] < split_index, "train", "test"
    )
    split_membership.to_csv(output_dir / "split_membership.csv", index=False)

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    cv_split_rows: list[dict[str, Any]] = []
    for fold, (fit_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        cv_split_rows.append(
            {
                "fold": fold,
                "train_n": len(fit_idx),
                "validation_n": len(val_idx),
                "train_start_YearDay": int(train.iloc[fit_idx[0]][TIME_COLUMN]),
                "train_end_YearDay": int(train.iloc[fit_idx[-1]][TIME_COLUMN]),
                "gap_YearDay": int(train.iloc[fit_idx[-1] + 1][TIME_COLUMN]),
                "validation_start_YearDay": int(train.iloc[val_idx[0]][TIME_COLUMN]),
                "validation_end_YearDay": int(train.iloc[val_idx[-1]][TIME_COLUMN]),
            }
        )
    pd.DataFrame(cv_split_rows).to_csv(output_dir / "cv_splits.csv", index=False)

    models, grids = _model_definitions()
    selected: dict[str, Pipeline] = {}
    selection_records: dict[str, Any] = {}
    cv_results_frames: list[pd.DataFrame] = []

    baseline_features = {
        "mean_baseline": MAIN_FEATURES,
        "temperature_linear": ["Temp"],
    }
    for name in ["mean_baseline", "temperature_linear"]:
        features = baseline_features[name]
        scores = cross_validate(
            models[name],
            train[features],
            y_train,
            cv=tscv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            return_train_score=False,
            n_jobs=-1,
        )
        cv_record = {
            "model": name,
            "candidate": "fixed_baseline",
            "mean_cv_rmse": float(-scores["test_rmse"].mean()),
            "std_cv_rmse": float(scores["test_rmse"].std(ddof=0)),
            "mean_cv_mae": float(-scores["test_mae"].mean()),
            "std_cv_mae": float(scores["test_mae"].std(ddof=0)),
            "rank_cv_rmse": 1,
            "params_json": "{}",
        }
        for fold in range(CV_SPLITS):
            cv_record[f"fold_{fold + 1}_validation_rmse"] = float(-scores["test_rmse"][fold])
            cv_record[f"fold_{fold + 1}_validation_mae"] = float(-scores["test_mae"][fold])
        cv_results_frames.append(pd.DataFrame([cv_record]))
        fitted = clone(models[name]).fit(train[features], y_train)
        selected[name] = fitted
        selection_records[name] = {
            "selection": "fixed_baseline; no hyperparameters selected",
            "features": [] if name == "mean_baseline" else features,
            "cv_mean_rmse": float(-scores["test_rmse"].mean()),
            "cv_std_rmse": float(scores["test_rmse"].std(ddof=1)),
        }

    for name in ["ridge", "random_forest"]:
        search = GridSearchCV(
            estimator=models[name],
            param_grid=grids[name],
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
            },
            cv=tscv,
            refit="rmse",
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        selected[name] = search.best_estimator_
        selection_records[name] = {
            "selection": "minimum mean validation RMSE in training-only TimeSeriesSplit",
            "features": MAIN_FEATURES,
            "best_params": search.best_params_,
            "best_cv_rmse": float(-search.best_score_),
        }
        results = pd.DataFrame(search.cv_results_)
        compact = pd.DataFrame({
            "model": name,
            "candidate": np.arange(len(results)) + 1,
            "mean_cv_rmse": -results["mean_test_rmse"],
            "std_cv_rmse": results["std_test_rmse"],
            "mean_cv_mae": -results["mean_test_mae"],
            "std_cv_mae": results["std_test_mae"],
            "rank_cv_rmse": results["rank_test_rmse"],
            "params_json": [json.dumps(p, sort_keys=True) for p in results["params"]],
        })
        for fold in range(CV_SPLITS):
            compact[f"fold_{fold + 1}_validation_rmse"] = -results[f"split{fold}_test_rmse"]
            compact[f"fold_{fold + 1}_validation_mae"] = -results[f"split{fold}_test_mae"]
        cv_results_frames.append(compact)

    cv_results = pd.concat(cv_results_frames, ignore_index=True)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)

    predictions = test[[ID_COLUMN, TIME_COLUMN, TARGET]].rename(columns={TARGET: "observed"}).copy()
    metric_rows: list[dict[str, Any]] = []
    for name in ["mean_baseline", "temperature_linear", "ridge", "random_forest"]:
        features = baseline_features.get(name, MAIN_FEATURES)
        prediction = selected[name].predict(test[features])
        predictions[f"pred_{name}"] = prediction
        row = {
            "model": name,
            "predictor_set": "Temp only" if name == "temperature_linear" else (
                "training target mean" if name == "mean_baseline" else "10 contemporaneous predictors"
            ),
            "n_train": len(train),
            "n_test": len(test),
            **_regression_metrics(y_test, prediction),
        }
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    _plot_overview(df, split_index, figures_dir / "01_data_overview_and_split.png")
    diagnostics = _plot_training_diagnostics(
        train, selected, figures_dir / "02_training_feature_diagnostics.png"
    )
    diagnostics.to_csv(output_dir / "training_feature_diagnostics.csv", index=False)
    _plot_performance(metrics, figures_dir / "03_heldout_model_performance.png")
    _plot_predictions(predictions, figures_dir / "04_heldout_predictions_over_time.png")

    data_audit = {
        "source_url": SOURCE_URL,
        "local_file": data_path.name,
        "rows": n_total,
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "main_predictors": MAIN_FEATURES,
        "excluded_from_main_predictors": EXCLUDED_COLUMNS,
        "exclusion_rationale": {
            ID_COLUMN: "identifier, not an environmental predictor",
            TIME_COLUMN: "time/order variable reserved for splitting, not a main predictor",
            "Projected_Daily_Insolation": "course starter explicitly drops it; strong deterministic seasonal proxy",
            "MLLW": "course starter explicitly drops it and its physical interpretation is uncertain in the supplied notes",
        },
        "target": TARGET,
        "target_units": TARGET_UNITS,
        "target_units_note": (
            "Verified against the source dataset documentation; figures divide values by 1e6."
        ),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_sample_ids": int(df[ID_COLUMN].duplicated().sum()),
        "duplicate_YearDay": int(df[TIME_COLUMN].duplicated().sum()),
        "original_YearDay_monotonic_increasing": original_monotonic,
        "YearDay_min": int(df[TIME_COLUMN].min()),
        "YearDay_max": int(df[TIME_COLUMN].max()),
        "YearDay_median_interval": float(df[TIME_COLUMN].diff().median()),
        "target_summary": df[TARGET].describe().to_dict(),
        "split": {
            "method": "sort by YearDay; hold out final ceil(20%)",
            "n_train": len(train),
            "n_test": len(test),
            "train_YearDay_min": int(train[TIME_COLUMN].min()),
            "train_YearDay_max": int(train[TIME_COLUMN].max()),
            "test_YearDay_min": int(test[TIME_COLUMN].min()),
            "test_YearDay_max": int(test[TIME_COLUMN].max()),
        },
    }
    _save_json(output_dir / "data_audit.json", data_audit)
    _save_json(output_dir / "selected_models.json", selection_records)

    run_metadata = {
        "analysis": "contemporaneous bacteria-abundance regression with later-period evaluation",
        "interpretation": (
            "Environmental covariates and target are measured at the same sampling occasion. "
            "The holdout tests later-period generalization; it is not a forecast with unknown future covariates."
        ),
        "target_scale": "raw supplied target values; no target transform",
        "target_units": TARGET_UNITS,
        "test_used_for_selection": False,
        "random_seed": RANDOM_SEED,
        "test_fraction": TEST_FRACTION,
        "time_series_cv": {"n_splits": CV_SPLITS, "gap_rows": CV_GAP},
        "grids": grids,
        "data_sha256": _sha256(data_path),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_environment": Path(sys.prefix).name,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "command_template": (
            "python src/run_analysis.py --data data/PIDweekly_env_data.txt --output results"
        ),
    }
    _save_json(output_dir / "run_metadata.json", run_metadata)

    return {
        "data_audit": data_audit,
        "metrics": metrics.to_dict(orient="records"),
        "selected_models": selection_records,
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to PIDweekly_env_data.txt")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args(argv)
    summary = run_analysis(args.data, args.output)
    print(json.dumps(_json_ready(summary), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
