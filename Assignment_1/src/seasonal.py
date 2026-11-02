"""Training-only post hoc comparison of fixed annual and environmental Ridge baselines.

The analysis is deliberately bounded to the first 120 time-sorted observations
(YearDay 19--872).  The later 31 observations have already been seen in the
course analysis, so this module does not evaluate or inspect their outcomes.
"""

from __future__ import annotations

import argparse
import json
import platform
import runpy
import sys
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "Bacteria_abundance"
TARGET_UNITS = "cells mL^-1"
TIME_COLUMN = "YearDay"
ID_COLUMN = "SampleID"
TRAIN_ROWS = 120
EXPECTED_DAY_RANGE = (19, 872)
ANNUAL_PERIOD_DAYS = 365.25
PHASE_ORIGIN_DAY = 0.0
OUTER_SPLITS = 5
INNER_SPLITS = 3
GAP_ROWS = 1
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
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
ANNUAL_FEATURES = ["annual_sin", "annual_cos"]
VARIANTS = {
    "annual_only": ANNUAL_FEATURES,
    "temperature_only": ["Temp"],
    "all10_env": MAIN_FEATURES,
    "annual_plus_all10": [*ANNUAL_FEATURES, *MAIN_FEATURES],
}
DISPLAY_LABELS = {
    "annual_only": "Annual sin/cos only",
    "temperature_only": "Temperature only",
    "all10_env": "All 10 environmental",
    "annual_plus_all10": "Annual sin/cos + all 10",
}
SHARED_STYLE = Path(
    "/Users/caorunzhen/PythonLateX/2026_OCES_5303/"
    "Assignment_1_Bacteria/figures/scripts/paper_plot_style.py"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def ridge_pipeline() -> Pipeline:
    """Return the common standardized Ridge pipeline used by all variants."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def _prepare_training_frame(data_path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(data_path, sep=r"\s+")
    required = {ID_COLUMN, TIME_COLUMN, TARGET, *MAIN_FEATURES}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(raw) <= TRAIN_ROWS:
        raise ValueError("Expected observations after the 120-row development period")

    ordered = raw.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    train = ordered.iloc[:TRAIN_ROWS].copy()
    if (int(train[TIME_COLUMN].min()), int(train[TIME_COLUMN].max())) != EXPECTED_DAY_RANGE:
        raise ValueError("Development-period YearDay boundary does not match 19--872")
    if train[ID_COLUMN].duplicated().any() or train[TIME_COLUMN].duplicated().any():
        raise ValueError("Development SampleID and YearDay values must be unique")
    values = train[[TIME_COLUMN, TARGET, *MAIN_FEATURES]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Development target, time, and environmental values must be finite")

    angle = 2.0 * np.pi * (train[TIME_COLUMN] - PHASE_ORIGIN_DAY) / ANNUAL_PERIOD_DAYS
    train[ANNUAL_FEATURES[0]] = np.sin(angle)
    train[ANNUAL_FEATURES[1]] = np.cos(angle)
    train["row_after_time_sort"] = np.arange(TRAIN_ROWS)
    return train, len(ordered) - TRAIN_ROWS


def _split_memberships(
    train: pd.DataFrame, outer_splits: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for outer_fold, (fit_idx, val_idx) in enumerate(outer_splits, start=1):
        gap_idx = np.arange(fit_idx[-1] + 1, val_idx[0])
        if len(gap_idx) != GAP_ROWS:
            raise AssertionError("Outer split does not contain the required one-row gap")
        if set(fit_idx) & set(gap_idx) or set(fit_idx) & set(val_idx) or set(gap_idx) & set(val_idx):
            raise AssertionError("Outer train, gap, and validation membership overlaps")
        roles = np.full(TRAIN_ROWS, "not_available", dtype=object)
        roles[fit_idx] = "train"
        roles[gap_idx] = "gap"
        roles[val_idx] = "validation"
        for row, role in enumerate(roles):
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "row_after_time_sort": row,
                    ID_COLUMN: train.loc[row, ID_COLUMN],
                    TIME_COLUMN: int(train.loc[row, TIME_COLUMN]),
                    "role": role,
                }
            )

        inner_splits = list(TimeSeriesSplit(n_splits=INNER_SPLITS, gap=GAP_ROWS).split(fit_idx))
        inner_metadata: list[dict[str, Any]] = []
        for inner_fold, (inner_fit_pos, inner_val_pos) in enumerate(inner_splits, start=1):
            inner_gap_pos = np.arange(inner_fit_pos[-1] + 1, inner_val_pos[0])
            if len(inner_gap_pos) != GAP_ROWS:
                raise AssertionError("Inner split does not contain the required one-row gap")
            role_by_position = np.full(len(fit_idx), "not_available", dtype=object)
            role_by_position[inner_fit_pos] = "train"
            role_by_position[inner_gap_pos] = "gap"
            role_by_position[inner_val_pos] = "validation"
            for outer_train_position, role in enumerate(role_by_position):
                row = int(fit_idx[outer_train_position])
                inner_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "outer_train_position": outer_train_position,
                        "row_after_time_sort": row,
                        ID_COLUMN: train.loc[row, ID_COLUMN],
                        TIME_COLUMN: int(train.loc[row, TIME_COLUMN]),
                        "role": role,
                    }
                )
            inner_metadata.append(
                {
                    "inner_fold": inner_fold,
                    "train_sample_ids": train.iloc[fit_idx[inner_fit_pos]][ID_COLUMN].tolist(),
                    "gap_sample_ids": train.iloc[fit_idx[inner_gap_pos]][ID_COLUMN].tolist(),
                    "validation_sample_ids": train.iloc[fit_idx[inner_val_pos]][ID_COLUMN].tolist(),
                }
            )

        fold_metadata.append(
            {
                "outer_fold": outer_fold,
                "train_sample_ids": train.iloc[fit_idx][ID_COLUMN].tolist(),
                "gap_sample_ids": train.iloc[gap_idx][ID_COLUMN].tolist(),
                "validation_sample_ids": train.iloc[val_idx][ID_COLUMN].tolist(),
                "inner_folds": inner_metadata,
            }
        )
    return pd.DataFrame(outer_rows), pd.DataFrame(inner_rows), fold_metadata


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        group = fold_metrics[fold_metrics["variant"] == variant]
        row: dict[str, Any] = {"variant": variant, "folds": len(group)}
        for metric in ["rmse", "mae", "r2"]:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    difference_rows: list[dict[str, Any]] = []
    for variant_a, variant_b in combinations(VARIANTS, 2):
        left = fold_metrics[fold_metrics["variant"] == variant_a].set_index("outer_fold")
        right = fold_metrics[fold_metrics["variant"] == variant_b].set_index("outer_fold")
        for fold in range(1, OUTER_SPLITS + 1):
            row: dict[str, Any] = {
                "outer_fold": fold,
                "comparison": f"{variant_a}_minus_{variant_b}",
                "variant_a": variant_a,
                "variant_b": variant_b,
            }
            for metric in ["rmse", "mae", "r2"]:
                row[f"{metric}_difference_a_minus_b"] = float(
                    left.loc[fold, metric] - right.loc[fold, metric]
                )
            difference_rows.append(row)
    differences = pd.DataFrame(difference_rows)

    summary_rows: list[dict[str, Any]] = []
    for comparison, group in differences.groupby("comparison", sort=False):
        row = {
            "comparison": comparison,
            "variant_a": group["variant_a"].iloc[0],
            "variant_b": group["variant_b"].iloc[0],
            "folds": len(group),
        }
        for metric in ["rmse", "mae", "r2"]:
            column = f"{metric}_difference_a_minus_b"
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"std_{column}"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    return differences, pd.DataFrame(summary_rows)


def _plot_fold_rmse(fold_metrics: pd.DataFrame, output_dir: Path) -> None:
    if not SHARED_STYLE.exists():
        raise FileNotFoundError(f"Required shared plot style is missing: {SHARED_STYLE}")
    runpy.run_path(str(SHARED_STYLE))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for (variant, color, marker) in zip(VARIANTS, colors, markers):
        subset = fold_metrics[fold_metrics["variant"] == variant]
        ax.plot(
            subset["outer_fold"],
            subset["rmse"] / 1e6,
            color=color,
            marker=marker,
            label=DISPLAY_LABELS[variant],
        )
    ax.set_xlabel("Outer chronological fold")
    ax.set_ylabel(r"Validation RMSE ($10^6$ cells mL$^{-1}$)")
    ax.set_xticks(range(1, OUTER_SPLITS + 1))
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "seasonal_fold_rmse.pdf")
    fig.savefig(output_dir / "seasonal_fold_rmse.png", dpi=300)
    plt.close(fig)


def _verify_outputs(
    train: pd.DataFrame,
    outer_membership: pd.DataFrame,
    inner_membership: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_path: Path | None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected_outer_role_counts = {
        1: {"train": 19, "gap": 1, "validation": 20, "not_available": 80},
        2: {"train": 39, "gap": 1, "validation": 20, "not_available": 60},
        3: {"train": 59, "gap": 1, "validation": 20, "not_available": 40},
        4: {"train": 79, "gap": 1, "validation": 20, "not_available": 20},
        5: {"train": 99, "gap": 1, "validation": 20, "not_available": 0},
    }
    actual_counts: dict[int, dict[str, int]] = {}
    for fold, group in outer_membership.groupby("outer_fold"):
        actual_counts[int(fold)] = {
            role: int((group["role"] == role).sum())
            for role in ["train", "gap", "validation", "not_available"]
        }
    if actual_counts != expected_outer_role_counts:
        raise AssertionError(f"Unexpected outer membership counts: {actual_counts}")
    checks["outer_membership_counts"] = actual_counts

    inner_groups = inner_membership.groupby(["outer_fold", "inner_fold"])
    if len(inner_groups) != OUTER_SPLITS * INNER_SPLITS:
        raise AssertionError("Expected exactly 15 recorded inner folds")
    if not all(int((group["role"] == "gap").sum()) == 1 for _, group in inner_groups):
        raise AssertionError("Every inner fold must have a one-row gap")
    checks["inner_fold_count"] = len(inner_groups)
    checks["inner_gap_rows_per_fold"] = 1

    if len(fold_metrics) != len(VARIANTS) * OUTER_SPLITS:
        raise AssertionError("Expected one fold metric row per variant and outer fold")
    expected_prediction_rows = len(VARIANTS) * OUTER_SPLITS * 20
    if len(predictions) != expected_prediction_rows:
        raise AssertionError("Unexpected number of outer-validation predictions")
    if int(predictions["row_after_time_sort"].max()) >= TRAIN_ROWS:
        raise AssertionError("A post-development row entered the predictions")
    if int(train[TIME_COLUMN].max()) != EXPECTED_DAY_RANGE[1]:
        raise AssertionError("A date after the development boundary entered analysis")

    for (variant, fold), group in predictions.groupby(["variant", "outer_fold"]):
        recorded = fold_metrics[
            (fold_metrics["variant"] == variant) & (fold_metrics["outer_fold"] == fold)
        ].iloc[0]
        direct = _metrics(group["observed"].to_numpy(), group["predicted"].to_numpy())
        for metric, value in direct.items():
            if not np.isclose(value, recorded[metric], rtol=1e-12, atol=1e-8):
                raise AssertionError(f"Direct {metric} check failed for {variant}, fold {fold}")
    checks["direct_metrics_recomputed_from_saved_predictions"] = True
    checks["prediction_rows"] = len(predictions)
    checks["maximum_prediction_row_after_time_sort"] = int(
        predictions["row_after_time_sort"].max()
    )

    by_variant = {
        variant: set(group["row_after_time_sort"])
        for variant, group in predictions.groupby("variant")
    }
    first_membership = by_variant[next(iter(VARIANTS))]
    if any(rows != first_membership for rows in by_variant.values()):
        raise AssertionError("Variants were not scored on identical outer-validation membership")
    checks["identical_validation_membership_across_variants"] = True

    if reference_path is not None:
        reference = pd.read_csv(reference_path)
        mapping = {
            "temperature_only": "temperature_standardized",
            "all10_env": "all10_standardized",
        }
        checked_columns = [
            "selected_alpha", "inner_best_mean_rmse", "rmse", "mae", "r2"
        ]
        maximum_absolute_difference = 0.0
        for variant, reference_variant in mapping.items():
            current = (
                fold_metrics[fold_metrics["variant"] == variant]
                .sort_values("outer_fold")
                .reset_index(drop=True)
            )
            previous = (
                reference[reference["variant"] == reference_variant]
                .sort_values("outer_fold")
                .reset_index(drop=True)
            )
            if current["outer_fold"].tolist() != previous["outer_fold"].tolist():
                raise AssertionError(f"Reference fold IDs differ for {variant}")
            for column in checked_columns:
                delta = np.abs(current[column].to_numpy() - previous[column].to_numpy())
                maximum_absolute_difference = max(maximum_absolute_difference, float(delta.max()))
                if not np.allclose(current[column], previous[column], rtol=1e-12, atol=1e-8):
                    raise AssertionError(f"Reference reproduction failed: {variant} {column}")
        checks["existing_temperature_and_all10_nested_results_reproduced"] = True
        checks["reference_checked_columns"] = checked_columns
        checks["reference_maximum_absolute_difference"] = maximum_absolute_difference
        checks["reference_path"] = str(reference_path.resolve())
    return checks


def run_seasonal_comparison(
    data_path: str | Path,
    output_dir: str | Path,
    comparison_reference: str | Path | None = None,
) -> dict[str, Any]:
    """Run and save the four-way nested development-period comparison."""
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = (
        None if comparison_reference is None else Path(comparison_reference).resolve()
    )
    train, excluded_later_rows = _prepare_training_frame(data_path)

    outer_splits = list(TimeSeriesSplit(n_splits=OUTER_SPLITS, gap=GAP_ROWS).split(train))
    outer_membership, inner_membership, fold_metadata = _split_memberships(
        train, outer_splits
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_fold, (fit_idx, val_idx) in enumerate(outer_splits, start=1):
        inner_splits = list(TimeSeriesSplit(n_splits=INNER_SPLITS, gap=GAP_ROWS).split(fit_idx))
        gap_idx = np.arange(fit_idx[-1] + 1, val_idx[0])
        for variant, features in VARIANTS.items():
            search = GridSearchCV(
                ridge_pipeline(),
                {"model__alpha": ALPHAS},
                scoring="neg_root_mean_squared_error",
                cv=inner_splits,
                refit=True,
                n_jobs=1,
            )
            search.fit(train.iloc[fit_idx][features], train.iloc[fit_idx][TARGET])
            prediction = search.predict(train.iloc[val_idx][features])
            y_validation = train.iloc[val_idx][TARGET].to_numpy()
            fold_rows.append(
                {
                    "variant": variant,
                    "outer_fold": outer_fold,
                    "train_n": len(fit_idx),
                    "gap_n": len(gap_idx),
                    "validation_n": len(val_idx),
                    "train_start_YearDay": int(train.iloc[fit_idx[0]][TIME_COLUMN]),
                    "train_end_YearDay": int(train.iloc[fit_idx[-1]][TIME_COLUMN]),
                    "validation_start_YearDay": int(train.iloc[val_idx[0]][TIME_COLUMN]),
                    "validation_end_YearDay": int(train.iloc[val_idx[-1]][TIME_COLUMN]),
                    "selected_alpha": float(search.best_params_["model__alpha"]),
                    "inner_best_mean_rmse": float(-search.best_score_),
                    **_metrics(y_validation, prediction),
                }
            )
            for row, observed, predicted in zip(val_idx, y_validation, prediction):
                residual = float(observed - predicted)
                prediction_rows.append(
                    {
                        "variant": variant,
                        "outer_fold": outer_fold,
                        "row_after_time_sort": int(row),
                        ID_COLUMN: train.iloc[row][ID_COLUMN],
                        TIME_COLUMN: int(train.iloc[row][TIME_COLUMN]),
                        "observed": float(observed),
                        "predicted": float(predicted),
                        "residual_observed_minus_predicted": residual,
                        "absolute_error": abs(residual),
                        "squared_error": residual**2,
                    }
                )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = _summarize(fold_metrics)
    paired_differences, paired_summary = _paired_differences(fold_metrics)
    checks = _verify_outputs(
        train,
        outer_membership,
        inner_membership,
        fold_metrics,
        predictions,
        reference_path,
    )

    fold_metrics.to_csv(output_dir / "seasonal_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "seasonal_summary.csv", index=False)
    predictions.to_csv(output_dir / "seasonal_predictions.csv", index=False)
    paired_differences.to_csv(output_dir / "seasonal_paired_differences.csv", index=False)
    paired_summary.to_csv(output_dir / "seasonal_paired_summary.csv", index=False)
    outer_membership.to_csv(output_dir / "seasonal_outer_membership.csv", index=False)
    inner_membership.to_csv(output_dir / "seasonal_inner_membership.csv", index=False)
    _plot_fold_rmse(fold_metrics, output_dir)

    metadata_payload = {
        "analysis_scope": "post hoc development-period baseline comparison",
        "test_evaluation_performed": False,
        "test_status": "already seen in the primary course analysis; excluded here",
        "source_data_path": str(data_path),
        "development_rows": TRAIN_ROWS,
        "development_YearDay_range": list(EXPECTED_DAY_RANGE),
        "later_rows_excluded": excluded_later_rows,
        "maximum_row_after_time_sort_used": int(train["row_after_time_sort"].max()),
        "target": TARGET,
        "target_units": TARGET_UNITS,
        "target_transform": "none",
        "annual_encoding": {
            "period_days": ANNUAL_PERIOD_DAYS,
            "phase_origin_YearDay": PHASE_ORIGIN_DAY,
            "formula": "sin/cos(2*pi*(YearDay-phase_origin)/period)",
            "harmonic_count": 1,
            "design_status": "fixed before fitting; period and harmonic count were not tuned",
            "phase_note": "with both sine and cosine components, the fixed origin changes coefficients but not their linear span",
        },
        "variants": {
            variant: {
                "display_label": DISPLAY_LABELS[variant],
                "features": features,
                "standardized_within_pipeline": True,
            }
            for variant, features in VARIANTS.items()
        },
        "ridge_alphas": ALPHAS,
        "nested_validation": {
            "outer": {"type": "TimeSeriesSplit", "n_splits": OUTER_SPLITS, "gap_rows": GAP_ROWS},
            "inner": {"type": "TimeSeriesSplit", "n_splits": INNER_SPLITS, "gap_rows": GAP_ROWS},
            "selection_metric": "mean inner validation RMSE",
            "fold_ids": fold_metadata,
        },
        "artifact_files": {
            "fold_metrics": "seasonal_fold_metrics.csv",
            "summary": "seasonal_summary.csv",
            "predictions_and_errors": "seasonal_predictions.csv",
            "paired_fold_differences": "seasonal_paired_differences.csv",
            "paired_difference_means_and_sample_sds": "seasonal_paired_summary.csv",
            "outer_membership": "seasonal_outer_membership.csv",
            "inner_membership": "seasonal_inner_membership.csv",
            "figure_vector": "seasonal_fold_rmse.pdf",
            "figure_300dpi": "seasonal_fold_rmse.png",
        },
        "plot_style_source": str(SHARED_STYLE),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_environment": Path(sys.prefix).name,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }
    _save_json(output_dir / "seasonal_metadata.json", metadata_payload)
    _save_json(
        output_dir / "seasonal_verification.json",
        {"status": "passed", "checks": checks},
    )
    return {
        "summary": summary.to_dict(orient="records"),
        "checks": checks,
        "output_dir": str(output_dir),
        "test_evaluation_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-reference", type=Path)
    args = parser.parse_args(argv)
    result = run_seasonal_comparison(
        args.data,
        args.output,
        comparison_reference=args.comparison_reference,
    )
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
