"""Training-only supplemental comparisons for OCES 5303 Assignment 1.

This module never evaluates the locked final 31 observations.  It compares
Ridge feature/preprocessing choices with nested chronological validation and
contrasts chronological transfer with a matched-size shuffled interpolation
diagnostic.  The latter is descriptive: its validation observations differ and
it does not estimate the causal effect or bias of chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

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
TIME_COLUMN = "YearDay"
ID_COLUMN = "SampleID"
TRAIN_ROWS = 120
EXPECTED_TRAIN_MAX_DAY = 872
OUTER_SPLITS = 5
INNER_SPLITS = 3
GAP_ROWS = 1
RANDOM_SEED = 5303
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
VARIANTS = {
    "temperature_standardized": {"features": ["Temp"], "standardize": True},
    "all10_standardized": {"features": MAIN_FEATURES, "standardize": True},
    "all10_unscaled": {"features": MAIN_FEATURES, "standardize": False},
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packages() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy"]:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _pipeline(standardize: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", Ridge()))
    return Pipeline(steps)


def _metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def _validate_training_frame(train: pd.DataFrame) -> pd.DataFrame:
    required = {ID_COLUMN, TIME_COLUMN, TARGET, *MAIN_FEATURES}
    absent = sorted(required.difference(train.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if len(train) != TRAIN_ROWS:
        raise ValueError(f"Expected exactly {TRAIN_ROWS} locked development rows, got {len(train)}")
    checked = train.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True).copy()
    if int(checked[TIME_COLUMN].max()) != EXPECTED_TRAIN_MAX_DAY:
        raise ValueError("Development boundary mismatch; refusing possible access to locked test rows")
    if not checked[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("YearDay must be monotonic after sorting")
    if checked[ID_COLUMN].duplicated().any() or checked[TIME_COLUMN].duplicated().any():
        raise ValueError("SampleID and YearDay must be unique")
    values = checked[[TARGET, *MAIN_FEATURES]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training target and predictors must be finite")
    checked["row_after_time_sort"] = np.arange(TRAIN_ROWS)
    return checked


def _chronological_membership(
    train: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
        gap_idx = np.arange(fit_idx[-1] + 1, val_idx[0])
        for row in range(TRAIN_ROWS):
            role = "not_seen"
            if row in fit_idx:
                role = "train"
            elif row in gap_idx:
                role = "gap"
            elif row in val_idx:
                role = "validation"
            records.append(
                {
                    "outer_fold": fold,
                    "row_after_time_sort": row,
                    ID_COLUMN: train.loc[row, ID_COLUMN],
                    TIME_COLUMN: int(train.loc[row, TIME_COLUMN]),
                    "role": role,
                }
            )
    result = pd.DataFrame(records)
    assert int(result["row_after_time_sort"].max()) == TRAIN_ROWS - 1
    return result


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["folds"] = len(group)
        for measure in ["rmse", "mae", "r2"]:
            row[f"mean_{measure}"] = float(group[measure].mean())
            row[f"std_{measure}"] = float(group[measure].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(
    frame: pd.DataFrame, label_column: str, labels: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for label_a, label_b in combinations(labels, 2):
        a = frame[frame[label_column] == label_a].set_index("outer_fold")
        b = frame[frame[label_column] == label_b].set_index("outer_fold")
        for fold in sorted(set(a.index).intersection(b.index)):
            records.append(
                {
                    "outer_fold": fold,
                    "comparison": f"{label_a}_minus_{label_b}",
                    "label_a": label_a,
                    "label_b": label_b,
                    "rmse_difference_a_minus_b": float(a.loc[fold, "rmse"] - b.loc[fold, "rmse"]),
                    "mae_difference_a_minus_b": float(a.loc[fold, "mae"] - b.loc[fold, "mae"]),
                    "r2_difference_a_minus_b": float(a.loc[fold, "r2"] - b.loc[fold, "r2"]),
                }
            )
    differences = pd.DataFrame(records)
    summaries: list[dict[str, Any]] = []
    for comparison, group in differences.groupby("comparison", sort=False):
        row: dict[str, Any] = {
            "comparison": comparison,
            "label_a": group["label_a"].iloc[0],
            "label_b": group["label_b"].iloc[0],
            "folds": len(group),
        }
        for measure in ["rmse", "mae", "r2"]:
            column = f"{measure}_difference_a_minus_b"
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"std_{column}"] = float(group[column].std(ddof=1))
        summaries.append(row)
    return differences, pd.DataFrame(summaries)


def _matched_random_inner_splits(
    n_samples: int,
    chronological_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, int]]]:
    random_splits: list[tuple[np.ndarray, np.ndarray]] = []
    audit: list[dict[str, int]] = []
    for inner_fold, (chron_train, chron_val) in enumerate(chronological_splits, start=1):
        rng = np.random.default_rng(seed + 100 * inner_fold)
        permutation = rng.permutation(n_samples)
        train_n, val_n = len(chron_train), len(chron_val)
        train_positions = permutation[:train_n]
        omitted_positions = permutation[train_n : train_n + GAP_ROWS]
        val_positions = permutation[train_n + GAP_ROWS : train_n + GAP_ROWS + val_n]
        if len(val_positions) != val_n:
            raise ValueError("Matched random inner split is too large for the outer training set")
        random_splits.append((train_positions, val_positions))
        audit.append(
            {
                "inner_fold": inner_fold,
                "train_n": train_n,
                "validation_n": val_n,
                "omitted_match_n": len(omitted_positions),
                "unused_n": n_samples - train_n - val_n - len(omitted_positions),
            }
        )
    return random_splits, audit


def _plot_nested_ridge(frame: pd.DataFrame, path: Path) -> None:
    labels = {
        "temperature_standardized": "Temperature only, standardized",
        "all10_standardized": "All 10, standardized",
        "all10_unscaled": "All 10, unscaled",
    }
    colors = {
        "temperature_standardized": "#f28e2b",
        "all10_standardized": "#28628b",
        "all10_unscaled": "#7b2cbf",
    }
    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    for variant in VARIANTS:
        subset = frame[frame["variant"] == variant]
        ax.plot(
            subset["outer_fold"], subset["rmse"] / 1e6, marker="o", lw=1.7,
            color=colors[variant], label=labels[variant],
        )
    ax.set(
        xlabel="Outer chronological fold",
        ylabel=r"Validation RMSE ($10^6$ cells mL$^{-1}$)",
        xticks=range(1, OUTER_SPLITS + 1),
        title="Nested chronological comparison of Ridge specifications",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_validation_design(frame: pd.DataFrame, path: Path) -> None:
    labels = {"chronological": "Chronological later-block", "shuffled": "Matched shuffled"}
    colors = {"chronological": "#28628b", "shuffled": "#d1495b"}
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for design in ["chronological", "shuffled"]:
        subset = frame[frame["design"] == design]
        ax.plot(
            subset["outer_fold"], subset["rmse"] / 1e6, marker="o", lw=1.7,
            color=colors[design], label=labels[design],
        )
    ax.set(
        xlabel="Matched outer fold size",
        ylabel=r"Validation RMSE ($10^6$ cells mL$^{-1}$)",
        xticks=range(1, OUTER_SPLITS + 1),
        title="Validation design diagnostic for standardized all-feature Ridge",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_existing_sensitivity(cv_results: pd.DataFrame, path: Path) -> None:
    ridge = cv_results[cv_results["model"] == "ridge"].copy()
    forest = cv_results[cv_results["model"] == "random_forest"].copy()
    if len(ridge) != 6 or len(forest) != 12:
        raise ValueError("Expected exactly 6 Ridge and 12 random-forest candidates")
    ridge["alpha"] = ridge["params_json"].map(lambda value: json.loads(value)["model__alpha"])
    forest = forest.sort_values("mean_cv_rmse").reset_index(drop=True)
    forest["short_label"] = forest["params_json"].map(
        lambda value: (
            "D" + ("∞" if json.loads(value)["model__max_depth"] is None else str(json.loads(value)["model__max_depth"]))
            + f"/F{json.loads(value)['model__max_features']:g}"
            + f"/L{json.loads(value)['model__min_samples_leaf']}"
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    axes[0].errorbar(
        ridge["alpha"], ridge["mean_cv_rmse"] / 1e6,
        yerr=ridge["std_cv_rmse"] / 1e6, marker="o", capsize=3, color="#28628b",
    )
    axes[0].set_xscale("log")
    axes[0].set(
        xlabel=r"Ridge penalty $\alpha$",
        ylabel=r"Mean validation RMSE ($10^6$ cells mL$^{-1}$)",
        title="Ridge sensitivity",
    )
    axes[0].grid(alpha=0.25)

    x = np.arange(len(forest))
    axes[1].errorbar(
        x, forest["mean_cv_rmse"] / 1e6,
        yerr=forest["std_cv_rmse"] / 1e6, fmt="o", capsize=3, color="#4f772d",
    )
    axes[1].set_xticks(x, forest["short_label"], rotation=55, ha="right")
    axes[1].set(
        xlabel="RF candidate sorted by mean RMSE (D/F/L)",
        ylabel=r"Mean validation RMSE ($10^6$ cells mL$^{-1}$)",
        title="Random-forest sensitivity",
    )
    axes[1].grid(alpha=0.25)
    fig.suptitle("Original training-only hyperparameter sensitivity; bars show fold SD", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_comparisons(
    train: pd.DataFrame,
    output_dir: str | Path,
    existing_cv_results_path: str | Path,
    source_data_sha256: str | None = None,
) -> dict[str, Any]:
    """Run all supplemental analyses using exactly the 120 development rows."""
    train = _validate_training_frame(train)
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outer_cv = TimeSeriesSplit(n_splits=OUTER_SPLITS, gap=GAP_ROWS)
    outer_splits = list(outer_cv.split(train))
    chronological_membership = _chronological_membership(train, outer_splits)
    chronological_membership.to_csv(output_dir / "nested_ridge_outer_membership.csv", index=False)

    fold_rows: list[dict[str, Any]] = []
    inner_size_rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_val) in enumerate(outer_splits, start=1):
        inner_cv = TimeSeriesSplit(n_splits=INNER_SPLITS, gap=GAP_ROWS)
        inner_splits = list(inner_cv.split(outer_train))
        for inner_fold, (inner_train, inner_val) in enumerate(inner_splits, start=1):
            inner_size_rows.append(
                {
                    "outer_fold": outer_fold,
                    "design": "chronological",
                    "inner_fold": inner_fold,
                    "train_n": len(inner_train),
                    "validation_n": len(inner_val),
                    "omitted_match_n": GAP_ROWS,
                    "unused_n": len(outer_train) - len(inner_train) - len(inner_val) - GAP_ROWS,
                }
            )
        for variant, specification in VARIANTS.items():
            features = specification["features"]
            search = GridSearchCV(
                _pipeline(specification["standardize"]),
                {"model__alpha": ALPHAS},
                scoring="neg_root_mean_squared_error",
                cv=inner_splits,
                refit=True,
                n_jobs=1,
            )
            search.fit(train.iloc[outer_train][features], train.iloc[outer_train][TARGET])
            prediction = search.predict(train.iloc[outer_val][features])
            fold_rows.append(
                {
                    "variant": variant,
                    "outer_fold": outer_fold,
                    "train_n": len(outer_train),
                    "gap_n": outer_val[0] - outer_train[-1] - 1,
                    "validation_n": len(outer_val),
                    "train_start_YearDay": int(train.iloc[outer_train[0]][TIME_COLUMN]),
                    "train_end_YearDay": int(train.iloc[outer_train[-1]][TIME_COLUMN]),
                    "validation_start_YearDay": int(train.iloc[outer_val[0]][TIME_COLUMN]),
                    "validation_end_YearDay": int(train.iloc[outer_val[-1]][TIME_COLUMN]),
                    "selected_alpha": float(search.best_params_["model__alpha"]),
                    "inner_best_mean_rmse": float(-search.best_score_),
                    **_metrics(train.iloc[outer_val][TARGET], prediction),
                }
            )

    nested_folds = pd.DataFrame(fold_rows)
    nested_folds.to_csv(output_dir / "nested_ridge_fold_metrics.csv", index=False)
    nested_summary = _summary(nested_folds, ["variant"])
    nested_summary.to_csv(output_dir / "nested_ridge_summary.csv", index=False)
    nested_differences, nested_difference_summary = _paired_differences(
        nested_folds, "variant", list(VARIANTS)
    )
    nested_differences.to_csv(output_dir / "nested_ridge_paired_differences.csv", index=False)
    nested_difference_summary.to_csv(
        output_dir / "nested_ridge_paired_summary.csv", index=False
    )
    _plot_nested_ridge(nested_folds, figures_dir / "01_nested_ridge_comparison.png")

    design_rows: list[dict[str, Any]] = []
    design_membership_rows: list[dict[str, Any]] = []
    chronological_all10 = nested_folds[nested_folds["variant"] == "all10_standardized"]
    for row in chronological_all10.to_dict(orient="records"):
        design_rows.append({"design": "chronological", **{k: v for k, v in row.items() if k != "variant"}})
    for record in chronological_membership.to_dict(orient="records"):
        design_membership_rows.append({"design": "chronological", **record})

    for outer_fold, (chron_train, chron_val) in enumerate(outer_splits, start=1):
        seed = RANDOM_SEED + outer_fold - 1
        prefix = np.arange(chron_val[-1] + 1)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(prefix)
        train_n, val_n = len(chron_train), len(chron_val)
        random_train = np.sort(permutation[:train_n])
        random_omitted = np.sort(permutation[train_n : train_n + GAP_ROWS])
        random_val = np.sort(permutation[train_n + GAP_ROWS : train_n + GAP_ROWS + val_n])
        if train_n + GAP_ROWS + val_n != len(prefix):
            raise AssertionError("Outer matched partition must exhaust the chronological prefix")

        chronological_inner = list(
            TimeSeriesSplit(n_splits=INNER_SPLITS, gap=GAP_ROWS).split(chron_train)
        )
        random_inner, random_inner_audit = _matched_random_inner_splits(
            train_n, chronological_inner, seed
        )
        for audit in random_inner_audit:
            inner_size_rows.append(
                {"outer_fold": outer_fold, "design": "shuffled", **audit}
            )

        search = GridSearchCV(
            _pipeline(True),
            {"model__alpha": ALPHAS},
            scoring="neg_root_mean_squared_error",
            cv=random_inner,
            refit=True,
            n_jobs=1,
        )
        search.fit(train.iloc[random_train][MAIN_FEATURES], train.iloc[random_train][TARGET])
        prediction = search.predict(train.iloc[random_val][MAIN_FEATURES])
        design_rows.append(
            {
                "design": "shuffled",
                "outer_fold": outer_fold,
                "train_n": train_n,
                "gap_n": len(random_omitted),
                "validation_n": val_n,
                "train_start_YearDay": int(train.iloc[random_train][TIME_COLUMN].min()),
                "train_end_YearDay": int(train.iloc[random_train][TIME_COLUMN].max()),
                "validation_start_YearDay": int(train.iloc[random_val][TIME_COLUMN].min()),
                "validation_end_YearDay": int(train.iloc[random_val][TIME_COLUMN].max()),
                "selected_alpha": float(search.best_params_["model__alpha"]),
                "inner_best_mean_rmse": float(-search.best_score_),
                "seed": seed,
                "training_later_than_earliest_validation_n": int(
                    (train.iloc[random_train][TIME_COLUMN] > train.iloc[random_val][TIME_COLUMN].min()).sum()
                ),
                **_metrics(train.iloc[random_val][TARGET], prediction),
            }
        )
        roles = {int(row): "not_available" for row in range(TRAIN_ROWS)}
        roles.update({int(row): "train" for row in random_train})
        roles.update({int(row): "omitted_match" for row in random_omitted})
        roles.update({int(row): "validation" for row in random_val})
        for row in range(TRAIN_ROWS):
            design_membership_rows.append(
                {
                    "design": "shuffled",
                    "outer_fold": outer_fold,
                    "row_after_time_sort": row,
                    ID_COLUMN: train.loc[row, ID_COLUMN],
                    TIME_COLUMN: int(train.loc[row, TIME_COLUMN]),
                    "role": roles[row],
                }
            )

    design_folds = pd.DataFrame(design_rows)
    design_folds.to_csv(output_dir / "validation_design_fold_metrics.csv", index=False)
    _summary(design_folds, ["design"]).to_csv(
        output_dir / "validation_design_summary.csv", index=False
    )
    design_differences, design_difference_summary = _paired_differences(
        design_folds, "design", ["shuffled", "chronological"]
    )
    design_differences.to_csv(
        output_dir / "validation_design_paired_differences.csv", index=False
    )
    design_difference_summary.to_csv(
        output_dir / "validation_design_paired_summary.csv", index=False
    )
    design_membership = pd.DataFrame(design_membership_rows)
    if int(design_membership["row_after_time_sort"].max()) >= TRAIN_ROWS:
        raise AssertionError("Locked test row entered comparison membership")
    design_membership.to_csv(output_dir / "validation_design_membership.csv", index=False)
    pd.DataFrame(inner_size_rows).to_csv(
        output_dir / "validation_design_inner_split_sizes.csv", index=False
    )
    _plot_validation_design(
        design_folds, figures_dir / "02_validation_design_diagnostic.png"
    )

    existing_cv_results = pd.read_csv(existing_cv_results_path)
    _plot_existing_sensitivity(
        existing_cv_results, figures_dir / "03_existing_hyperparameter_sensitivity.png"
    )

    metadata_payload = {
        "analysis_scope": "training-only supplemental comparisons",
        "locked_test_rows_used": 0,
        "maximum_row_after_time_sort_used": int(train["row_after_time_sort"].max()),
        "development_rows": TRAIN_ROWS,
        "development_YearDay_range": [int(train[TIME_COLUMN].min()), int(train[TIME_COLUMN].max())],
        "target": TARGET,
        "target_units": "cells mL^-1",
        "target_transform": "none",
        "main_features": MAIN_FEATURES,
        "ridge_alphas": ALPHAS,
        "nested_chronological": {
            "outer": {"type": "TimeSeriesSplit", "n_splits": OUTER_SPLITS, "gap_rows": GAP_ROWS},
            "inner": {"type": "TimeSeriesSplit", "n_splits": INNER_SPLITS, "gap_rows": GAP_ROWS},
            "selection_metric": "mean inner validation RMSE",
        },
        "shuffled_diagnostic": {
            "outer": "random partition of each chronological prefix with exactly matched train, omitted, and validation counts",
            "inner": "custom seeded random folds with train, omitted, and validation counts matched to corresponding inner TimeSeriesSplit folds",
            "seeds": list(range(RANDOM_SEED, RANDOM_SEED + OUTER_SPLITS)),
            "interpretation": (
                "Shuffling permits later dates in training for earlier validation dates and evaluates interpolation. "
                "Because validation membership differs, paired score differences are descriptive and do not estimate causal validation bias."
            ),
        },
        "source_data_sha256": source_data_sha256,
        "existing_cv_results_path": str(Path(existing_cv_results_path).resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_environment": Path(sys.prefix).name,
        "platform": platform.platform(),
        "package_versions": _packages(),
        "random_seed": RANDOM_SEED,
    }
    _save_json(output_dir / "comparison_metadata.json", metadata_payload)

    return {
        "nested_summary": nested_summary.to_dict(orient="records"),
        "validation_design_summary": _summary(design_folds, ["design"]).to_dict(orient="records"),
        "output_dir": str(output_dir),
        "locked_test_rows_used": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--existing-cv-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = pd.read_csv(args.data, sep=r"\s+").sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    if len(raw) <= TRAIN_ROWS:
        raise ValueError("Expected a locked test block after the first 120 rows")
    train = raw.iloc[:TRAIN_ROWS].copy()
    summary = run_comparisons(
        train,
        args.output,
        args.existing_cv_results,
        source_data_sha256=_sha256(args.data),
    )
    print(json.dumps(_json_ready(summary), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
