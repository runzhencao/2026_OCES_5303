"""Post-hoc scale sensitivity with frozen outer configurations and matched CPU seeds."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew

import study_v2 as study

LOG_FEATURES = ['Chlorophyll', 'NH4', 'NO2.NO3', 'PO4', 'SiO4']
TREATMENTS = ['raw', 'log_target', 'log_inputs']
FAMILIES = ['ridge', 'shallow_mlp', 'residual_mlp']


def transform_inputs(train, valid):
    """Compress nonnegative skewed inputs using fitting-block positive medians."""
    train, valid = train.copy(), valid.copy()
    scales = {}
    for name in LOG_FEATURES:
        j = study.FEATURES.index(name)
        scale = np.median(train[train[:, j] > 0, j])
        scales[name] = float(scale)
        train[:, j] = np.log1p(train[:, j] / scale)
        valid[:, j] = np.log1p(valid[:, j] / scale)
    return train, valid, scales


def scale_prediction(X_train, y_train, X_valid, candidate, seed, treatment):
    """Fit the unchanged selected model; return predictions in original units."""
    scales = {}
    if treatment == 'log_inputs':
        X_train, X_valid, scales = transform_inputs(X_train, X_valid)
    target = np.log(y_train) if treatment == 'log_target' else y_train
    # Evaluation targets are not inputs to fitting or model selection.
    dummy = np.ones(len(X_valid))
    if candidate['family'] in study.NEURAL:
        fitted = study.neural_fit(X_train, target, X_valid, dummy, candidate,
                                  seed, [candidate['epoch']], 'cpu', [None])
    else:
        fitted, _ = study.traditional_fit(X_train, target, X_valid, dummy,
                                          candidate, seed)
    prediction = np.asarray(next(iter(fitted['predictions'].values())))
    if treatment == 'log_target':
        prediction = np.exp(prediction)
    return prediction, scales


def summarize(predictions):
    rows = []
    for (scope, family, treatment, seed), group in predictions.groupby(
            ['scope', 'family', 'treatment', 'seed']):
        rows.append(dict(scope=scope, family=family, treatment=treatment,
                         seed=seed, **study.metrics(group.observed, group.predicted)))
    metrics = pd.DataFrame(rows)
    folds = metrics.groupby(['family', 'treatment', 'scope'])[['rmse', 'mae', 'r2']].mean()
    summary = folds.groupby(['family', 'treatment']).mean()
    summary['fold_sd'] = folds.groupby(['family', 'treatment']).rmse.std()
    return metrics, folds.reset_index(), summary.reset_index()


def run(data_path, results_root, config_path, output):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, X, y = study.load_data(data_path)
    config = json.loads(Path(config_path).read_text())
    splits = json.loads((Path(results_root) / 'splits.json').read_text())
    selected = json.loads((Path(results_root) / 'formal/selected_configurations.json').read_text())
    protocol = dict(families=FAMILIES, treatments=TREATMENTS,
                    seeds=config['evaluation_seeds'], configurations=selected,
                    splits={k: v for k, v in splits.items() if k.startswith('outer_')},
                    device='cpu', target_inverse='exp; no mean bias correction',
                    interpretation='post-hoc fixed-configuration sensitivity; no retuning',
                    log_features=LOG_FEATURES, python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__)
    study.dump(output / 'protocol.json', protocol)
    diagnostic = [{'variable': 'Bacteria_abundance', 'skew_raw': float(skew(y[:120], bias=False)),
                   'skew_log': float(skew(np.log(y[:120]), bias=False)), 'zeros': 0}]
    for name in LOG_FEATURES:
        values = data[name].to_numpy()[:120]
        scale = np.median(values[values > 0])
        diagnostic.append(dict(variable=name, skew_raw=float(skew(values, bias=False)),
                               skew_log=float(skew(np.log1p(values / scale), bias=False)),
                               zeros=int((values == 0).sum())))
    pd.DataFrame(diagnostic).to_csv(output / 'distribution.csv', index=False)
    predictions, preprocessing = [], []
    fits = 0
    for scope, candidates in selected.items():
        train = np.array(splits[scope]['train'])
        valid = np.array(splits[scope]['validation'])
        for family in FAMILIES:
            seeds = config['evaluation_seeds'] if family in study.NEURAL else [0]
            for treatment in TREATMENTS:
                for seed in seeds:
                    p, scales = scale_prediction(X[train], y[train], X[valid],
                                                 candidates[family], seed, treatment)
                    assert np.isfinite(p).all()
                    for index, prediction in zip(valid, p):
                        predictions.append(dict(scope=scope, family=family, treatment=treatment,
                                                seed=seed, row_index=int(index),
                                                YearDay=int(data.iloc[index].YearDay),
                                                observed=float(y[index]), predicted=float(prediction)))
                    if treatment == 'log_inputs' and seed == seeds[0]:
                        preprocessing.append(dict(scope=scope, family=family, scales=scales))
                    fits += 1
                pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
                print(f'{scope} {family} {treatment}: {fits}/189 fits', flush=True)
    metrics, folds, summary = summarize(pd.DataFrame(predictions))
    metrics.to_csv(output / 'metrics.csv', index=False)
    folds.to_csv(output / 'fold_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    study.dump(output / 'preprocessing.json', preprocessing)
    study.dump(output / 'completion.json', dict(status='complete', fits=fits,
                neural_fits=180, ridge_fits=9, elapsed_seconds=time.perf_counter() - started))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.results, args.config, args.output)
