# Coastal prokaryotic abundance: OCES 5303 Assignment 2

Cao Runzhen · 21073359

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/runzhencao/2026_OCES_5303/blob/submission-v2/Assignment_2/ass2_oces5303_21073359.ipynb)

The submitted report is [ass2_oces5303_21073359.ipynb](ass2_oces5303_21073359.ipynb). It compares direct neural regression and learned environmental representations on 151 course observations. The scientific question is whether greater capacity or denoising improves estimation in later chronological blocks.

## Run the report

Open the Colab link, connect to a Python CPU runtime, and select **Runtime → Run all**. No Drive mount, private files, access token or GPU is required for the default run. Colab supplies the numerical packages. If running elsewhere, create a Python environment, install `requirements.txt`, and open the notebook in its containing directory.

The default run loads the course table remotely, reproduces selection scores and prediction metrics, rebuilds five figures and the scale comparison, and trains four small CPU demonstrations. The frozen table in notebook metadata allows local offline execution. In Colab the first cell downloads the notebook from this tagged release because an open notebook is not automatically a file in the runtime.

## Repeat the experiments

- `REPEAT_FULL_STUDY=True` repeats the original search in a new directory, using available CUDA devices or CPU. The original experiment used 1,888 neural fits, 472 forest fits and 328 deterministic reference fits. CPU execution can take longer than the archived GPU run.
- `REPEAT_SCALE_CHECK=True` repeats the 189-fit CPU scale sensitivity experiment in its own directory.

Both switches default to `False`. Reanalysis of archived predictions and fresh model training are separate operations. The report's fixed tables describe their recorded experiments; a new training run can differ across package versions and devices. Original package versions are recorded in `results/environment.json`, and the scale experiment records its own versions in `results/scale_sensitivity_2026-10-14/protocol.json`. `requirements-notebook.txt` records the local report-validation environment, while `requirements.txt` gives compatible installation ranges.

## What the extensions test

The matched AE–DAE diagnostic holds architecture, initialization, training duration and Ridge penalty constant, testing the effect of added training noise. The independently selected comparison tests complete pipelines. Both find inconsistent denoising gains. Ten seeds and equal-weight ensembles separate initialization variability from variation across time blocks. The scale sensitivity experiment shows why reducing skewness need not reduce absolute abundance error.

Denoising autoencoders are part of the course material. The extension lies in the controlled comparisons and their interpretation; this repository also supplies the open reproduction route named in the assignment. No grade or originality credit is assumed.

## Files and data attribution

`code/` contains the readable training and plotting sources, `config/` the fixed search, `data/` the supplied course table, and `results/` the archived predictions and numerical records. The same scientific files are embedded in the notebook; no other file is needed when submitting the executed notebook itself.

The [course table](https://raw.githubusercontent.com/julianmak/OCES4303_ML_ocean/refs/heads/main/PIDweekly_env_data.txt) was supplied through Julian Mak's course materials. Dataset and method references appear in the notebook. The final 31 observations had already been inspected in Assignment 1 and are reported as a reused historical holdout. AI assistance is disclosed in the notebook.
