# Assignment 1: environmental regression

[Open in Colab](https://colab.research.google.com/github/runzhencao/2026_OCES_5303/blob/submission-v1/Assignment_1/ass1_oces5303_21073359.ipynb)

The executed notebook is the complete report and contains the data and executable analysis. Open it in a Python CPU runtime and select **Runtime → Run all**. It fits the regression models and performs the nested development and seasonal comparisons. The original 31-observation test block was subsequently reused in Assignment 2 as a historical holdout.

`src/` contains the analysis sources, `data/` the course table, and `report/` a readable report and bibliography. The notebook has no private-file dependency. For local execution, install `requirements.txt` into a Python environment and run all notebook cells.

To reproduce the separate analysis files from this directory:

```bash
python src/run_analysis.py --data data/PIDweekly_env_data.txt --output results
python src/comparisons.py --data data/PIDweekly_env_data.txt --existing-cv-results results/cv_results.csv --output results/comparisons
python src/seasonal.py --data data/PIDweekly_env_data.txt --output results/seasonal --comparison-reference results/comparisons
```

Dataset and method attribution appears in the report. AI assistance is disclosed in the notebook metadata. This release preserves the submitted Assignment 1 analysis.
