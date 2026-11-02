# 2026 OCES 5303

Coursework by Cao Runzhen: environmental regression and neural-network estimation of coastal prokaryotic abundance.

| Assignment | Executed English report | Reproduce |
| --- | --- | --- |
| 1: regression, validation and seasonal references | [Notebook](Assignment_1/ass1_oces5303_21073359.ipynb) | [Open in Colab](https://colab.research.google.com/github/runzhencao/2026_OCES_5303/blob/submission-v2/Assignment_1/ass1_oces5303_21073359.ipynb) |
| 2: neural capacity, autoencoders and scale sensitivity | [Notebook](Assignment_2/ass2_oces5303_21073359.ipynb) | [Open in Colab](https://colab.research.google.com/github/runzhencao/2026_OCES_5303/blob/submission-v2/Assignment_2/ass2_oces5303_21073359.ipynb) |

Each assignment directory contains its sources, report and data. Open a notebook in Colab, use a Python CPU runtime, and select **Runtime → Run all**. Assignment 1 fits its analyses from data. Assignment 2 recalculates the archived experiment's metrics and figures and trains four fresh CPU examples; full search and scale-experiment retraining are separate optional switches. This distinction is documented in the Assignment 2 report.

The course table was supplied in [Julian Mak's course materials](https://raw.githubusercontent.com/julianmak/OCES4303_ML_ocean/refs/heads/main/PIDweekly_env_data.txt). Measurement definitions and method references are attributed within each report. The reports describe exploratory analyses of one 151-observation record. AI assistance is disclosed in the notebooks.

The tagged release `submission-v2` fixes the reproduction source. Read each assignment's README for dependencies, training scope and numerical reproducibility limits.

## Verified execution

On 2 November 2026, Assignment 2 completed all 13 code cells in a fresh Google Colab CPU runtime (Python 3.13.18, PyTorch 2.11.0+cpu). The remote course table matched the frozen data; metric assertions passed, five figures and the scale table were regenerated, and four CPU examples were fitted. See the [runtime record](validation/colab_hw2_2026-11-02.json). The optional full search and 189-fit scale retraining were disabled. Both notebooks also completed fresh local-kernel execution.
