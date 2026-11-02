**Table 1. Fixed model families and search ranges.** Neural parameter counts include weights and biases, but exclude Ridge heads. Every tuned model minimizes mean inner RMSE. All neural families evaluate 100, 300 and 800 epochs; the fixed configurations are a sparse combination of the stated ranges.

| Pipeline | Neural parameters | Candidates or structural range |
|---|---:|---|
| Training mean | — | No tuning |
| Temperature OLS | — | One predictor; no tuning |
| Ridge | — | Six penalties: 0.01–1000 |
| Random forest | — | 12 configurations; 400 trees; depth 3 or unrestricted; leaf size 2, 5, 10; feature fraction 0.7, 1.0 |
| Shallow MLP | 97–385 | 12 configurations; hidden width 8, 16, 32 |
| Residual MLP | 737–17,409 | 12 configurations; width 16, 32, 64; one or two blocks |
| PCA + Ridge | — | 2, 4, 6 components × six penalties |
| AE + Ridge | 428–1,968 | 12 configurations; encoder [16] or [32, 16]; latent dimension 2, 4, 6; six Ridge penalties |
| DAE + Ridge | 428–1,968 | Matched AE configurations; Gaussian input noise σ = 0.1 |

The definitions below implement the two direct networks and the common encoder–decoder. The matching AE and DAE architecture lets the later comparison isolate the effect of input corruption.
