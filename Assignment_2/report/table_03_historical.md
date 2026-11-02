**Table 3. Historical holdout.** RMSE and MAE use $10^6$ cells mL⁻¹. The first three metric columns describe individual models. Individual metrics average ten seeds for stochastic models. Seed SD is the RMSE standard deviation across those ten runs. Ensemble metrics score their equal-weight mean prediction. Deterministic models use one fit.

| Pipeline | RMSE | MAE | R² | Seed SD | Ensemble RMSE |
|---|---|---|---|---|---|
| Training mean | 1.114 | 0.948 | -0.034 | — | 1.114 |
| Temperature OLS | 1.115 | 0.856 | -0.036 | — | 1.115 |
| Ridge | 1.217 | 0.971 | -0.234 | — | 1.217 |
| Random forest | 1.036 | 0.804 | 0.106 | 0.011 | 1.035 |
| Shallow MLP | 1.077 | 0.823 | 0.033 | 0.027 | 1.074 |
| Residual MLP | 1.096 | 0.844 | -0.002 | 0.044 | 1.089 |
| PCA + Ridge | 0.995 | 0.826 | 0.175 | — | 0.995 |
| AE + Ridge | 0.975 | 0.767 | 0.206 | 0.046 | 0.959 |
| DAE + Ridge | 0.981 | 0.768 | 0.195 | 0.059 | 0.961 |
