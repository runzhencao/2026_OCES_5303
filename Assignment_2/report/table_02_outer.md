**Table 2. Outer chronological validation.** RMSE and MAE use $10^6$ cells mL⁻¹. The first three metric columns describe individual models. Metrics are mean ± sample SD across the three outer folds after averaging seed metrics within each fold. The seed-SD column averages the within-fold RMSE SDs; ensemble RMSE averages the three fold-specific ensemble errors.

| Pipeline | RMSE | MAE | R² | Seed SD | Ensemble RMSE |
|---|---|---|---|---|---|
| Training mean | 1.146 ± 0.160 | 0.874 ± 0.099 | -0.133 ± 0.182 | — | 1.146 |
| Temperature OLS | 1.045 ± 0.146 | 0.768 ± 0.143 | 0.034 ± 0.305 | — | 1.045 |
| Ridge | 1.009 ± 0.257 | 0.750 ± 0.169 | 0.108 ± 0.360 | — | 1.009 |
| Random forest | 1.021 ± 0.194 | 0.713 ± 0.087 | 0.088 ± 0.271 | 0.010 | 1.021 |
| Shallow MLP | 1.120 ± 0.197 | 0.834 ± 0.136 | -0.103 ± 0.327 | 0.051 | 1.111 |
| Residual MLP | 1.028 ± 0.090 | 0.744 ± 0.070 | 0.076 ± 0.144 | 0.071 | 1.019 |
| PCA + Ridge | 1.154 ± 0.189 | 0.894 ± 0.132 | -0.169 ± 0.339 | — | 1.154 |
| AE + Ridge | 1.135 ± 0.128 | 0.844 ± 0.070 | -0.125 ± 0.217 | 0.061 | 1.126 |
| DAE + Ridge | 1.146 ± 0.144 | 0.844 ± 0.093 | -0.154 ± 0.290 | 0.055 | 1.136 |

The next block recomputes abundance errors from individual predictions to check the model comparison. The following figure separates variation across time blocks from variation across initializations.
