### Scale sensitivity

In the 120-observation development period, abundance has sample skewness 0.639, falling to −0.274 after taking its natural logarithm. Standardization preserves skewness. Chlorophyll and four nutrient columns have skewness from 1.214 to 6.265; `NO2.NO3` and `PO4` include 62 and 16 zeros. These distributions motivate a direct test of whether skewness treatment reduces abundance error.

Three treatments compare raw data, a log target, and log inputs with the raw target. The log-target treatment fits $z=\ln(y/y_0)$, where $y_0=10^6$ cells mL$^{-1}$, and predicts $y_0\exp(\widehat z)$. Log inputs apply $\ln(1+x_j/m_j)$ to chlorophyll and the four nutrient columns, where $m_j$ is the positive median in the current fitting block. This dimensionless transform retains zeros and precedes standardization; all other predictors remain unchanged. Log loss gives less weight to large absolute abundance errors. Direct exponentiation estimates a geometric center and does not include an arithmetic-mean bias correction.

The three outer folds, raw-selected hyperparameters, epochs and ten neural seeds are fixed across treatments; Ridge is deterministic. All 189 fits run on CPU, including a fresh raw baseline. This post-hoc analysis tests sensitivity conditional on the selected settings; a new ranking of optimized transformed pipelines would require their own inner searches. The historical holdout is excluded.

**Table 4. Scale sensitivity of direct regressors.** Mean RMSE in $10^6$ cells mL$^{-1}$, averaging seeds within each fold and then the three folds. The code below recomputes the comparison from original-unit predictions.
