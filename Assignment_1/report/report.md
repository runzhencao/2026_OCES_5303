# Estimating coastal prokaryotic abundance from environmental conditions

**OCES 5303 — Assignment 1**  
**Cao Runzhen · Student ID 21073359**


## Background and objective

Seasonal environmental variation accompanies changes in coastal bacterioplankton communities. Weekly observations at the Pivers Island Coastal Observatory (34.7181°N, 76.6707°W) from January 2011 to December 2013 documented recurring seasonal community structure and rapid transitions [1]. Temperature provides a single-variable reference for estimating abundance. Multivariate regression assesses the additional predictive information in other contemporaneous environmental measurements and its transfer to a later period at the same site.

The response `Bacteria_abundance` is interpreted as total prokaryotic abundance in cells mL$^{-1}$. The archive definition includes Bacteria, Archaea and *Prochlorococcus* [2]; overlapping abundance values support this interpretation. The archive covers July 2012–December 2014, whereas the monitoring study covers 2011–2013. All fitted values retain the supplied environmental table.


## Data

**Data source:** [`PIDweekly_env_data.txt`, OCES 5303 dataset](https://github.com/julianmak/OCES5303_ML_ocean/blob/main/assignments/fall26/bacteria_data_scrap.ipynb).


The table contains 151 observations and 15 columns, covering YearDay 19–1091. Of the 150 sampling intervals, 134 are seven days; the others range from four to 21 days. Numerical values are complete and finite, and sample identifiers are unique. Abundance ranges from $1.1404\times10^6$ to $6.7091\times10^6$ cells mL$^{-1}$.

The ten predictors are temperature (`Temp`), salinity, oxygen saturation, pH, dissolved inorganic carbon (`DIC`), chlorophyll, ammonium (`NH4`), `NO2.NO3`, phosphate (`PO4`) and silicate (`SiO4`). They represent water properties, carbonate chemistry, phytoplankton biomass and nutrient availability [1,2]. Temperature is in °C. The supplied `NH4` scale and the construction and unit of `NO2.NO3` remain unresolved; these fields retain their numerical scales. `SampleID` identifies observations and `YearDay` defines temporal order. The predictor specification retains measured water properties, excludes projected insolation, and excludes `MLLW` because its supplied definition is unresolved. Annual harmonics provide a separate seasonal comparison.

## Evaluation design

The first 120 observations (YearDay 19–872) form the development set; the final 31 (YearDay 879–1091) form the test set. Five expanding-window `TimeSeriesSplit` folds within the development set have fitting sizes of 19, 39, 59, 79 and 99, validation sizes of 20, and a one-observation gap. The gap separates adjacent observations while longer temporal dependence remains. Validation periods have equal observation counts and different calendar durations.

Chronological validation evaluates transfer to later observations. Imputation and standardization are fitted within each training fold. Hyperparameters minimize mean validation RMSE; MAE and $R^2$ complement the squared-error criterion. RMSE gives greater weight to large abundance errors. All candidates are refitted on the 120 development observations before test evaluation.


## Models

Abundance varies substantially at a given temperature (Figure 1). The training-mean predictor supplies a constant reference. Models are implemented with scikit-learn [3]. Three fitted models compare a temperature-only linear regression, standardized ten-predictor Ridge regression and a random forest.

Temperature-only ordinary least squares minimizes squared residuals over an intercept and slope. Ridge minimizes

$$\sum_{i\in\mathcal{T}}(y_i-b-\boldsymbol{z}_i^\mathsf{T}\boldsymbol{w})^2+\alpha\lVert\boldsymbol{w}\rVert_2^2,$$

where $\mathcal{T}$ is the training subset and $\boldsymbol{z}_i$ contains standardized predictors. The intercept is unpenalized. Standardization places coefficients on comparable scales under the penalty; coefficient shrinkage addresses correlated predictors. The six penalties are $\{0.01,0.1,1,10,100,1000\}$.

The forest averages 400 regression trees fitted to bootstrap samples, representing nonlinear responses and interactions [4]. Its 12 configurations combine maximum depth 3 or unrestricted depth, minimum leaf sizes 2, 5 or 10, and predictor fractions 0.7 or 1.0. The random seed is 5303. Trees retain original predictor scales because threshold splits do not require standardization. Median imputation is fitted in each pipeline and leaves the complete data unchanged.


## Model performance

Temperature-only regression has the lowest mean validation RMSE (0.952 million cells mL$^{-1}$), followed by the forest (0.967 million); Ridge gives 1.105 million. Ridge mean validation RMSE decreases from 1.592 million at $\alpha=0.01$ to 1.105 million at the selected $\alpha=100$. The selected forest uses unrestricted depth, all predictors and a minimum leaf size of 5, giving RMSE 0.9666 million. A leaf size of 2 gives 0.9671 million. The difference is small relative to the selected forest's fold SD of 0.124 million.

Test performance reverses the validation ranking. Ridge has the lowest RMSE (0.987 million) and highest $R^2$ (0.188), reducing RMSE by 11.4% relative to the training-mean reference. The forest has the lowest MAE (0.775 million; an 18.2% reduction), with RMSE 1.008 million. Temperature-only regression has test $R^2=-0.036$ and RMSE 1.115 million, close to the mean reference's RMSE of 1.114 million.

The held-out comparison evaluates four fixed candidates. Selection of a single model family by validation RMSE favors temperature-only regression. The later-period comparison favors different models for RMSE and MAE; the highest test $R^2$ accounts for approximately 19% of variance relative to the test-set mean.


## Predictor and validation comparisons

Post hoc comparisons are restricted to the 120 development observations. Five outer chronological folds evaluate temperature-only Ridge, standardized ten-predictor Ridge and unscaled ten-predictor Ridge. Three inner chronological folds select $\alpha$ from the same six-value grid within each outer training window. Every split retains a one-observation gap.


Temperature-only Ridge has the lowest mean outer-fold RMSE (0.996 million cells mL$^{-1}$), compared with 1.163 million for standardized ten-predictor Ridge and 1.250 million for unscaled ten-predictor Ridge. Adding the other predictors increases mean validation error under these splits and the tested penalty grid.

Standardization reduces the ten-predictor mean RMSE by 0.087 million, with the difference concentrated in the earliest outer fold. Unscaled Ridge has lower RMSE in the other four folds and a fold SD of 0.607 million. The comparison combines scaling with independent penalty selection on the same numerical grid. Its outcome depends on the supplied predictor scales and fitting periods.


Matched shuffled validation gives mean RMSE 1.134 million cells mL$^{-1}$, compared with 1.163 million chronologically. The paired shuffled-minus-chronological difference is $-0.029\pm0.279$ million, with dispersion expressed as fold SD. Training, omitted and validation counts are matched, while sample membership and temporal direction differ. Shuffled training includes later observations and evaluates interpolation within each prefix. The mean difference is small relative to fold variation.


Development-period hyperparameter sensitivity. Points show mean RMSE across the five original chronological folds; error bars show fold SD (ddof = 0). Forest labels specify maximum depth, predictor fraction and minimum leaf size.


Temperature has the largest training Spearman correlation with abundance ($\rho=0.540$) and forest impurity importance (0.425). Chlorophyll ($\rho=0.404$) and silicate ($\rho=0.365$) also have positive rank correlations. These associations combine seasonal covariation and correlated environmental measurements. Impurity importance quantifies predictor usage in the fitted forest. Split-selection bias is documented for impurity-based tree importance [5].

## Annual seasonal baseline

A fixed annual harmonic supplies a seasonal reference [6]:

$$s(t)=\sin(2\pi t/365.25),\qquad c(t)=\cos(2\pi t/365.25),$$

where $t$ is `YearDay`. The period and single harmonic are fixed. Four standardized Ridge specifications use annual terms, temperature, ten environmental predictors, or annual terms combined with those predictors. The same five outer and three inner chronological folds evaluate each specification, with penalties selected within each outer training window. This post hoc comparison is restricted to the development period.


Errors are in $10^6$ cells mL$^{-1}$; dispersion is sample SD across five outer folds. Annual-only Ridge has mean RMSE 0.949, compared with 0.996 for temperature, 1.163 for ten environmental predictors, and 1.065 for annual terms combined with environmental predictors. Annual-minus-temperature and annual-minus-combined paired differences are $-0.046\pm0.204$ and $-0.115\pm0.246$, respectively.

Adding environmental predictors to the annual reference increases mean error under the tested specifications and penalty grid. Fold variation is substantial, and the earliest fitting window covers less than one annual cycle. The annual basis captures predictive structure within the observed development periods.


## Discussion and conclusion

Evaluation covers one later block from a single-site, 151-observation record. Its 31 test observations are temporally dependent and cover part of an annual cycle. Predictions use environmental measurements from the same sampling occasion. Interannual stability, spatial transfer and prediction with future covariates require independent validation.

Multivariable Ridge gives the lowest held-out RMSE, while the forest gives the lowest MAE. Their test $R^2$ values of 0.188 and 0.154 leave substantial later-period variation unresolved. Nested development validation favors temperature-only Ridge; the annual reference gives the lowest mean RMSE in the seasonal comparison. Model preference depends on the evaluation period, predictor specification and error metric.

## References

1. Ward, C. S. *et al.* Annual community patterns are driven by seasonal switching between closely related marine bacteria. *The ISME Journal* **11**, 1412–1422 (2017). https://doi.org/10.1038/ismej.2017.4
2. Johnson, Z. & Hunt, D. (2017). *Environmental and physical data associated with ocean acidification microbe adaptation from 2012–2014* [Data set]. Biological and Chemical Oceanography Data Management Office (BCO-DMO). Version 1, 25 May 2017. https://doi.org/10.1575/1912/bco-dmo.700974.1 (accessed 1 October 2026).
3. Pedregosa, F. *et al.* Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* **12**, 2825–2830 (2011). https://jmlr.org/papers/v12/pedregosa11a.html
4. Breiman, L. Random forests. *Machine Learning* **45**, 5–32 (2001). https://doi.org/10.1023/A:1010933404324
5. Strobl, C., Boulesteix, A.-L., Zeileis, A. & Hothorn, T. Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics* **8**, 25 (2007). https://doi.org/10.1186/1471-2105-8-25
6. Cornelissen, G. Cosinor-based rhythmometry. *Theoretical Biology and Medical Modelling* **11**, 16 (2014). https://doi.org/10.1186/1742-4682-11-16
