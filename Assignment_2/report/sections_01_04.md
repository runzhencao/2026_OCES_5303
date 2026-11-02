# Neural network capacity and denoising representations for coastal prokaryotic abundance estimation

**Cao Runzhen · 21073359 · OCES 5303 · Assignment 2**

## 1. Scientific motivation and objectives

Coastal microbial abundance varies alongside the physical, chemical and biological conditions of seawater. At the Pivers Island Coastal Observatory (PICO), Ward et al. (2017) examined three years of weekly observations and identified recurring seasonal changes in bacterioplankton community composition. Total prokaryotic abundance quantifies aggregate cell concentration. Estimating this quantity from concurrent environmental measurements tests how much of its variation can be represented by a compact set of covariates.

Direct regression maps the environmental variables to abundance using either a shallow multilayer perceptron (MLP) or a larger residual MLP. Representation learning first compresses the environmental inputs with an autoencoder (AE) or denoising autoencoder (DAE), then estimates abundance with Ridge regression. These approaches impose different learning objectives: the MLP learns abundance directly, whereas the autoencoder learns environmental reconstruction. Linear regression, random forest and principal component analysis (PCA) provide simpler references.

The comparison assesses whether increased capacity improves estimation in later chronological blocks, and whether nonlinear compression or denoising preserves useful abundance information. Repeated initializations and equal-weight ensembles assess the stability of both approaches. A separate scale sensitivity analysis tests whether skewness treatment changes the errors of direct regressors.

## 2. Data and estimation task

The supplied table contains 151 complete observations and 15 columns, with unique sample identifiers and finite numerical values. After chronological sorting, `YearDay` spans 19–1091 with a median sampling interval of seven days. This index supplies temporal order; its calendar origin is not explicitly encoded in the file. The first 120 observations, ending at YearDay 872, form the development period. The remaining 31, YearDay 879–1091, form the historical holdout already examined in Assignment 1 (Figure 1).

The response `Bacteria_abundance` ranges from 1.1404 × 10⁶ to 6.7091 × 10⁶ cells mL⁻¹. Archive parameter definitions and overlapping values support interpreting this response as total prokaryotic concentration, including Bacteria, Archaea and *Prochlorococcus* (Johnson and Hunt, 2017). All models use the supplied extract; the external archive supplies measurement definitions.

The ten inputs comprise temperature and salinity; oxygen saturation, pH and dissolved inorganic carbon; chlorophyll *a*; and `NH4`, `NO2.NO3`, `PO4` and `SiO4`. They represent water properties, carbonate chemistry, phytoplankton biomass and nutrient availability (Johnson and Hunt, 2017). Temperature is in °C. The main comparison retains the supplied numerical scales because the scale of `NH4` and the construction and unit of `NO2.NO3` remain unresolved. `SampleID` identifies observations and `YearDay` orders them. The predictor specification retains measured water properties, excludes projected insolation, and excludes `MLLW` because its supplied definition is unresolved. All 151 observations are retained; transformed predictions in the sensitivity analysis are evaluated in the original abundance units.

The mapping is environment at time $t$ to abundance at time $t$. Later-block evaluation therefore requires concurrent environmental measurements; it assesses transfer of an estimation relationship through time.

The first code block loads the [course table](https://raw.githubusercontent.com/julianmak/OCES4303_ML_ocean/refs/heads/main/PIDweekly_env_data.txt) remotely and checks it against the experiment's frozen copy. The notebook metadata stores that copy, source code and results so that the analysis also runs offline. For local execution, save the notebook in the kernel's working directory under the filename below. In Colab, the first cell retrieves the tagged public copy automatically. Figure 1 then places abundance changes alongside the chronological fitting and evaluation windows.

## 3. Models and training

### 3.1 Direct abundance regression

For a standardized input vector $\mathbf{x}^{*}$, both MLPs estimate a standardized target and return predictions to the abundance scale:

$$\widehat y=\mu_y+s_y f_\theta(\mathbf{x}^{*}),\qquad
\mathbf{x}^{*}=(\mathbf{x}-\boldsymbol\mu_x)/\mathbf{s}_x.$$

All means and scales are learned from the current fitting block. Standardization puts inputs with different numerical scales on comparable footing during optimization. The shallow MLP has one Tanh hidden layer with 8, 16 or 32 units, followed by dropout and a linear output. Its 97–385 parameters provide a low-capacity nonlinear reference. The residual MLP projects the ten inputs to width 16, 32 or 64 and applies one or two residual blocks:

$$\mathbf h_{b+1}=\mathbf h_b+0.5\left[W_{b,2}\operatorname{Dropout}\!\left(\tanh(W_{b,1}\mathbf h_b+\mathbf b_{b,1})\right)+\mathbf b_{b,2}\right].$$

A final Tanh and linear layer produce the standardized prediction. This fully connected adaptation uses the shortcut principle of He et al. (2016). Its 737–17,409 parameters provide a joint comparison of increased capacity and residual architecture.

Training minimizes mean squared error with full-batch AdamW in `PyTorch` (PyTorch contributors, n.d.). Each neural family has 12 fixed base configurations. Candidate learning rates are 0.0003, 0.001 and 0.003; weight decay values are 0.0001, 0.001, 0.01 and 0.1; MLP dropout probabilities are 0, 0.1 and 0.2. A sparse set of combinations covers these ranges. Epoch counts of 100, 300 and 800 are selected by inner validation, with checkpoints sharing one training trajectory. Predictions retain the unconstrained linear output.

### 3.2 Environmental representations followed by Ridge

The AE encodes the standardized inputs into two, four or six latent variables and decodes them to reconstruct all ten inputs. Encoder hidden widths are either [16] or [32, 16], with a symmetric decoder, Tanh hidden activations and linear bottleneck and reconstruction layers. The network has 428–1,968 parameters. Its objective is

$$L_{\mathrm{rec}}=\frac{1}{10n}\sum_{i=1}^{n}\left\|g_\phi(e_\theta(\widetilde{\mathbf{x}}_i^{*}))-\mathbf{x}_i^{*}\right\|_2^2.$$

For the ordinary AE, $\widetilde{\mathbf{x}}^{*}=\mathbf{x}^{*}$. The DAE adds independent Gaussian noise with standard deviation 0.1 to each standardized training input, resampled every epoch, while retaining the clean reconstruction target. This training regularization follows the corrupted-input, clean-target principle of Vincent et al. (2008). Both pipelines receive clean inputs at evaluation.

After reconstruction training, the encoder is frozen. Its latent variables are standardized using the fitting block and supplied to Ridge regression. AE and DAE candidates match in architecture, optimization settings, seeds, epochs and Ridge penalties, differing only in input noise. Reconstruction uses environmental inputs alone; selection of the complete pipeline uses inner abundance RMSE. Compression may retain shared environmental structure while losing low-variance information relevant to abundance, motivating comparison of reconstruction and regression errors.

### 3.3 Reference models

Five references are the fitting-block mean, temperature-only ordinary least squares, multivariable Ridge, random forest, and PCA followed by Ridge (Table 1). The mean measures improvement over a constant estimate; temperature OLS isolates one physical covariate; Ridge, forest and PCA test linear prediction, nonlinear partitioning and linear compression. Ridge penalties range from 0.01 to 1000 in powers of ten. PCA retains two, four or six components, with the same input and latent standardization used for the autoencoder pipelines. The forest has 400 trees and 12 configurations spanning depth 3 or unrestricted, minimum leaf sizes 2, 5 or 10, and feature fractions 0.7 or 1.0. It retains raw input and target scales.

## 4. Chronological validation and performance measures

Three expanding outer folds use 59, 79 and 99 fitting observations, a one-observation gap, and 20 validation observations each. Their validation YearDay ranges are 459–592, 599–732 and 740–872. These are the final three blocks of the earlier five-fold pilot, whose earliest fitting blocks contained 19 and 39 observations. The formal comparison assesses estimation after at least 59 fitting observations.

Within each outer fitting block, three expanding inner folds each reserve ten validation observations and a one-row gap. Inner fitting counts are (28, 38, 48), (48, 58, 68) and (68, 78, 88), respectively. Every scaler, representation and regression model is fitted within the appropriate block. Epoch and hyperparameter selection averages RMSE over three seeds within each inner fold, then over folds. Selected pipelines are refitted from scratch using ten fixed seeds for outer evaluation; deterministic references require one fit.

Final selection repeats the inner procedure within all 120 development observations. The selected configurations are then frozen, refitted and evaluated on the historical holdout.

For observations $y_i$ and predictions $\widehat y_i$, RMSE is $[n^{-1}\sum_i(y_i-\widehat y_i)^2]^{1/2}$, MAE is $n^{-1}\sum_i|y_i-\widehat y_i|$, and $R^2=1-\sum_i(y_i-\widehat y_i)^2/\sum_i(y_i-\bar y)^2$. Negative $R^2$ indicates larger squared error than the evaluated block's observed mean. Errors are reported in $10^6$ cells mL⁻¹.

Main outer summaries average seed metrics within folds, then report the three-fold mean and sample standard deviation. Ensemble metrics score the equal-weight mean prediction of ten members. Seed variation measures optimization variability on fixed observations, while fold variation describes performance across the three time blocks.

To verify which settings minimize later-block error, the next code block reconstructs the inner selection scores from individual trial records, averaging seeds before time folds.
