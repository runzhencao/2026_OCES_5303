## 5. Results

### 5.1 Later-block estimation within the development period

Ridge had the lowest mean outer RMSE, 1.009 × 10⁶ cells mL⁻¹, followed by random forest (1.021), residual MLP (1.028) and temperature OLS (1.045; Table 2). The shallow MLP averaged 1.120, exceeding Ridge in all three blocks by 0.009, 0.088 and 0.236. Increasing capacity with the residual architecture reduced mean RMSE by 0.091 relative to the shallow network. This improvement occurred in blocks 1 and 3; block 2 instead increased by 0.025 (Figure 2a).

Rankings depended on time. The residual MLP had the smallest mean individual error in block 1 (1.122), close to temperature OLS (1.128), whereas Ridge led blocks 2 and 3 (0.830 and 0.893). Mean outer R² was 0.076 for the residual MLP and 0.108 for Ridge; this average also reflects differences in response variance among blocks.

AE + Ridge improved on PCA + Ridge by 0.018 in mean RMSE, with fold differences −0.062, +0.056 and −0.049. DAE + Ridge averaged 1.146, compared with 1.135 for AE + Ridge. Denoising slightly reduced error in blocks 1 and 2 but increased it by 0.055 in block 3. Both compression pipelines had higher mean errors than Ridge and random forest.

Initialization mattered most visibly for the residual MLP: its ten seed-specific three-fold mean RMSEs ranged from 0.913 to 1.116 (Figure 2b). Equal-weight ensembling reduced its mean RMSE from 1.028 to 1.019; corresponding reductions were 1.120 to 1.111 for shallow MLP, 1.135 to 1.126 for AE, and 1.146 to 1.136 for DAE. Ensemble gains were modest relative to differences among time blocks. All outer predictions were nonnegative.

### 5.2 Capacity, training and compression diagnostics

Final inner selection chose a 193-parameter shallow MLP trained for 800 epochs and a 9,089-parameter residual MLP trained for 300 epochs. For the selected residual base configuration, extending training from 300 to 800 epochs reduced training RMSE from 0.675 to 0.389 but increased validation RMSE from 0.745 to 0.894, indicating overfitting over this interval (Figure 3b). The selected shallow configuration improved both errors across the sampled checkpoints. The chosen residual configuration combined weight decay 0.1 and dropout 0.2. These results favor selecting capacity, regularization and training duration jointly.

Six-dimensional representations gave the best final inner scores for PCA, AE and DAE (Figure 4a). The selected PCA achieved lower validation reconstruction MSE (0.105) than the selected AE (0.127), while its abundance RMSE was higher (0.909 versus 0.854). Reconstruction quality and abundance estimation therefore ranked these representations differently. The selected AE and DAE used the same 560-parameter architecture and 300 epochs, with Ridge penalties of 10 and 1, respectively. In the matched diagnostic at 300 epochs and penalty 10, denoising reduced mean RMSE for seven of twelve base configurations; differences ranged from −0.006 to +0.013 (Figure 4c). The effect of denoising was small and varied across configurations.

### 5.3 Historical holdout

The final 31 observations changed the ordering (Table 3). AE + Ridge had the lowest mean individual RMSE, 0.975, followed by DAE + Ridge (0.981) and PCA + Ridge (0.995). Shallow and residual MLP errors were 1.077 and 1.096, reversing their average development ranking; Ridge increased to 1.217. The AE and DAE ensembles achieved RMSEs of 0.959 and 0.961, with R² values of 0.234 and 0.230. All historical predictions were nonnegative.

Figure 5 shows overestimation of several low observations early in the block and underestimation of the peaks at YearDay 1035 and 1049. At YearDay 1035, observed abundance was 5.450, whereas AE and DAE ensemble estimates were 3.100 and 3.114, respectively. Their lower overall RMSE coexisted with substantial errors at individual sampling occasions.

## 6. Discussion and limitations

The larger residual MLP improved direct neural estimation in the outer comparison, while Ridge achieved the lowest mean error. The residual MLP's learning curve showed that longer training could reduce fitting error while worsening later-block estimation. At this sample size, model selection must balance capacity, regularization and training duration. The sparse candidate design evaluates combinations of these settings, and the shallow–residual comparison combines changes in parameter count and architecture.

Representation quality depended on the learning objective and evaluation period. PCA reconstructed environmental inputs more accurately than AE, whereas AE achieved lower abundance RMSE in final inner selection. AE and DAE also led the historical comparison but had higher outer errors than the best direct models. These changes in ranking favor evaluating representations by later-block abundance error. Both the matched diagnostic and the separately selected outer results showed inconsistent gains from denoising.

Temperature OLS approaching the best outer errors is consistent with a strong seasonal environmental signal, but its shared seasonality with other predictors limits mechanistic interpretation. Ward et al. (2017) investigated community composition; the present aggregate concentration cannot resolve taxon-specific responses or establish growth and loss rates.

The 151 observations come from one site, with seasonally covarying predictors and potential temporal dependence across the one-observation gaps. Repeated seeds describe optimization variability on this fixed record. The pilot informed the formal design, and Assignment 1 had already examined the historical holdout, making the comparisons exploratory. Unresolved nutrient scales further limit physical interpretation. Evaluation on independent years or sites under a prespecified protocol would strengthen assessment of transfer.

## 7. Conclusion

Across nine pipelines for contemporaneous coastal prokaryotic abundance estimation, Ridge achieved the lowest mean outer RMSE of 1.009 × 10⁶ cells mL⁻¹. The residual MLP improved on the shallow MLP, and the AE ensemble led the historical comparison at 0.959 × 10⁶ cells mL⁻¹. Denoising gave inconsistent improvements. Model rankings varied across time blocks and between reconstruction and abundance objectives, supporting chronological validation against simple references when selecting neural capacity and representations for this dataset.
