## 8. Computation and reproducibility

The formal experiment used two RTX 4090 D GPUs with 24 GB memory each, Python 3.11.15 and PyTorch 2.12.1 with CUDA 12.6. Training used float32, fixed seeds and deterministic settings, with TF32 and mixed precision disabled. Neural trials ran on GPU and reference models on CPU. Search, refitting and evaluation took 916.99 s (15.3 min).

The notebook loads the course data remotely, with a frozen offline copy and the source, configuration and results stored in metadata. Default execution recomputes selection scores, prediction metrics, the scale-sensitivity table and figures, then fits four CPU examples with `PyTorch`. Tables 1–3 describe the archived GPU experiment; Table 4 reports the separate CPU sensitivity experiment. Running every sensitivity treatment on CPU isolates scale changes from device-specific dropout trajectories. The optional full-search mode writes separate results.

[Public repository](https://github.com/runzhencao/2026_OCES_5303) · [Open this report in Colab](https://colab.research.google.com/github/runzhencao/2026_OCES_5303/blob/submission-v1/Assignment_2/ass2_oces5303_21073359.ipynb). Select a Python CPU runtime and **Runtime → Run all**. The default run needs no private files or Drive mount. The public `submission-v1` release contains both assignments, their readable sources and reports. `REPEAT_FULL_STUDY` and `REPEAT_SCALE_CHECK` enable independent training runs; both default to `False`. The final output records the current runtime versions.

**AI assistance:** AI tools assisted analysis design, code preparation, verification, figure preparation, report organization, English revision and Chinese translation.
