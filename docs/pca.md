# External PCA

The `pca500` profile evaluates TabPFNv2.5, RealTabPFNv2.5 and TabICLv2 on the 36 released datasets with more than 2,000 features. Both Default and HPO use external PCA capped at 500 components, limited further by training rows and nonconstant columns. Imputation uses training-column means, with zero for entirely unobserved columns. Imputation and PCA are fitted within each training partition, including inner HPO folds.

Native model search choices remain available, including PFN preprocessing, SVD, polynomial features, checkpoints and regression target transformations. Default remains eligible for validation selection. The ordinary six-hour fixed-split and two-hour outer-fold selection budgets, 100 official attempts and 200-attempt trajectory collection remain unchanged.

Run one complete pair:

```bash
tabbiomed run --profile pca500 --model TabPFNv2.5 --dataset true_fluorescence --gpu
```

Prepare all 108 model/dataset pairs:

```bash
tabbiomed jobs prepare --profile pca500 --directory jobs/pca500 --gpu-partition gpu --gpu-type a5500
```

Choose partition and GPU names available on your cluster. Each task runs Default, HPO, all applicable outer folds and selected final fits in one 24-hour job with 10 CPUs, 128 GB RAM and one GPU. Preparing the plan submits nothing. Review the submission command, then submit explicitly:

```bash
tabbiomed jobs submit --directory jobs/pca500
tabbiomed jobs submit --directory jobs/pca500 --execute
```

PCA raw runs use `results/raw_pca500/<Model>/<Dataset>/run_<timestamp>/` by default, separate from Main runs in `results/raw/`. `--models` and `--datasets` select a smaller supported subset when preparing jobs.

After consolidating the corresponding Main evaluations, collect and consolidate PCA runs:

```bash
tabbiomed results manifest --raw results/raw_pca500 --output results/pca500_runs.json
tabbiomed results consolidate --manifest results/pca500_runs.json --output results/consolidated
tabbiomed results consolidate --manifest results/pca500_runs.json --output results/consolidated --execute
```

The exporter writes PCA evaluations under `<Model>/<Dataset>/pca_500/`, preserving Main results. Manifest collection retains this scope automatically.
