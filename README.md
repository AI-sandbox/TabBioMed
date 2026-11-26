<p align="center"><img src="docs/assets/logo.png" width="128" height="128" alt="TabBioMed logo"></p>

<h1 align="center">TabBioMed: A Large-Scale Benchmark for<br>Biomedical Tabular Learning</h1>

<p align="center">A living benchmark for biomedical tabular learning.</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.23779">📄 Read the paper on arXiv</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/TabBioMed/leaderboard">🏆 Live leaderboard</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/TabBioMed/TabBioMed">🤗 Datasets</a>
</p>

TabBioMed compares tabular models on 94 datasets across eight biomedical domains. It covers classification and regression, from clinical tables to high-dimensional omics data.

## Installation

From the repository root, install TabBioMed with `uv` and Python 3.12.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Install extras for the models you plan to run. For example, `uv pip install -e ".[trees,neural]"` installs the boosted-tree and neural model extras.

| Group | Models |
|---|---|
| Core | LinearModel, LassoModel, RidgeModel, ElasticNet, KNN, SVM, DecisionTree, RandomForest, MLP |
| `trees` | XGBoost, LightGBM, CatBoost |
| `neural` | RealMLP, TabM |
| `pfns` | TabPFNv2.5, RealTabPFNv2.5 |
| `tabicl`, `tabdpt`, `hyperfast`, `iltm` | TabICLv2, TabDPT, HyperFast, iLTM (one extra per model) |
| `kumo` | Kumo-Tabular-Small, Kumo-Tabular-Medium, and Kumo-Tabular (Large), in a separate GPU environment |
| `kernels` | NeoLsSVM |
| `xrfm` | xRFM, requiring Linux and CUDA |

TabPFN-Wide uses a different TabPFN version. Install it in its own environment.

```bash
uv venv --python 3.12 .venv-wide
uv pip install --python .venv-wide/bin/python -e ".[tabpfn-wide]"
```

## Data

Official benchmark evaluations use a pinned revision of the [TabBioMed dataset release](https://huggingface.co/datasets/TabBioMed/TabBioMed). Use `download` to fetch datasets and `verify` to check an existing copy against the release.

```text
tabbiomed data download [DATASET ...] [--data-root PATH]
tabbiomed data verify [DATASET ...] [--data-root PATH]
```

Omit the dataset names to download or verify the full benchmark. Run `tabbiomed data list` to see the names accepted by the CLI.

Datasets go in `data/processed` by default. If you set `--data-root PATH`, use the same path for downloading, verifying and running evaluations. The download and conversion cache lives in `~/.cache/tabbiomed` by default. Set `TBM_DATA_CACHE` to store it elsewhere.

The dataset guide also documents how the released tables were built from the original sources. See [reference curation](docs/datasets.md#reference-curation).

## Evaluation

Each run evaluates one model on a dataset's fixed split or all five stored outer folds. For Tuned results, the runner compares the default configuration and up to 100 sampled configurations using validation scores. iLTM uses its 25 recommended configurations. LinearModel and NeoLsSVM run with defaults only. The [evaluation guide](docs/evaluation.md#validation-and-refitting) explains refits, failed trials and the 25, 50, 100 and 200 trial comparisons.

### Quick check

To check that TabBioMed is working, try a CPU baseline on Pima Diabetes. The first command downloads the dataset, and the second evaluates `LinearModel`.

```bash
tabbiomed data download pima_diabetes
tabbiomed run --dataset pima_diabetes --model LinearModel --output results/raw
```

Look for the metrics in `report.json`, along with the test predictions and a run manifest, in the new `run_<timestamp>/` directory under `results/raw/LinearModel/pima_diabetes/`.

### Slurm

Slurm gives each model and dataset pair 24 hours for the full evaluation, including all outer folds and final fits. TabBioMed enforces a separate time budget for hyperparameter optimization (HPO) in both local and Slurm runs. This budget includes validation of the default configuration and is capped at 6 hours for a fixed split or 2 hours per outer fold for cross-validation.

Use a partition available on your cluster. `jobs prepare` writes the job files, `jobs submit` previews the submission, and `--execute` submits the jobs.

```bash
tabbiomed jobs prepare --directory jobs/pima --datasets pima_diabetes --models LinearModel --partition long
tabbiomed jobs submit --directory jobs/pima
tabbiomed jobs submit --directory jobs/pima --execute
```

Select TabPFN-Wide explicitly and use its environment's Python when preparing its jobs. See the [Slurm guide](docs/evaluation.md#slurm) for GPU partitions and array settings.

## Consolidate your runs

To use your own runs as input for analysis, make a manifest, preview the consolidation and write the exports.

```bash
tabbiomed results manifest --raw results/raw --output results/runs.json
tabbiomed results consolidate --manifest results/runs.json --output results/consolidated
tabbiomed results consolidate --manifest results/runs.json --output results/consolidated --execute
```

The export includes metrics, timings, test predictions and labels, and HPO histories for tunable models. See the [consolidation guide](docs/evaluation.md#consolidation) for the file layout and how to select a run when you have multiple runs for the same pair.

## Analysis and plots

Install the optional analysis dependencies.

```bash
uv pip install -e ".[analysis]"
```

Use the commands below to compute Elo rankings, generate the PCA comparison figures, or reproduce the paper's figures and tables.

```bash
tabbiomed analyze elo --output outputs/elo
tabbiomed analyze pca --output outputs/pca
tabbiomed analyze reproduce --profile paper --output outputs/paper
```

These commands download the pinned reference results to `data/results/` by default. To analyze your own exports, pass `--results results/consolidated` or set `TBM_RESULTS`. See the [analysis guide](docs/analysis.md) for individual plots and comparisons.

## Add a model

To benchmark your own model, implement and register it in an importable Python module. Pass `--model-module` when running evaluations, preparing jobs or consolidating results. The [model guide](docs/models.md) covers the interface, search spaces, supported tasks and GPU settings.

## Development

Run the tests from the repository root.

```bash
python -m unittest discover -s tests -v
```

The suite uses small CPU models to test the local workflow, including model registration, evaluation and consolidation.

See [Contributing](CONTRIBUTING.md) to report a bug or suggest a change.

## Citation and license

If you use TabBioMed, please cite:

```bibtex
@inproceedings{tabbiomed2026,
  title     = {TabBioMed: A Large-Scale Benchmark for Biomedical Tabular Learning},
  author    = {Mateo-Bernad{\'o}, Pau and Polapragada, Saivenkata Nagavyjayanthi and Bonet, David and Arbiol Rakuljic, Pol and Mogas Pladevall, Laia and Comajoan Cara, Mar{\c{c}}al and Gonzalez-Ferrer, Jesus and Abante, Jordi and Mas Montserrat, Daniel and Haussler, David and Ioannidis, Alexander G.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

If you use individual datasets, cite the original authors and follow the source licenses. The TabBioMed code uses the [Apache-2.0 license](LICENSE). Model checkpoints come with their own licenses.
