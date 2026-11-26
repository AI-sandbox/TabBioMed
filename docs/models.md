# Models

Models implement `BaseModel(params, args)` and return NumPy predictions through
`predict` and classification probabilities through `predict_proba`. Define
`get_default_params`, `needs_missing_value_imputation`, `needs_cat_encoding`, `needs_scaling`
and `supports_native_multioutput`. Override `uses_validation_during_fit` when
training uses validation for early stopping. All preprocessing is fitted on the
training partition by the benchmark or the model itself.

`get_base_params(args)` defines base wrapper settings shared by Default and HPO.
`get_default_params()` supplies the Default configuration's overrides, and
`required_parameters` declares required input keys. The runner resolves and records
the complete configuration before construction. Constructors use `params["name"]`
and reject missing or unrecognized keys. Declare additional optional keys in
`optional_parameters`. Wrappers forwarding native backend options can extend
`get_supported_parameters(args)`. For direct use, call
`Model(Model.resolve_parameters(parameters, args), args)`.

Register a model in an importable module:

```python
from tabbiomed.models import register_model
from my_models.example import ExampleModel

register_model(
    "ExampleModel",
    ExampleModel,
    search_space={"regularization": {"type": "float", "low": 0.001, "high": 1.0, "log": True}},
    tasks=("binary", "classification", "regression"),
    gpu=False,
)
```

Pass that module with `--model-module my_models.registration` when running or
preparing jobs. The same module must be importable on compute nodes. Set
`hpo_enabled=False` and omit `search_space` for a Default-only model. Set `gpu=True` for GPU job resources. Model names
and aliases are unique. The benchmark records the parameters and predictions
using the same output schema for built-in and registered models.

Set `requires_separate_env=True` when a model needs its own Python environment.
Select it explicitly with `--models` and pass that environment's interpreter
with `--python` when preparing Slurm jobs.

TabPFN and RealTabPFN download their explicitly selected v2.5 checkpoints from
`Prior-Labs/tabpfn_2_5`. TabPFN-Wide downloads its v0.3.0 author release and the
explicit v2 base checkpoint. Source revisions and weight identities are pinned in
`tabbiomed/configs/model_checkpoints.json`. Set `TABBIOMED_CHECKPOINTS` to choose a checkpoint
cache. Supply `custom_model_dir` to use an existing local checkpoint directory.
HyperFast uses its author-distributed checkpoint through the same cache. TabDPT,
TabICLv2 and iLTM use their libraries' checkpoint loaders.
Checkpoint access and licenses remain governed by the respective authors.

TabPFN-Wide requires a separate environment because its TabPFN dependency differs
from the TabPFNv2.5 and RealTabPFNv2.5 wrappers. Use its optional installation group
in that environment, then select its Python executable when preparing Slurm jobs.

The opt-in [`pca500` profile](pca.md) extends the three PCA models' registered search spaces with fixed external PCA settings. Main evaluations keep external PCA disabled.

### TabPFN-3.5

Install `uv pip install -e '.[tabpfn35]'` in a separate environment.
Select `--model TabPFN-3.5` for a run, or `--models TabPFN-3.5` with that
environment's `--python` when preparing Slurm jobs. TabPFN-3.5 runs without HPO.
It uses the pinned v3.5 checkpoint, the library's ensemble and preprocessing
defaults, and the benchmark seed. The final fit uses the combined training and
validation rows. Multi-output regression fits one regressor per target.
