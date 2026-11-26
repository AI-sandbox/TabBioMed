"""Kumo-Tabular inference with the checkpoint and ensemble settings used by TabArena."""

import math

import numpy as np
import pandas as pd
import torch

from ..base_model import BaseModel


HF_REVISION = "3c3e10bbdb590ace29e7026847f92db3c603096d"


def _feature_frame(X):
    frame = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])], copy=False)
    if X.dtype.kind == "O":
        frame = frame.infer_objects(copy=False)
    # Torch indexing supports signed integers for wide integer columns.
    casts = {}
    for name, dtype in frame.dtypes.items():
        if dtype.kind == "u" and dtype.itemsize > 1:
            fits = dtype.itemsize < 8 or int(frame[name].max()) <= np.iinfo(np.int64).max
            casts[name] = "int64" if fits else "float64"
    return frame.astype(casts) if casts else frame


def _context_index(n_rows, members, limit, seed):
    if limit is None or n_rows <= limit:
        return None
    generator = torch.Generator().manual_seed(seed)
    rows = members * limit
    permutations = [torch.randperm(n_rows, generator=generator) for _ in range(math.ceil(rows / n_rows))]
    return torch.cat(permutations)[:rows].reshape(members, limit)


def _available_memory(device):
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    limit = total * torch.cuda.get_per_process_memory_fraction(device) - allocated
    return max(0, int(min(free + reusable, limit)))


class KumoTabular(BaseModel):
    size = "large"
    members = 16

    @classmethod
    def get_default_params(cls):
        return {"num_estimators": cls.members, "max_context_size": 200_000}

    @classmethod
    def needs_missing_value_imputation(cls):
        return False

    @classmethod
    def needs_cat_encoding(cls):
        return False

    @classmethod
    def needs_scaling(cls):
        return False

    @classmethod
    def supports_native_multioutput(cls):
        return False

    def fit(self, X, y, X_val=None, y_val=None):
        import sdm
        from sdm.models._huggingface import download_checkpoint
        from sdm.models.kumo.tabular import estimate_fit_batch_size

        self.device = torch.device("cpu")
        if self.args.use_gpu:
            self.device = torch.device("cuda", (self.args.gpu_ids or [0])[0])
        self.regression = self.args.objective in ("regression", "multiregression")
        task = "regression" if self.regression else "classification"
        filename = "regressor.pt" if self.regression else "classifier.pt"
        checkpoint = download_checkpoint(
            repo_id="nvidia/Kumo-Tabular", filename=f"{self.size}/{filename}",
            revision=HF_REVISION, config_filename="config.json",
        )
        self.model = sdm.models.KumoTabular(task=task, size=self.size, pretrained=False, device="meta")
        weights = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.models[task].load_state_dict(weights, assign=True)
        self.model.eval()

        frame = _feature_frame(X)
        categorical = {str(i): "categorical" for i in self.args.cat_idx or []}
        self.stypes = sdm.infer_stypes(frame, overrides=categorical, _low_cardinality="infer")
        x = sdm.TableTensor.from_pandas(df=frame, stypes=self.stypes)
        y = sdm.TableTensor.from_pandas(
            df=pd.DataFrame({"target": np.asarray(y).reshape(-1)}),
            stypes={"target": "numerical" if self.regression else "categorical"},
        )
        self.y_metadata = y[:0].cpu()
        if not self.regression:
            self.y_metadata = sdm.TableTensor(
                columns=y.columns,
                categorical=sdm.CategoricalTensor(
                    code=self.y_metadata.categorical.code, categories=(torch.arange(self.args.num_classes),),
                ),
            )

        members = self.params["num_estimators"]
        index = _context_index(len(x), members, self.params["max_context_size"], self.args.seed)
        self.subsampled = index is not None
        if self.subsampled:
            x = x[index.flatten()].unflatten(0, index.shape)
            y = y[index.flatten()].unflatten(0, index.shape)
        x, y = x.to(self.device), y.to(self.device)
        self.estimator_batch_size = 1
        if self.device.type == "cuda":
            self.estimator_batch_size = estimate_fit_batch_size(
                model=self.model, x=x, y=self.y_metadata, num_estimators=members,
                memory_budget=_available_memory(self.device),
            )
        with torch.inference_mode(), self._autocast():
            self.model.fit(
                x=x, y=y, num_estimators=None if self.subsampled else members,
                estimator_batch_size=self.estimator_batch_size,
                generator=torch.Generator(self.device).manual_seed(self.args.seed),
            )
        # SDM streams the attention cache from host memory but applies the recipe on the GPU.
        self.model._cache = self.model._cache.cpu()
        recipe = self.model._cache["recipe_execution"].recipe
        for processor in (recipe.features, recipe.target, recipe.output):
            processor.to(self.device)
        return [], []

    def _autocast(self):
        return torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda")

    def _query_batch_size(self, X):
        import sdm
        from sdm._memory import chunk_memory_limit
        from sdm.models.kumo.tabular import estimate_predict_batch_size

        if self.device.type != "cuda":
            return len(X)
        cache = self.model._cache
        staging_bytes = 2 * max(cache[i].size() for i in range(cache["num_batches"]))
        budget = min(chunk_memory_limit(self.device), max(0, _available_memory(self.device) - staging_bytes) // 2)
        return estimate_predict_batch_size(
            model=self.model, x=sdm.TableTensor.from_tensor(torch.empty(0, X.shape[1])),
            y=self.y_metadata, num_estimators=self.params["num_estimators"],
            estimator_batch_size=self.estimator_batch_size, memory_budget=2 * budget,
        )

    def _forward(self, X):
        import sdm

        query = sdm.TableTensor.from_pandas(df=_feature_frame(X), stypes=self.stypes, device=self.device)
        if self.subsampled:
            query = query.expand(self.params["num_estimators"], *query.size())
        with torch.inference_mode(), self._autocast():
            output = self.model.predict(query)
            values = output.numerical.float()
            if self.regression:
                return values.mean(dim=-1).cpu().numpy()
            labels = np.asarray(output.columns[sdm.Stype.numerical], dtype=np.intp)
            return self._align_class_probabilities(values.cpu().numpy(), labels)

    def _predict_batch(self, X):
        try:
            return self._forward(X)
        except torch.OutOfMemoryError:
            if self.device.type != "cuda" or len(X) <= 512:
                raise
        torch.cuda.empty_cache()
        mid = len(X) // 2
        return np.concatenate([self._predict_batch(X[:mid]), self._predict_batch(X[mid:])])

    def _predict_values(self, X):
        batch_size = self._query_batch_size(X)
        return np.concatenate([self._predict_batch(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])

    def predict(self, X):
        if not hasattr(self, "_output_models_") and self.args.objective in ("regression", "multiregression"):
            self.predictions = self._predict_values(X)
            return self.predictions
        return super().predict(X)

    def predict_proba(self, X):
        return self._predict_values(X)


class KumoTabularMedium(KumoTabular):
    size = "medium"
    members = 8


class KumoTabularSmall(KumoTabular):
    size = "small"
    members = 8
