"""TabM ensemble with training-only numerical preprocessing.

Heads share parameters and their predictions are averaged at inference. The
pipeline imputes missing values and provides integer categorical codes, TabM
encodes categorical features internally. Each fit drops numerical columns that
are constant on its training rows and applies that mask to validation and test."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.special
from sklearn.preprocessing import QuantileTransformer
from typing import Dict, Any, Tuple, Union, Optional, List
from loguru import logger

try:
    import tabm
except ImportError:
    raise ImportError(
        "tabm library is required. Install the neural optional dependency group (e.g., uv pip install -e '.[neural]')."
    )

try:
    import rtdl_num_embeddings
except ImportError:
    raise ImportError(
        "TabM requires rtdl-num-embeddings. Install the neural optional dependency group (e.g., uv pip install -e '.[neural]')."
    )

from tabbiomed.models.base_model import BaseModel, ModelArgs


def _tabm_n_unique_non_nan(col: np.ndarray) -> int:
    """Distinct values in a column, ignoring NaNs (float), object dtype uses plain unique."""
    c = np.asarray(col)
    if c.dtype == object:
        return len(np.unique(c))
    c = c.astype(np.float64, copy=False)
    finite = c[~np.isnan(c)]
    return len(np.unique(finite))


class TabM(BaseModel):
    """
    TabM (Tabular Model) implementation.

    TabM efficiently represents an ensemble of k MLPs through weight sharing
    and parallel training. It produces k predictions per sample, which are
    averaged during inference.
    """

    @classmethod
    def get_base_params(cls, args):
        return {
            "arch_type": "tabm-mini",
            "tabm_k": 16,
            "n_blocks": 3,
            "d_block": 512,
            "d_embedding": 16,
            "dropout_choice": 1.0,
            "lr": 0.002,
            "weight_decay_choice": 1.0,
            "patience": 16,
            "n_epochs": 1000000000,
            "batch_size": "auto",
            "num_emb_type": "pwl",
            "num_emb_n_bins": 16,
            "gradient_clipping_norm": 1.0,
            "amp": True,
            "compile_model": False,
            "bool_to_cat": True,
            "average_logits": False,
            "activation": "ReLU",
            "random_state": args.seed if hasattr(args, "seed") else None,
            "dropout": 0.1,
            "weight_decay": 0.0003,
            "train_chunk_size": 32,
            "eval_batch_size": 64,
            "share_training_batches": True,
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)

        if hasattr(args, "seed"):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        if args.use_gpu and torch.cuda.is_available():
            self.device = torch.device(
                f"cuda:{args.gpu_ids[0]}" if args.gpu_ids else "cuda"
            )
        else:
            self.device = torch.device("cpu")

        self.arch_type = params["arch_type"]
        self.tabm_k = params["tabm_k"]
        self.n_blocks = params["n_blocks"]
        self.d_block = params["d_block"]
        self.d_embedding = params["d_embedding"]

        self.num_outputs = args.num_outputs if hasattr(args, "num_outputs") else 1

        dropout_choice = params["dropout_choice"]
        if dropout_choice == 0.0:
            self.dropout = 0.0
        else:
            self.dropout = params["dropout"]

        self.lr = params["lr"]

        weight_decay_choice = params["weight_decay_choice"]
        if weight_decay_choice == 0.0:
            self.weight_decay = 0.0
        else:
            self.weight_decay = params["weight_decay"]
        self.patience = params["patience"]
        self.n_epochs = params["n_epochs"]
        self.batch_size = params["batch_size"]

        self.train_chunk_size = int(params["train_chunk_size"])
        self.eval_batch_size = int(params["eval_batch_size"])
        if self.train_chunk_size < 1 or self.eval_batch_size < 1:
            raise ValueError(
                "TabM training and evaluation chunk sizes must be positive"
            )
        self.train_chunk_size_ = self.train_chunk_size
        self.eval_batch_size_ = self.eval_batch_size
        self.num_emb_type = params["num_emb_type"]
        self.num_emb_n_bins = params["num_emb_n_bins"]
        self.effective_num_emb_n_bins_ = None
        self.gradient_clipping_norm = params["gradient_clipping_norm"]
        self.share_training_batches = bool(params["share_training_batches"])
        self.amp = params["amp"]
        self.compile_model = params["compile_model"]
        self.bool_to_cat = params["bool_to_cat"]
        self.average_logits = params["average_logits"]
        self.activation = params["activation"]
        self.random_state = params["random_state"]

        self._cat_idx_input = tuple(args.cat_idx or ())
        self.cat_idx = list(self._cat_idx_input)
        self._binary_cat_idx = set()
        self.num_idx = [i for i in range(args.num_features) if i not in self.cat_idx]
        self.n_num_features = len(self.num_idx)
        self.n_cat_features = len(self.cat_idx)

        self.cat_cardinalities = None
        self.num_embeddings_module = None
        self.label_mean = None
        self.label_std = None
        self.quantile_transformer = None

        self.model = None
        self.optimizer = None

        if self.amp and self.device.type == "cuda":
            self.amp_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16 if torch.cuda.is_available() else None
            )
            self.grad_scaler = (
                torch.cuda.amp.GradScaler() if self.amp_dtype == torch.float16 else None
            )
        else:
            self.amp_dtype = None
            self.grad_scaler = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

        self._cols_keep_orig: Optional[np.ndarray] = None

    def _drop_constant_numerical_train_val(
        self,
        X: np.ndarray,
        X_val: Union[None, np.ndarray],
    ) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
        """Drop training-constant numerical columns and apply that mask to validation.

        Remap categorical and numerical column indices for the reduced matrix."""
        self._cols_keep_orig = None
        n_feat = X.shape[1]
        cat_set = set(self.cat_idx) if self.cat_idx else set()
        num_indices = [j for j in range(n_feat) if j not in cat_set]

        to_drop: List[int] = []
        for j in num_indices:
            if _tabm_n_unique_non_nan(X[:, j]) < 2:
                to_drop.append(j)
                logger.warning(
                    f"TabM: filtering numerical column {j} (<2 distinct values on training data)"
                )

        if not to_drop:
            return X, X_val

        drop_set = set(to_drop)
        cols_to_keep = [i for i in range(n_feat) if i not in drop_set]
        if not cols_to_keep:
            logger.error(
                "TabM: all features would be removed by constant-numerical filter, skipping filter."
            )
            return X, X_val

        self._cols_keep_orig = np.asarray(cols_to_keep, dtype=np.int64)
        old_to_new = {old: new for new, old in enumerate(cols_to_keep)}
        self.cat_idx = [old_to_new[c] for c in self.cat_idx] if self.cat_idx else []
        self.num_idx = [
            i for i in range(len(cols_to_keep)) if i not in set(self.cat_idx)
        ]
        self.n_num_features = len(self.num_idx)
        self.n_cat_features = len(self.cat_idx)
        logger.warning(
            f"TabM: dropped {len(to_drop)} constant numerical column(s), "
            f"{len(cols_to_keep)} features remain."
        )

        X_new = X[:, cols_to_keep]
        X_val_new = X_val[:, cols_to_keep] if X_val is not None else None
        return X_new, X_val_new

    def _promote_binary_features(self, X: np.ndarray) -> None:
        """Use training-observed 0/1 numerical columns as categorical features."""
        self._binary_cat_idx = set()
        if not self.bool_to_cat:
            return
        for idx in self.num_idx:
            column = np.asarray(X[:, idx], dtype=np.float64)
            values = np.unique(column[~np.isnan(column)])
            if np.array_equal(values, [0.0, 1.0]):
                self._binary_cat_idx.add(idx)
        self.cat_idx.extend(idx for idx in self.num_idx if idx in self._binary_cat_idx)
        self.num_idx = [idx for idx in self.num_idx if idx not in self._binary_cat_idx]
        self.n_cat_features = len(self.cat_idx)
        self.n_num_features = len(self.num_idx)

    def _apply_learned_column_mask(self, X: np.ndarray) -> np.ndarray:
        """For ``predict`` only: ``X`` is full pipeline width, select columns kept in ``fit``."""
        ck = getattr(self, "_cols_keep_orig", None)
        if ck is None:
            return X
        return np.asarray(X)[:, ck]

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return True

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        return False

    @classmethod
    def uses_validation_during_fit(cls) -> bool:
        return True

    @classmethod
    def get_cat_encoding_strategy(cls) -> str:
        """TabM uses label encoding (integer encoding) for categoricals."""
        return "label"

    @classmethod
    def get_mv_imputation_strategy(cls) -> str:
        """TabM uses mean imputation for numerical features."""
        return "mean"

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        """Default hyperparameters for TabM based on reference implementation and YAML search space."""
        return {
            "arch_type": "tabm-mini",
            "tabm_k": 32,
            "share_training_batches": False,
            "batch_size": "auto",
            "train_chunk_size": 32,
            "patience": 16,
            "n_epochs": 1_000_000_000,
            "weight_decay_choice": 1.0,
            "weight_decay": 3e-4,
            "lr": 2e-3,
            "d_embedding": 16,
            "d_block": 512,
            "dropout": 0.1,
            "dropout_choice": 1.0,
            "amp": False,
            "n_blocks": 3,
            "num_emb_n_bins": 16,
            "eval_batch_size": 64,
            "gradient_clipping_norm": 1.0,
            "compile_model": False,
            "bool_to_cat": True,
            "average_logits": False,
            "num_emb_type": "pwl",
        }

    def _get_auto_batch_size(self, n_samples: int) -> int:
        """Auto batch size heuristic from TabM paper."""
        if n_samples < 2_800:
            return 32
        if n_samples < 4_500:
            return 64
        if n_samples < 6_400:
            return 128
        if n_samples < 32_000:
            return 256
        if n_samples < 108_000:
            return 512
        return 1024

    def _compute_cat_cardinalities(self, X: np.ndarray) -> List[int]:
        """Compute cardinalities for categorical features.

        Accounts for unknown categories (-1) by adding an extra category.
        If max category is N, we reserve N+1 for unknown categories, so cardinality = N+2.
        """
        if not self.cat_idx:
            return []

        cardinalities = []
        for idx in self.cat_idx:
            unique_vals = np.unique(X[:, idx])

            valid_vals = unique_vals[unique_vals >= 0]
            if len(valid_vals) > 0:
                max_val = int(valid_vals.max())

                cardinality = max_val + 2
            else:
                cardinality = 1
            cardinalities.append(cardinality)

        return cardinalities

    def _create_num_embeddings(self, X_num_tensor: Optional[torch.Tensor] = None):
        """Create numerical feature embeddings based on configuration."""
        if not rtdl_num_embeddings:
            return None

        if self.num_emb_type == "pwl":
            if X_num_tensor is None:
                raise ValueError(
                    "X_num_tensor is required for piecewise linear embeddings"
                )
            if len(X_num_tensor) < 3:
                raise ValueError(
                    "TabM piecewise-linear embeddings require at least 3 training rows"
                )
            self.effective_num_emb_n_bins_ = min(
                self.num_emb_n_bins, len(X_num_tensor) - 1
            )
            logger.info(
                "TabM numerical bins: requested={}, effective={}, training rows={}",
                self.num_emb_n_bins,
                self.effective_num_emb_n_bins_,
                len(X_num_tensor),
            )
            bins = rtdl_num_embeddings.compute_bins(
                X_num_tensor, n_bins=self.effective_num_emb_n_bins_
            )
            return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins, d_embedding=self.d_embedding, activation=False, version="B"
            )
        elif self.num_emb_type == "periodic":
            return rtdl_num_embeddings.PeriodicEmbeddings(
                self.n_num_features, lite=False
            )
        elif self.num_emb_type == "linear_relu":
            return rtdl_num_embeddings.LinearReLUEmbeddings(self.n_num_features)
        return None

    def _apply_quantile_transform(
        self, X: np.ndarray, is_train: bool = False
    ) -> np.ndarray:
        """
        Apply QuantileTransformer to numerical features only (skips binary columns).

        Args:
            X: Input features (imputed, label-encoded, but NOT scaled)
            is_train: Whether this is training data (for fitting the transformer)

        Returns:
            X with numerical features transformed using QuantileTransformer
        """
        if self.n_num_features == 0:
            return X

        if not is_train and hasattr(self, "num_idx_train_"):
            if (
                hasattr(self, "n_features_train_")
                and self.n_features_train_ is not None
            ):
                if X.shape[1] != self.n_features_train_:
                    raise ValueError(
                        f"TabM: Validation data has {X.shape[1]} total features "
                        f"but training had {self.n_features_train_} total features. "
                        f"This mismatch can occur when feature counts differ between train/val after imputation. "
                        f"Training num_idx_train_ expects {len(self.num_idx_train_)} numerical columns."
                    )

            if (
                hasattr(self, "n_num_features_train_")
                and self.n_num_features_train_ is not None
            ):
                try:
                    X_num_val = X[:, self.num_idx_train_].astype(np.float32)
                except IndexError as e:
                    raise ValueError(
                        f"TabM: Cannot extract numerical features using training indices. "
                        f"Validation X has {X.shape[1]} features, but training indices require at least "
                        f"{max(self.num_idx_train_) + 1} features. This mismatch can occur when feature counts "
                        f"differ between train/val after imputation."
                    ) from e

                if X_num_val.shape[1] != self.n_num_features_train_:
                    raise ValueError(
                        f"TabM: Validation data has {X_num_val.shape[1]} numerical columns "
                        f"after using training indices, but training had {self.n_num_features_train_} numerical columns. "
                        f"This mismatch can occur when feature counts differ between train/val after imputation. "
                        f"Original num_features: {getattr(self, 'n_num_features_original_', 'unknown')}, "
                        f"Training filtered to: {self.n_num_features_train_}"
                    )

            num_idx_to_use = self.num_idx_train_
        else:
            num_idx_to_use = self.num_idx

        X_num = X[:, num_idx_to_use].astype(np.float32)

        if is_train:
            binary_col_indices = []
            for col_idx in range(X_num.shape[1]):
                try:
                    unique_vals = np.unique(
                        X_num[:, col_idx][~np.isnan(X_num[:, col_idx])]
                    )
                    if len(unique_vals) <= 2:
                        is_binary = all(
                            np.isclose(val, 0, atol=1e-6)
                            or np.isclose(val, 1, atol=1e-6)
                            for val in unique_vals
                        )
                        if is_binary:
                            binary_col_indices.append(col_idx)
                except (TypeError, ValueError):
                    continue
            self.num_cols_to_transform_train_ = [
                i for i in range(X_num.shape[1]) if i not in binary_col_indices
            ]
            self.quantile_transformer = None

        num_cols_to_transform = self.num_cols_to_transform_train_

        if not num_cols_to_transform:
            return X

        if not is_train:
            if max(num_cols_to_transform) >= X_num.shape[1]:
                raise ValueError(
                    f"TabM: Validation data has {X_num.shape[1]} numerical columns "
                    f"but training used column indices up to {max(num_cols_to_transform)}. "
                    f"This mismatch can occur when feature counts differ between train/val after imputation."
                )

        X_num_to_transform = X_num[:, num_cols_to_transform]

        if is_train:
            noise = (
                np.random.default_rng(self.random_state or 0)
                .normal(0.0, 1e-5, X_num_to_transform.shape)
                .astype(X_num_to_transform.dtype)
            )

            n_samples = len(X_num_to_transform)
            n_quantiles = max(min(n_samples // 30, 1000), 10)

            self.quantile_transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                subsample=10**9,
                random_state=self.random_state,
            )

            self.quantile_transformer.fit(X_num_to_transform + noise)

        if self.quantile_transformer is not None:
            X_num_transformed = X_num.copy()
            X_num_transformed[:, num_cols_to_transform] = (
                self.quantile_transformer.transform(X_num_to_transform)
            )
        else:
            X_num_transformed = X_num

        X_transformed = X.astype(float if X.dtype.kind in "biu" else X.dtype, copy=True)

        X_transformed[:, num_idx_to_use] = X_num_transformed

        return X_transformed

    def _prepare_data(
        self, X: np.ndarray
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare CPU tensors, only individual execution chunks move to the model device.

        Args:
            X: Preprocessed features (imputed, label-encoded, QuantileTransformer applied)

        Returns:
            X_num: Numerical features tensor (or None)
            X_cat: Categorical features tensor (or None)
        """

        if self.n_num_features > 0:
            X_num_array = X[:, self.num_idx].astype(np.float32)
            X_num = torch.from_numpy(X_num_array)
        else:
            X_num = None

        if self.n_cat_features > 0:
            X_cat_array = np.empty((len(X), self.n_cat_features), dtype=np.int64)
            for i, idx in enumerate(self.cat_idx):
                unknown = self.cat_cardinalities[i] - 1
                if idx in self._binary_cat_idx:
                    values = np.asarray(X[:, idx], dtype=np.float64)
                    X_cat_array[:, i] = np.where(
                        values == 0, 0, np.where(values == 1, 1, unknown)
                    )
                else:
                    codes = np.asarray(X[:, idx], dtype=np.int64)
                    X_cat_array[:, i] = np.where(codes < 0, unknown, codes)

            X_cat = torch.from_numpy(X_cat_array)
        else:
            X_cat = None

        return X_num, X_cat

    def _create_model(self):
        """
        Create the TabM model using the official tabm library.
        """

        if self.args.objective == "regression":
            d_out = 1
        elif self.args.objective == "multiregression":
            if hasattr(self, "num_outputs") and self.num_outputs:
                d_out = self.num_outputs
            else:
                logger.warning(
                    "TabM: num_outputs not specified for multiregression, defaulting to 1 output"
                )
                d_out = 1
        elif self.args.objective == "binary":
            d_out = 2
        elif self.args.objective == "classification":
            d_out = (
                self.args.num_classes
                if hasattr(self.args, "num_classes") and self.args.num_classes
                else 2
            )
        else:
            raise ValueError(f"Unknown objective: {self.args.objective}")

        model = tabm.TabM.make(
            n_num_features=self.n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=d_out,
            num_embeddings=self.num_embeddings_module,
            k=self.tabm_k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            arch_type=self.arch_type,
            activation=self.activation,
        ).to(self.device)

        if self.compile_model:
            model = torch.compile(model)

        return model

    def _coerce_multiregression_y_to_2d(self, y: np.ndarray, n_rows: int) -> np.ndarray:
        """
        Ensure multiregression ``y`` is ``(n_rows, num_outputs)``.

        The trainer sometimes supplies a length ``n_rows * num_outputs`` vector instead of
        a 2D matrix. With per-head batching, training uses ``y_tensor[flat_idx]`` of shape
        ``(batch_size * k, num_outputs)``.
        """
        y = np.asarray(y, dtype=np.float64)
        if self.args.objective != "multiregression":
            return y
        d = int(self.num_outputs) if self.num_outputs else 1
        if d <= 1:
            return y
        if y.ndim == 2:
            if y.shape[0] != n_rows:
                raise ValueError(
                    f"TabM multiregression: y has {y.shape[0]} rows but data has {n_rows} samples"
                )
            if y.shape[1] != d:
                raise ValueError(
                    f"TabM multiregression: y has width {y.shape[1]} but num_outputs={d}"
                )
            return y
        if y.ndim == 1 and y.shape[0] == n_rows * d:
            return y.reshape(n_rows, d)
        if y.ndim == 1:
            raise ValueError(
                f"TabM multiregression: expected y of shape ({n_rows}, {d}) or ({n_rows * d},), "
                f"got {y.shape}"
            )
        raise ValueError(f"TabM multiregression: invalid y ndim/shape {y.shape}")

    def _compute_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, is_regression: bool
    ) -> torch.Tensor:
        """
        Compute loss for TabM predictions.

        Controlled by ``self.share_training_batches``:

        - **True:** one row index batch for all heads, ``y_true`` is ``(B,)`` or ``(B, d_out)``
          and is repeated ``k`` times to match ``y_pred_flat`` of shape ``(B * k, …)``.

        - **False:** independent indices per head, multiregression training uses
          ``y_true`` ``(B * k, d_out)``, scalar regression ``(B, k)``, classification ``(B, k)``
          or ``(B * k,)``. Validation still passes ``(B,)`` / ``(B, d_out)`` with repeat.
        """
        y_pred_flat = y_pred.flatten(0, 1)
        d_out = int(y_pred_flat.shape[1])
        n_flat = y_pred_flat.shape[0]
        k = self.tabm_k
        shared = self.share_training_batches

        if is_regression:
            if shared:
                if y_true.dim() == 1:
                    if d_out != 1:
                        raise ValueError(
                            f"TabM: 1D targets imply d_out=1, got d_out={d_out}"
                        )
                    if y_true.shape[0] * k != n_flat:
                        raise ValueError(
                            f"TabM (shared batches): expected len(y) * k == {n_flat}, "
                            f"got len={y_true.shape[0]}, k={k}"
                        )
                    y_rep = y_true.repeat_interleave(k)
                    return nn.functional.mse_loss(y_pred_flat.squeeze(-1), y_rep)
                if y_true.dim() == 2:
                    if y_true.shape[1] != d_out:
                        raise ValueError(
                            f"TabM: target width {y_true.shape[1]} != d_out {d_out}"
                        )
                    if y_true.shape[0] * k == n_flat:
                        y_rep = y_true.repeat_interleave(k, dim=0)
                    elif y_true.shape[0] == n_flat:
                        y_rep = y_true
                    else:
                        raise ValueError(
                            f"TabM (shared batches): y rows {y_true.shape[0]} with "
                            f"d_out={d_out} must satisfy rows=={n_flat} or rows*k=={n_flat}"
                        )
                    if d_out == 1:
                        return nn.functional.mse_loss(
                            y_pred_flat.squeeze(-1), y_rep.squeeze(-1)
                        )
                    return nn.functional.mse_loss(y_pred_flat, y_rep)
                raise ValueError(
                    f"TabM: regression targets must be 1D or 2D, got {tuple(y_true.shape)}"
                )

            if y_true.dim() == 1:
                if d_out != 1:
                    raise ValueError(
                        f"TabM: 1D targets imply d_out=1, got d_out={d_out}"
                    )
                if y_true.shape[0] == n_flat:
                    return nn.functional.mse_loss(y_pred_flat.squeeze(-1), y_true)
                if y_true.shape[0] * k == n_flat:
                    y_rep = y_true.repeat_interleave(k)
                    return nn.functional.mse_loss(y_pred_flat.squeeze(-1), y_rep)
                raise ValueError(
                    f"TabM: 1D y_true length {y_true.shape[0]} vs flattened preds {n_flat}"
                )
            if y_true.dim() == 2:
                if y_true.shape[0] == n_flat and y_true.shape[1] == d_out:
                    y_rep = y_true
                elif y_true.shape[1] == d_out and y_true.shape[0] * k == n_flat:
                    y_rep = y_true.repeat_interleave(k, dim=0)
                elif (
                    d_out == 1
                    and y_true.shape[1] == k
                    and y_true.shape[0] * k == n_flat
                ):
                    y_rep = y_true.flatten(0, 1)
                else:
                    raise ValueError(
                        "TabM regression: expected y_true of shape "
                        f"({n_flat}, {d_out}) (train multiregression), "
                        f"({n_flat // k}, {d_out}) (eval slice), or "
                        f"({n_flat // k}, {k}) (train scalar), got {tuple(y_true.shape)} "
                        f"with n_flat={n_flat}, d_out={d_out}, k={k}"
                    )
                if d_out == 1:
                    return nn.functional.mse_loss(
                        y_pred_flat.squeeze(-1), y_rep.squeeze(-1)
                    )
                return nn.functional.mse_loss(y_pred_flat, y_rep)
            raise ValueError(
                f"TabM: regression targets must be 1D or 2D, got shape {tuple(y_true.shape)}"
            )

        if shared:
            if y_true.dim() == 1:
                if y_true.shape[0] * k != n_flat:
                    raise ValueError(
                        f"TabM (shared classification): len(y)*k must be {n_flat}, "
                        f"got len={y_true.shape[0]}"
                    )
                y_true_repeated = y_true.repeat_interleave(k)
            elif y_true.dim() == 2 and y_true.shape[1] == 1:
                if y_true.shape[0] * k != n_flat:
                    raise ValueError(
                        f"TabM (shared classification): (B,1) needs B*k={n_flat}, "
                        f"got B={y_true.shape[0]}"
                    )
                y_true_repeated = y_true.squeeze(-1).repeat_interleave(k)
            else:
                raise ValueError(
                    f"TabM (shared classification): expected 1D labels or (B, 1), "
                    f"got {tuple(y_true.shape)}"
                )
            return nn.functional.cross_entropy(y_pred_flat, y_true_repeated.long())

        if y_true.dim() == 2:
            if y_true.shape[0] * y_true.shape[1] != n_flat:
                raise ValueError(
                    f"TabM: classification y_true {tuple(y_true.shape)} does not match "
                    f"flattened logits rows {n_flat}"
                )
            y_true_repeated = y_true.flatten(0, 1)
        elif y_true.dim() == 1:
            if y_true.shape[0] == n_flat:
                y_true_repeated = y_true
            elif y_true.shape[0] * k == n_flat:
                y_true_repeated = y_true.repeat_interleave(k)
            else:
                raise ValueError(
                    f"TabM: classification y_true length {y_true.shape[0]} vs n_flat {n_flat}"
                )
        else:
            raise ValueError(
                f"TabM: classification y_true must be 1D or 2D, got {tuple(y_true.shape)}"
            )
        return nn.functional.cross_entropy(y_pred_flat, y_true_repeated.long())

    def _backward_chunk(
        self, X_num, X_cat, y_tensor, indices, batch_rows, is_regression
    ):
        """Keep forward/backward tensors local so they are released before the next chunk."""
        batch_X_num = X_num[indices].to(self.device) if X_num is not None else None
        batch_X_cat = X_cat[indices].to(self.device) if X_cat is not None else None
        batch_y = y_tensor[indices].to(self.device)
        if not self.share_training_batches and batch_y.ndim == 3:
            batch_y = batch_y.flatten(0, 1)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.amp_dtype is not None,
            dtype=self.amp_dtype,
        ):
            y_pred = self.model(batch_X_num, batch_X_cat)
            loss = self._compute_loss(y_pred, batch_y, is_regression)

            loss = loss * (len(indices) / batch_rows)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.item()

    def _train_batch(self, X_num, X_cat, y_tensor, indices, is_regression):
        """Accumulate one original batch, then clip and update exactly once."""
        chunk_size = min(self.train_chunk_size_, len(indices))
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                loss = sum(
                    self._backward_chunk(
                        X_num, X_cat, y_tensor, chunk, len(indices), is_regression
                    )
                    for chunk in indices.split(chunk_size)
                )
            except torch.cuda.OutOfMemoryError:
                self.optimizer.zero_grad(set_to_none=True)
                if chunk_size == 1:
                    raise
            else:
                break

            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)
                torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            self.train_chunk_size_ = chunk_size
            logger.warning(
                "TabM training OOM: retrying the same batch in chunks of {}", chunk_size
            )

        if self.gradient_clipping_norm is not None:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clipping_norm
            )

        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss

    def _inference_chunk(self, X_num, X_cat, start, stop, y_tensor, is_regression):
        """Return a CPU prediction or scalar loss without retaining GPU batch tensors."""
        batch_X_num = X_num[start:stop].to(self.device) if X_num is not None else None
        batch_X_cat = X_cat[start:stop].to(self.device) if X_cat is not None else None
        evaluation_mode = torch.no_grad if self.compile_model else torch.inference_mode
        with evaluation_mode():
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.amp_dtype is not None,
                dtype=self.amp_dtype,
            ):
                y_pred = self.model(batch_X_num, batch_X_cat)
            if y_tensor is not None:
                batch_y = y_tensor[start:stop].to(self.device)
                return self._compute_loss(y_pred, batch_y, is_regression).item()

            return y_pred.float().cpu()

    def _inference_batches(
        self, X_num, X_cat, n_samples, y_tensor=None, is_regression=False
    ):
        start = 0
        while start < n_samples:
            size = min(self.eval_batch_size_, n_samples - start)
            try:
                result = self._inference_chunk(
                    X_num, X_cat, start, start + size, y_tensor, is_regression
                )
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
            else:
                yield size, result
                start += size
                continue

            self.eval_batch_size_ = max(1, size // 2)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            logger.warning(
                "TabM inference OOM: reducing batch size to {}", self.eval_batch_size_
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Train ensemble heads with validation-based early stopping."""
        X = np.asarray(X)
        if X_val is not None:
            X_val = np.asarray(X_val)
        self.cat_idx = list(self._cat_idx_input)
        self.num_idx = [i for i in range(X.shape[1]) if i not in self.cat_idx]
        self.n_cat_features = len(self.cat_idx)
        self.n_num_features = len(self.num_idx)
        X, X_val = self._drop_constant_numerical_train_val(X, X_val)
        self._promote_binary_features(X)

        is_regression = self.args.objective in ("regression", "multiregression")

        if is_regression and self.args.objective == "multiregression":
            y = self._coerce_multiregression_y_to_2d(y, len(X))
            if X_val is not None and y_val is not None:
                y_val = self._coerce_multiregression_y_to_2d(y_val, len(X_val))

        self.n_features_train_ = X.shape[1]
        self.num_idx_train_ = self.num_idx.copy()
        self.n_num_features_train_ = self.n_num_features
        self.num_idx_original_ = self.num_idx.copy()
        self.n_num_features_original_ = self.n_num_features

        X = self._apply_quantile_transform(X, is_train=True)

        self.cat_cardinalities = self._compute_cat_cardinalities(X)

        if (
            self.n_num_features > 0
            and self.num_emb_type == "pwl"
            and rtdl_num_embeddings
        ):
            X_num = X[:, self.num_idx].astype(np.float32)
            X_num_tensor = torch.from_numpy(X_num)
            self.num_embeddings_module = self._create_num_embeddings(X_num_tensor)
        elif (
            self.n_num_features > 0
            and self.num_emb_type in ["periodic", "linear_relu"]
            and rtdl_num_embeddings
        ):
            self.num_embeddings_module = self._create_num_embeddings(None)
        else:
            self.num_embeddings_module = None

        if is_regression:
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 1:
                self.label_mean = float(np.mean(y))
                self.label_std = float(np.std(y))
                if self.label_std == 0:
                    self.label_std = 1.0
                y_train = (y - self.label_mean) / self.label_std
            elif y.ndim == 2:
                self.label_mean = np.mean(y, axis=0)
                self.label_std = np.std(y, axis=0)
                self.label_std = np.where(self.label_std == 0, 1.0, self.label_std)
                y_train = (y - self.label_mean) / self.label_std
            else:
                raise ValueError(
                    f"TabM: regression y must be 1D or 2D, got shape {y.shape}"
                )
        else:
            self.label_mean = None
            self.label_std = None
            y_train = y

        self.model = self._create_model()
        if self.model is None:
            raise NotImplementedError("TabM model creation failed.")

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        X_num, X_cat = self._prepare_data(X)
        y_tensor = torch.from_numpy(y_train).float()

        if self.batch_size == "auto":
            batch_size = self._get_auto_batch_size(len(X))
        else:
            batch_size = self.batch_size
        self.train_chunk_size_ = self.train_chunk_size
        self.eval_batch_size_ = self.eval_batch_size

        n_epochs = self.n_epochs
        n_samples = len(X)
        patience = self.patience

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

        logger.info(f"Starting TabM training for {n_epochs} epochs")
        logger.info(
            f"Model parameters: k={self.tabm_k}, n_blocks={self.n_blocks}, "
            f"d_block={self.d_block}, dropout={self.dropout}, "
            f"share_training_batches={self.share_training_batches}"
        )

        logger.info(
            "TabM batches: effective training={}, training chunk={}, evaluation={}, data stored on CPU",
            batch_size,
            self.train_chunk_size_,
            self.eval_batch_size_,
        )

        for epoch in range(n_epochs):
            self.model.train()
            epoch_losses = []

            if self.share_training_batches:
                indices = torch.randperm(n_samples, device=self.device).cpu()
                batches = indices.split(batch_size)
            else:
                indices = (
                    torch.rand((n_samples, self.tabm_k), device=self.device)
                    .argsort(dim=0)
                    .cpu()
                )
                batches = indices.split(batch_size, dim=0)

            for batch_idx in batches:
                epoch_losses.append(
                    self._train_batch(X_num, X_cat, y_tensor, batch_idx, is_regression)
                )

            avg_train_loss = np.mean(epoch_losses)
            self.train_loss_history.append(avg_train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, is_regression)
                self.val_loss_history.append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {
                        key: value.detach().to(device="cpu", copy=True)
                        for key, value in self.model.state_dict().items()
                    }
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if (
                    self.args.logging_period
                    and (epoch + 1) % self.args.logging_period == 0
                ):
                    logger.info(
                        f"Epoch {epoch + 1}/{n_epochs}: "
                        f"train_loss={avg_train_loss:.6f}, "
                        f"val_loss={val_loss:.6f}, "
                        f"patience={self.patience_counter}/{patience}"
                    )

                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if (
                    self.args.logging_period
                    and (epoch + 1) % self.args.logging_period == 0
                ):
                    logger.info(
                        f"Epoch {epoch + 1}/{n_epochs}: train_loss={avg_train_loss:.6f}"
                    )

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model from early stopping")
        elif X_val is not None:
            raise RuntimeError("TabM training produced no finite validation loss")

        return self.train_loss_history, self.val_loss_history

    def _evaluate(self, X: np.ndarray, y: np.ndarray, is_regression: bool) -> float:
        """Evaluate model on a dataset and return loss.

        When called from ``fit``, ``X`` is already sliced to the same columns as training
        (constant numerics were dropped at the start of ``fit``), do not apply
        ``_cols_keep_orig`` again.
        """
        self.model.eval()

        X = self._apply_quantile_transform(X, is_train=False)

        X_num, X_cat = self._prepare_data(X)
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float64)).float()

        if is_regression and self.label_mean is not None:
            if np.ndim(self.label_mean) == 0:
                y_tensor = (y_tensor - float(self.label_mean)) / float(self.label_std)
            else:
                t_mean = torch.as_tensor(
                    self.label_mean, dtype=y_tensor.dtype, device=y_tensor.device
                )
                t_std = torch.as_tensor(
                    self.label_std, dtype=y_tensor.dtype, device=y_tensor.device
                )
                y_tensor = (y_tensor - t_mean) / t_std

        total_loss = 0.0
        n_samples = len(X)
        for size, loss in self._inference_batches(
            X_num, X_cat, n_samples, y_tensor, is_regression
        ):
            total_loss += loss * size

        return total_loss / n_samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise NotImplementedError("Model must be fitted before prediction!")

        self.model.eval()

        X = self._apply_learned_column_mask(X)

        X = self._apply_quantile_transform(X, is_train=False)

        X_num, X_cat = self._prepare_data(X)
        n_samples = len(X)

        is_regression = self.args.objective in ("regression", "multiregression")
        predictions_list = []

        for _, y_pred in self._inference_batches(X_num, X_cat, n_samples):
            if is_regression:
                y_pred_mean = y_pred.mean(dim=1).cpu().numpy()
                if y_pred_mean.shape[-1] == 1:
                    y_pred_mean = y_pred_mean.squeeze(-1)

                if self.label_mean is not None:
                    if np.ndim(self.label_mean) == 0:
                        y_pred_mean = y_pred_mean * float(self.label_std) + float(
                            self.label_mean
                        )
                    else:
                        y_pred_mean = y_pred_mean * np.asarray(
                            self.label_std, dtype=np.float64
                        ) + np.asarray(self.label_mean, dtype=np.float64)

                predictions_list.append(y_pred_mean)
            else:
                y_pred_np = y_pred.cpu().numpy()
                if self.average_logits:
                    probs_mean = scipy.special.softmax(
                        y_pred_np.mean(axis=1), axis=-1
                    )
                else:
                    probs = scipy.special.softmax(y_pred_np, axis=-1)
                    probs_mean = probs.mean(axis=1)
                predictions_list.append(probs_mean)

        if is_regression:
            self.predictions = np.concatenate(predictions_list)
            self.prediction_probabilities = None
            return self.predictions
        else:
            self.prediction_probabilities = np.concatenate(predictions_list)

            if self.args.objective == "binary":
                self.predictions = (
                    self.prediction_probabilities[:, 1] >= self.threshold
                ).astype(int)
            else:
                self.predictions = np.argmax(
                    self.prediction_probabilities, axis=1
                ).astype(int)

            return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.args.objective in [
            "binary",
            "classification",
        ], "predict_proba only available for classification tasks"

        self.predict(X)

        probs = self.prediction_probabilities

        if self.args.objective == "binary" and probs.shape[1] == 1:
            probs = np.concatenate([1 - probs, probs], axis=1)

        return probs
