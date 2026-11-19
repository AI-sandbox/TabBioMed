import torch
import logging
import gc

logger = logging.getLogger(__name__)


def svd_flip(u, v, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    u *= signs
    v *= signs.unsqueeze(1)
    return u, v


class TorchPCA:
    def __init__(
        self,
        n_components=None,
        fit="auto",
        svd_driver=None,
        lowrank_oversamples=32,
        lowrank_niter=4,
    ):
        self.n_components = n_components
        self.fit_method = fit
        if fit == "lowrank" and svd_driver is not None:
            logger.warning("svd_driver is not used in 'lowrank' fit mode.")
        self.svd_driver = svd_driver
        self.lowrank_oversamples = lowrank_oversamples
        self.lowrank_niter = lowrank_niter

    def _array_on_device(self, X, device, fitting):
        tensor = torch.tensor(X, dtype=torch.float32, device=device)
        result = self.fit_transform(tensor) if fitting else self.transform(tensor)
        return result.cpu().numpy()

    def _array_with_cpu_fallback(self, X, device, fitting):
        device = torch.device(device)
        rng_state = (
            torch.random.get_rng_state() if fitting and device.type == "cuda" else None
        )
        try:
            return self._array_on_device(X, device, fitting)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            logger.warning(
                "PCA %s exhausted GPU memory, retrying the same operation on CPU.",
                "fit" if fitting else "transform",
            )

        if fitting:
            for name in ("mean_", "components_", "feature_mask_", "nonfinite_fill_"):
                self.__dict__.pop(name, None)
            torch.random.set_rng_state(rng_state)
        else:
            self.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        self.cpu_fallback_ = True
        return self._array_on_device(X, torch.device("cpu"), fitting)

    def fit_transform_array(self, X, device):
        """Fit on training rows, retrying CUDA OOM on CPU with identical PCA settings."""
        self.cpu_fallback_ = False
        result = self._array_with_cpu_fallback(X, device, fitting=True)
        self.fit_device_ = str(self.mean_.device)
        return result

    def transform_array(self, X):
        """Use the fitted statistics/basis on their device, never refit held-out rows."""
        return self._array_with_cpu_fallback(X, self.mean_.device, fitting=False)

    def _clean_fit_input(self, X):
        if not torch.is_floating_point(X):
            X = X.float()
        finite = torch.isfinite(X)
        if finite.all():
            self.nonfinite_fill_ = X.mean(dim=0)
            return X

        counts = finite.sum(dim=0).clamp_min(1)
        fill = (
            torch.where(finite, X, torch.zeros((), dtype=X.dtype, device=X.device)).sum(
                dim=0
            )
            / counts
        )
        fill = torch.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
        self.nonfinite_fill_ = fill
        return torch.where(finite, X, fill.unsqueeze(0))

    def _clean_transform_input(self, X):
        if not torch.is_floating_point(X):
            X = X.float()
        finite = torch.isfinite(X)
        if finite.all():
            return X
        fill = self.nonfinite_fill_.to(device=X.device, dtype=X.dtype)
        return torch.where(finite, X, fill.unsqueeze(0))

    def _resolve_fit_method(self):
        if self.fit_method != "auto":
            return self.fit_method

        return "lowrank"

    def _fit(self, X):
        X = self._clean_fit_input(X)
        n_samples, n_features = X.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError(
                f"PCA requires a non-empty 2D array, got shape {tuple(X.shape)}"
            )

        self.n_features_in_ = n_features
        self.n_components_ = max(
            1, min(self.n_components or min(X.shape), min(X.shape))
        )

        if self.n_components_ > min(X.shape):
            raise ValueError(
                f"n_components should be <= min(n_samples: {n_samples}, "
                f"n_features: {n_features})"
            )

        full_mean = torch.mean(X, dim=0)
        centered = X - full_mean
        active_mask = torch.any(centered != 0, dim=0)
        if not bool(active_mask.any()):
            active_mask = torch.zeros(n_features, dtype=torch.bool, device=X.device)
            active_mask[0] = True
        self.feature_mask_ = active_mask
        self.mean_ = full_mean[active_mask]
        X_active = X[:, active_mask]
        self.n_components_ = max(
            1, min(self.n_components_, X_active.shape[1], n_samples)
        )

        fit_method = self._resolve_fit_method()
        if fit_method == "full":
            U, S, Vt = torch.linalg.svd(
                X_active - self.mean_, full_matrices=True, driver=self.svd_driver
            )
        elif fit_method == "reduced":
            U, S, Vt = torch.linalg.svd(
                X_active - self.mean_, full_matrices=False, driver=self.svd_driver
            )
        elif fit_method == "lowrank":
            q = min(min(X_active.shape), self.n_components_ + self.lowrank_oversamples)
            U, S, V = torch.svd_lowrank(
                X_active,
                q=q,
                niter=self.lowrank_niter,
                M=self.mean_,
            )
            Vt = V.mT
        else:
            raise ValueError(f"Unrecognized fit method: {self.fit_method}")

        U = U[:, : self.n_components_]
        S = S[: self.n_components_]
        Vt = Vt[: self.n_components_]

        U, Vt = svd_flip(U, Vt)
        self.components_ = Vt

        return U, S, Vt

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError(
                f"PCA transform expects a 2D array, got shape {tuple(X.shape)}"
            )
        if X.shape[1] == self.n_components_ and X.shape[1] != self.n_features_in_:
            logger.debug("Input already has PCA output width, skipping PCA transform.")
            return X
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "PCA transform expected "
                f"{self.n_features_in_} input features, got {X.shape[1]}. "
                f"If this data was already transformed, pass the original features instead."
            )
        X = self._clean_transform_input(X)
        X_active = X[:, self.feature_mask_]
        return torch.matmul(X_active - self.mean_, self.components_.T)

    def fit_transform(self, X):
        U, S, _ = self._fit(X)
        return U * S.unsqueeze(0)

    def to(self, device):
        self.mean_ = self.mean_.to(device)
        self.components_ = self.components_.to(device)
        self.feature_mask_ = self.feature_mask_.to(device)
        self.nonfinite_fill_ = self.nonfinite_fill_.to(device)
        return self
