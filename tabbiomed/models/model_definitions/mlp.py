import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from tabbiomed.models.base_model import BaseModel, ModelArgs


class _BatchNormBatchSampler(BatchSampler):
    """Merge a final singleton into the preceding batch, without dropping rows."""

    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    batch = pending + batch
                else:
                    yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        merge_last = len(self.sampler) > 1 and len(self.sampler) % self.batch_size == 1
        return super().__len__() - int(merge_last)


class MLP(BaseModel):
    """
    Custom implementation for the standard multi-layer perceptron
    """

    required_parameters = ("n_layers", "hidden_dim", "dropout_rate", "learning_rate")

    @classmethod
    def get_base_params(cls, args):
        return {
            "n_epochs": 1000000000,
            "use_batch_norm": False,
            "activation": "relu",
            "weight_decay": 1e-06,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        self.n_epochs = params["n_epochs"]

        if args.objective == "binary":
            output_dim = 1
        elif args.objective == "multiregression":
            output_dim = args.num_outputs
        elif args.objective == "classification":
            output_dim = args.num_classes
        else:
            output_dim = 1

        self.model = MLP_Model(
            n_layers=self.params["n_layers"],
            input_dim=self.args.num_features,
            hidden_dim=self.params["hidden_dim"],
            output_dim=output_dim,
            task=self.args.objective,
            dropout_rate=self.params["dropout_rate"],
            use_batch_norm=self.params["use_batch_norm"],
            activation=self.params["activation"],
        )

        self.device = self.get_device()

        self.to_device()

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return True

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        return True

    @classmethod
    def uses_validation_during_fit(cls) -> bool:
        return True

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        # code from https://github.com/ferreirafabio/tabrepo/blob/651da4f3e1cb5701b084d65d2c1b43a99e59c8ff/tabrepo/models/nn_torch/generate.py
        """
        Default params for MLP based on TabRepo (https://github.com/ferreirafabio/tabrepo/blob/651da4f3e1cb5701b084d65d2c1b43a99e59c8ff/tabrepo/models/nn_torch/generate.py).
        These parameters work for both classification and regression.
        """
        return {
            "n_layers": 2,
            "hidden_dim": 128,
            "dropout_rate": 0.1,
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "use_batch_norm": False,
            "activation": "relu",
            "n_epochs": 1_000_000_000,
        }

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"
            else:
                device = "cuda"
        else:
            device = "cpu"

        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective in ("regression", "multiregression"):
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
            y = y.long()
            y_val = y_val.long()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        train_dataset = TensorDataset(X, y)
        loader_params = {"batch_size": self.args.batch_size, "shuffle": True}
        if self.params["use_batch_norm"]:
            if len(train_dataset) < 2 or self.args.batch_size < 2:
                raise ValueError(
                    "MLP BatchNorm requires at least two training rows and batch_size >= 2"
                )
            loader_params = {
                "batch_sampler": _BatchNormBatchSampler(
                    RandomSampler(train_dataset, generator=generator),
                    self.args.batch_size,
                    drop_last=False,
                )
            }
        train_loader = DataLoader(
            dataset=train_dataset,
            **loader_params,
            num_workers=2,
            generator=generator,
        )

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            generator=generator,
        )

        min_val_loss = float("inf")
        min_val_loss_idx = 0
        best_state = None

        loss_history = []
        val_loss_history = []

        start_time = time.time()
        for epoch in range(self.n_epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                out = self.model(batch_X.to(self.device))

                if self.args.objective in ("regression", "binary"):
                    out = out.squeeze(-1)

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            val_dim = 0
            with torch.no_grad():
                for batch_val_X, batch_val_y in val_loader:
                    out = self.model(batch_val_X.to(self.device))

                    if self.args.objective in ("regression", "binary"):
                        out = out.squeeze(-1)

                    val_loss += loss_func(
                        out, batch_val_y.to(self.device)
                    ).item() * len(batch_val_y)
                    val_dim += len(batch_val_y)

            val_loss /= val_dim
            val_loss_history.append(val_loss)

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

            if epoch - min_val_loss_idx >= self.args.early_stopping_rounds:
                print(
                    "Validation loss has not improved for %d steps!"
                    % self.args.early_stopping_rounds
                )
                print("Early stopping applies.")
                break

            runtime = time.time() - start_time
            if time_limit is not None and runtime > time_limit:
                print(
                    f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                )
                break

        if best_state is None:
            raise RuntimeError("MLP training produced no finite validation loss")
        self.model.load_state_dict(best_state)
        return loss_history, val_loss_history

    def predict(self, X):
        if hasattr(self, "_output_models_") and self._output_models_:
            return super().predict(X)

        if self.args.objective in ("regression", "multiregression"):
            preds = self.predict_helper(X)
            if (
                self.args.objective == "regression"
                and preds.ndim == 2
                and preds.shape[1] == 1
            ):
                preds = preds.squeeze(-1)
            self.predictions = preds
        elif self.args.objective == "binary":
            self.predict_proba(X)

            self.predictions = (
                self.prediction_probabilities[:, 1] >= self.threshold
            ).astype(int)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=2,
        )
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)
                elif self.args.objective == "classification":
                    preds = F.softmax(preds, dim=1)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)


class MLP_Model(nn.Module):
    """
    Multi-Layer Perceptron with optional Batch Normalization.
    Architecture follows: Linear -> BatchNorm (optional) -> Activation -> Dropout
    """

    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        output_dim,
        task,
        dropout_rate=0.2,
        use_batch_norm=False,
        activation="relu",
    ):
        super().__init__()

        self.task = task
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        if activation == "relu":
            self.act_fn = F.relu
        elif activation == "elu":
            self.act_fn = F.elu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.linear_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList()

        self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout_rate))

        for _ in range(n_layers - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for i, linear in enumerate(self.linear_layers):
            x = linear(x)

            if self.use_batch_norm:
                x = self.batch_norm_layers[i](x)

            x = self.act_fn(x)
            x = self.dropout_layers[i](x)

        x = self.output_layer(x)

        return x
