"""Trial process isolation and HPO wall-clock deadlines."""

from __future__ import annotations
import os
import pickle
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable
from loguru import logger

_EXEC_TIMEOUT_LOGGED = False


class HPOStepDeadlineExceeded(RuntimeError):
    """Raised to stop ``study.optimize`` when the HPO step wall clock is exhausted."""


class TrialProcessError(RuntimeError):
    """A training child exited without a valid result."""


def set_hpo_step_deadline(sweep: Any, timeout_seconds: int | None) -> float | None:
    """Attach a monotonic deadline to *sweep*; return the deadline timestamp."""
    if timeout_seconds is None or int(timeout_seconds) <= 0:
        sweep.hpo_step_deadline_monotonic = None
        return None
    sweep.hpo_step_started_monotonic = time.monotonic()
    deadline = sweep.hpo_step_started_monotonic + int(timeout_seconds)
    sweep.hpo_step_deadline_monotonic = deadline
    return deadline


def hpo_step_seconds_remaining(sweep: Any) -> float | None:
    deadline = getattr(sweep, "hpo_step_deadline_monotonic", None)
    if deadline is None:
        return None
    return deadline - time.monotonic()


def trial_ref_for_subprocess(trial: Any) -> SimpleNamespace:
    """Minimal, picklable trial handle that satisfies train() logging in a child."""
    return SimpleNamespace(number=int(trial.number))


def _child_main(
    result_path: str,
    train_func: Callable[..., Any],
    train_kwargs: dict[str, Any],
    return_rng_states: bool = False,
) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    try:
        local_test_preds: list = []
        kwargs = {**train_kwargs, "test_preds": local_test_preds}
        out = train_func(**kwargs)
        if len(out) == 5 and out[0] is None:
            error = getattr(kwargs.get("dataset"), "last_evaluation_error", None)
            if error:
                out = (*out, {"fit_error": error})
        with open(result_path, "wb") as f:
            payload = ("ok", out, local_test_preds)
            if return_rng_states:
                payload += (_trial_rng_states(),)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os._exit(0)
    except Exception as exc:
        try:
            with open(result_path, "wb") as f:
                pickle.dump(("err", repr(exc), []), f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
        os._exit(1)


def _read_trial_result(result_path: str) -> tuple[tuple[Any, ...], list] | None:
    try:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        (status, payload, local_test_preds) = result[:3]
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError, TypeError):
        return None
    if status == "err":
        logger.warning(f"Trial training child failed: {payload}")
        return None
    if len(result) == 4:
        _restore_trial_rng_states(result[3])
    return (payload, local_test_preds)


def _trial_rng_states() -> dict[str, Any]:
    """Preserve RNG states and execution settings without forking CUDA."""
    states = {"random": random.getstate()}
    if "numpy" in sys.modules:
        states["numpy"] = sys.modules["numpy"].random.get_state()
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        states["torch"] = torch.get_rng_state()
        states["torch_seed"] = torch.initial_seed()
        states["torch_settings"] = dict(
            num_threads=torch.get_num_threads(),
            deterministic=torch.backends.cudnn.deterministic,
            benchmark=torch.backends.cudnn.benchmark,
            algorithms=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        if torch.cuda.is_initialized():
            states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_trial_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["random"])
    if "numpy" in states:
        import numpy as np

        np.random.set_state(states["numpy"])
    if "torch" in states:
        import torch

        torch.set_rng_state(states["torch"])
        settings = states.get("torch_settings")
        if settings:
            torch.set_num_threads(settings["num_threads"])
            torch.backends.cudnn.deterministic = settings["deterministic"]
            torch.backends.cudnn.benchmark = settings["benchmark"]
            torch.use_deterministic_algorithms(
                settings["algorithms"], warn_only=settings["warn_only"]
            )
        if "cuda" in states:
            torch.cuda.set_rng_state_all(states["cuda"])
        else:
            torch.cuda.manual_seed_all(states.get("torch_seed", torch.initial_seed()))


def _exec_child_main(input_path: str, result_path: str) -> None:
    with open(input_path, "rb") as f:
        (train_func, train_kwargs, states) = pickle.load(f)
    _restore_trial_rng_states(states)
    _child_main(
        result_path, train_func, train_kwargs, states.get("propagate_rng_state", False)
    )


def _kill_exec_trial(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    finally:
        process.wait()


def _ipc_temp_dir_kwargs() -> dict[str, str]:
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp and os.path.isdir(slurm_tmp) and os.access(slurm_tmp, os.W_OK):
        return {"dir": slurm_tmp}
    return {}


def run_train_with_exec_timeout(
    train_func: Callable[..., Any],
    train_kwargs: dict[str, Any],
    timeout_seconds: float | None,
    *,
    raise_on_failure: bool = False,
    propagate_rng_state: bool = False,
) -> tuple[tuple[Any, ...], list] | None:
    """Run a trial in fresh Python with the parent's RNG and Torch settings.

    Serialization and interpreter startup consume the remaining HPO budget.
    The input goes through a private file so launch does not block on a large
    multiprocessing spawn pipe. The child and its workers are killed on timeout
    or parent-side exceptions; payloads are removed on every exit path.
    """
    global _EXEC_TIMEOUT_LOGGED
    if not _EXEC_TIMEOUT_LOGGED:
        logger.info(
            "Evaluation isolation: fresh Python interpreter with a hard deadline."
        )
        _EXEC_TIMEOUT_LOGGED = True
    deadline = (
        None
        if timeout_seconds is None
        else time.monotonic() + float(timeout_seconds)
    )
    if deadline is not None and time.monotonic() >= deadline:
        return None
    with tempfile.TemporaryDirectory(
        prefix="tabbiomed_fit_", **_ipc_temp_dir_kwargs()
    ) as tmp:
        input_path = os.path.join(tmp, "input.pkl")
        result_path = os.path.join(tmp, "result.pkl")
        with open(input_path, "wb") as f:
            pickle.dump(
                (
                    train_func,
                    {**train_kwargs, "test_preds": []},
                    {**_trial_rng_states(), "propagate_rng_state": propagate_rng_state},
                ),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if deadline is not None and time.monotonic() >= deadline:
            return None
        env = {
            **os.environ,
            "PYTHONFAULTHANDLER": "1",
            "PYTHONPATH": os.pathsep.join((os.path.abspath(p) for p in sys.path)),
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tabbiomed.trainer.hpo_step_timeout",
                input_path,
                result_path,
            ],
            env=env,
            start_new_session=True,
        )
        try:
            try:
                wait_timeout = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                code = process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Trial training exceeded remaining HPO budget ({timeout_seconds:.0f}s); SIGKILL on fresh trial process group."
                )
                return None
            if code != 0:
                logger.warning(f"Trial training child exited with code {code}.")
            result = _read_trial_result(result_path)
            if result is None and raise_on_failure:
                reason = (
                    f"signal {signal.Signals(-code).name}"
                    if code < 0
                    else f"exit code {code}"
                )
                raise TrialProcessError(f"Training child failed with {reason}")
            return result
        finally:
            _kill_exec_trial(process)


if __name__ == "__main__":
    _exec_child_main(*sys.argv[1:])
