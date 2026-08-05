"""Weights & Biases logging behind a ``tensorboardX.SummaryWriter``-compatible API.

The training code keeps calling ``writer.add_scalar(tag, value, global_step)``, so
every metric name and x-axis stays identical to the tensorboard setup this
replaces. Scalars sharing one ``global_step`` are buffered and shipped as a single
``wandb.log`` row, which matters for the distillation stage that logs on every
environment step.

Settings come from ``wandb_config.json`` at the repository root. Standard wandb
environment variables (``WANDB_API_KEY``, ``WANDB_PROJECT``, ``WANDB_ENTITY``,
``WANDB_MODE``) take precedence over that file.

If wandb is missing or fails to start, logging degrades to a no-op and training
continues; it never takes a run down.
"""

from __future__ import annotations

import atexit
import json
import os

CONFIG_FILE_NAME = "wandb_config.json"
CONFIG_PATH_ENV_VAR = "SHARPA_WANDB_CONFIG"
STEP_METRIC = "agent_steps"

DEFAULT_SETTINGS = {
    "api_key": "",
    "project": "sharpa",
    "entity": None,
    "mode": "online",
    "enabled": True,
}

_warned: set[str] = set()


def _warn_once(message: str) -> None:
    """Print a warning at most once per process, keyed on the message itself."""
    if message in _warned:
        return
    _warned.add(message)
    print(f"[wandb] {message}", flush=True)


def _repo_root() -> str:
    """Repository root, derived from this file's location."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_file_candidates() -> list[str]:
    """Config file locations, highest priority first."""
    candidates = []
    explicit = os.environ.get(CONFIG_PATH_ENV_VAR)
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(os.getcwd(), CONFIG_FILE_NAME))
    candidates.append(os.path.join(_repo_root(), CONFIG_FILE_NAME))
    return candidates


def load_settings() -> dict:
    """Resolve wandb settings from environment variables, then the JSON config file.

    A missing or malformed config file falls back to :data:`DEFAULT_SETTINGS` with a
    warning rather than raising, so training is never blocked on logging config.
    """
    settings = dict(DEFAULT_SETTINGS)

    for path in _config_file_candidates():
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception as error:  # malformed json, permissions, ...
            _warn_once(f"ignoring config file {path}: {error}")
            break
        if isinstance(loaded, dict):
            settings.update({k: v for k, v in loaded.items() if k in DEFAULT_SETTINGS})
        else:
            _warn_once(f"ignoring config file {path}: expected a JSON object")
        break

    # Standard wandb environment variables win over the config file.
    for key, env_var in (
        ("api_key", "WANDB_API_KEY"),
        ("project", "WANDB_PROJECT"),
        ("entity", "WANDB_ENTITY"),
        ("mode", "WANDB_MODE"),
    ):
        value = os.environ.get(env_var)
        if value:
            settings[key] = value

    return settings


def _to_float(value):
    """Convert torch/numpy scalars and plain python numbers to ``float``.

    Uses duck-typing on ``.item()`` so this module never has to import torch.
    """
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return float(value)


def _to_dict(cfg):
    """Best-effort conversion of a config object into a plain dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    try:
        import dataclasses

        if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
            return dataclasses.asdict(cfg)
    except Exception:
        pass
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return {"repr": repr(cfg)}


def full_config_to_dict(full_config):
    """Flatten a :class:`ConfigWrapper` into the dict recorded as the wandb config."""
    if full_config is None:
        return {}
    return {
        "train": _to_dict(getattr(full_config, "train", None)),
        "task": _to_dict(getattr(full_config, "task", None)),
        "test": getattr(full_config, "test", False),
    }


def run_identity(output_dir, experiment_name, stage):
    """Derive ``(run_name, group)`` for a wandb run from the log directory layout.

    ``output_dir`` is ``logs/<experiment_name>/<timestamp>``, and both training
    stages of one experiment share it, so grouping on that path puts the PPO run
    and its distillation run next to each other in the wandb UI.
    """
    timestamp = os.path.basename(os.path.normpath(output_dir)) if output_dir else ""
    group = f"{experiment_name}/{timestamp}" if timestamp else str(experiment_name)
    return f"{group}/{stage}", group


class WandbWriter:
    """Drop-in replacement for ``tensorboardX.SummaryWriter``, backed by wandb.

    Only the surface the training code actually uses is implemented faithfully:
    ``add_scalar``, plus ``flush``/``close``. ``add_scalars``, ``add_histogram`` and
    ``add_text`` are provided so a future caller does not hit an ``AttributeError``.
    """

    def __init__(self, log_dir=None, config=None, run_name=None, group=None, job_type=None):
        self.log_dir = log_dir
        self.run = None
        self._buffer: dict[str, object] = {}
        self._buffered_step = None
        self._last_logged_step = None
        self._closed = False

        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as error:
                _warn_once(f"could not create log dir {log_dir}: {error}")
                self.log_dir = None

        self._init_run(config=config, run_name=run_name, group=group, job_type=job_type)
        atexit.register(self._atexit_flush)

    def _init_run(self, config, run_name, group, job_type):
        """Start the wandb run, leaving ``self.run`` as ``None`` on any failure."""
        settings = load_settings()
        if not settings.get("enabled", True):
            _warn_once("logging disabled via wandb_config.json (enabled=false)")
            return

        try:
            import wandb
        except ImportError as error:
            _warn_once(f"wandb is not installed ({error}); metrics will not be recorded")
            return

        api_key = (settings.get("api_key") or "").strip()
        if api_key:
            # Consumed by wandb.init; avoids an interactive login prompt on a
            # headless machine, which would otherwise hang the run.
            os.environ["WANDB_API_KEY"] = api_key
        elif not os.environ.get("WANDB_API_KEY"):
            _warn_once(
                "no API key in wandb_config.json or WANDB_API_KEY; relying on a "
                "previous `wandb login`"
            )

        try:
            self.run = wandb.init(
                project=settings.get("project") or DEFAULT_SETTINGS["project"],
                entity=settings.get("entity") or None,
                mode=settings.get("mode") or "online",
                dir=self.log_dir or None,
                name=run_name,
                group=group,
                job_type=job_type,
                config=config or {},
            )
        except Exception as error:
            _warn_once(f"could not start run ({error}); metrics will not be recorded")
            self.run = None
            return

        # Plot every metric against agent_steps, matching the tensorboard x-axis.
        try:
            wandb.define_metric(STEP_METRIC)
            wandb.define_metric("*", step_metric=STEP_METRIC)
        except Exception as error:
            _warn_once(f"could not define the {STEP_METRIC} x-axis: {error}")

    """
    Operations -- SummaryWriter API
    """

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Record one scalar under ``tag``, buffered until ``global_step`` advances."""
        if self.run is None:
            return
        try:
            value = _to_float(scalar_value)
        except (TypeError, ValueError) as error:
            _warn_once(f"skipping non-scalar value for '{tag}': {error}")
            return
        self._stash(tag, value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Record several scalars sharing a ``main_tag`` prefix."""
        for tag, value in dict(tag_scalar_dict).items():
            self.add_scalar(f"{main_tag}/{tag}", value, global_step)

    def add_histogram(self, tag, values, global_step=None, *args, **kwargs):
        """Record a histogram. Unused by the current training code."""
        if self.run is None:
            return
        try:
            import numpy as np
            import wandb

            data = np.asarray(getattr(values, "detach", lambda: values)().cpu()
                              if hasattr(values, "detach") else values)
            self._stash(tag, wandb.Histogram(data), global_step)
        except Exception as error:
            _warn_once(f"skipping histogram '{tag}': {error}")

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Record a string. Unused by the current training code."""
        if self.run is None:
            return
        self._stash(tag, str(text_string), global_step)

    def _stash(self, tag, value, global_step):
        """Buffer a value, flushing the previous batch when the step advances."""
        if global_step is not None and self._buffered_step is not None \
                and global_step != self._buffered_step:
            self._flush_buffer()
        if global_step is not None:
            self._buffered_step = global_step
        self._buffer[tag] = value

    def _flush_buffer(self):
        """Ship the buffered batch as a single wandb history row."""
        if not self._buffer or self.run is None:
            self._buffer = {}
            self._buffered_step = None
            return

        payload = dict(self._buffer)
        step = self._buffered_step
        log_step = None
        if step is not None:
            payload[STEP_METRIC] = step
            # wandb requires a non-decreasing step. The true value always travels
            # in the payload, so clamping here costs no data.
            log_step = int(step)
            if self._last_logged_step is not None:
                log_step = max(log_step, self._last_logged_step)

        try:
            self.run.log(payload, step=log_step)
            if log_step is not None:
                self._last_logged_step = log_step
        except Exception as error:
            _warn_once(f"dropping a batch of metrics: {error}")

        self._buffer = {}
        self._buffered_step = None

    def flush(self):
        """Send anything still buffered."""
        self._flush_buffer()

    def close(self):
        """Flush and finish the run. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._flush_buffer()
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as error:
            _warn_once(f"could not finish the run cleanly: {error}")
        finally:
            self.run = None

    def _atexit_flush(self):
        """Last-chance flush so a killed long run keeps its final batch."""
        try:
            self.close()
        except Exception:
            pass
