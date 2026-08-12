"""Offline checks for the wandb logging shim.

Runs on a bare Python install: wandb is stubbed, and torch / isaaclab are never
imported. The point is to verify, without the simulator, that every metric the
tensorboard writer used to emit still reaches wandb, batched one row per step.

    python tests/test_wandb_logger.py
    python -m pytest tests/test_wandb_logger.py -v
"""

import os
import re
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PPO_SOURCE = os.path.join(REPO_ROOT, "rl_isaaclab", "algo", "ppo", "ppo.py")
PADAPT_SOURCE = os.path.join(REPO_ROOT, "rl_isaaclab", "algo", "padapt", "padapt.py")


class FakeRun:
    """Stand-in for a wandb Run that records every log call."""

    def __init__(self):
        self.calls = []
        self.finished = False

    def log(self, payload, step=None):
        self.calls.append((dict(payload), step))

    def finish(self):
        self.finished = True


class FakeWandb:
    """Stand-in for the wandb module."""

    def __init__(self, fail_init=False):
        self.run = FakeRun()
        self.init_kwargs = None
        self.defined_metrics = []
        self.fail_init = fail_init

    def init(self, **kwargs):
        if self.fail_init:
            raise RuntimeError("simulated init failure")
        self.init_kwargs = kwargs
        return self.run

    def define_metric(self, name, step_metric=None):
        self.defined_metrics.append((name, step_metric))


def install_fake_wandb(fail_init=False):
    """Put a stub wandb module on ``sys.modules`` and return the stub."""
    fake = FakeWandb(fail_init=fail_init)
    module = types.ModuleType("wandb")
    module.init = fake.init
    module.define_metric = fake.define_metric
    module.Histogram = lambda data: ("histogram", data)
    sys.modules["wandb"] = module
    return fake


def fresh_logger():
    """Import the logger module with per-test warning state cleared."""
    from rl_isaaclab.utils import wandb_logger

    wandb_logger._warned.clear()
    return wandb_logger


# Starting a run exports WANDB_API_KEY into the process environment (that is how
# the key reaches wandb without an interactive prompt), so tests must snapshot and
# restore these to stay independent of each other.
MANAGED_ENV_VARS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "SHARPA_WANDB_CONFIG",
)


def _snapshot_env():
    return {name: os.environ.get(name) for name in MANAGED_ENV_VARS}


def _restore_env(snapshot):
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


_ENV_SNAPSHOT = {}


def setup_function(function=None):
    """Called by pytest before each test, and by main() in standalone mode."""
    _ENV_SNAPSHOT.clear()
    _ENV_SNAPSHOT.update(_snapshot_env())
    for name in MANAGED_ENV_VARS:
        os.environ.pop(name, None)
    sys.modules.pop("wandb", None)


def teardown_function(function=None):
    """Restore the environment a test may have mutated."""
    _restore_env(_ENV_SNAPSHOT)
    sys.modules.pop("wandb", None)


# ---------------------------------------------------------------------------
# The metric inventory the tensorboard writer used to produce.
# ---------------------------------------------------------------------------

PPO_FIXED_TAGS = [
    "performance/RLTrainFPS",
    "performance/EnvStepFPS",
    "losses/actor_loss",
    "losses/bounds_loss",
    "losses/critic_loss",
    "losses/entropy",
    "info/last_lr",
    "info/e_clip",
    "info/kl",
]

EPISODE_TAGS = ["episode_rewards/step", "episode_lengths/step"]

REWARD_TERMS = [
    "total_reward",
    "rotate_reward",
    "object_linvel_penalty",
    "pos_diff_penalty",
    "torque_penalty",
    "work_penalty",
    "object_pos_diff",
    "position_reward",
    "object_z_penalty",
    "object_tip_z_penalty",
    "object_up_alignment_reward",
    "success_reward",
]

# Scalars the env puts in `extras`; ppo.play_steps forwards them verbatim as
# top-level tags via self.extra_info.
ENV_EXTRA_TAGS = [
    "object_z_diff",
    "object_tip_z_diff",
    "up_alignment",
    "roll",
    "pitch",
    "yaw",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "height_reset_upper",
    "height_reset_lower",
    "time_out",
]

EXPECTED_PPO_TAGS = set(
    PPO_FIXED_TAGS + EPISODE_TAGS + ENV_EXTRA_TAGS + [f"Reward/{t}" for t in REWARD_TERMS]
)
EXPECTED_PADAPT_TAGS = set(EPISODE_TAGS + [f"Reward/{t}" for t in REWARD_TERMS])


def test_reward_terms_match_source():
    """The term list above must stay in sync with EPISODE_REWARD_TERMS."""
    from rl_isaaclab.utils.reward_logging import EPISODE_REWARD_TERMS

    assert list(EPISODE_REWARD_TERMS) == REWARD_TERMS, (
        f"EPISODE_REWARD_TERMS changed: {EPISODE_REWARD_TERMS}"
    )


def test_ppo_metrics_all_reach_wandb():
    """Replay ppo.write_stats + train() and check nothing is dropped."""
    fake = install_fake_wandb()
    wandb_logger = fresh_logger()
    writer = wandb_logger.WandbWriter(config={"train": {}}, run_name="r", group="g")

    assert writer.run is not None, "writer should have started a run"
    assert fake.init_kwargs["project"] == "sharpa"
    assert (wandb_logger.STEP_METRIC, None) in fake.defined_metrics
    assert ("*", wandb_logger.STEP_METRIC) in fake.defined_metrics

    steps = [131072, 262144, 393216]
    for step in steps:
        for tag in PPO_FIXED_TAGS:          # write_stats, fixed tags
            writer.add_scalar(tag, 1.0, step)
        for tag in ENV_EXTRA_TAGS:          # write_stats, self.extra_info
            writer.add_scalar(tag, 0.5, step)
        for term in REWARD_TERMS:           # write_stats, episode reward terms
            writer.add_scalar(f"Reward/{term}", 2.0, step)
        for tag in EPISODE_TAGS:            # train() main loop
            writer.add_scalar(tag, 3.0, step)
    writer.flush()

    # One history row per step, not one per add_scalar call.
    assert len(fake.run.calls) == len(steps), (
        f"expected {len(steps)} batched log calls, got {len(fake.run.calls)}"
    )

    logged_tags = set()
    for (payload, log_step), expected_step in zip(fake.run.calls, steps):
        assert payload[wandb_logger.STEP_METRIC] == expected_step
        assert log_step == expected_step
        logged_tags.update(k for k in payload if k != wandb_logger.STEP_METRIC)

    missing = EXPECTED_PPO_TAGS - logged_tags
    extra = logged_tags - EXPECTED_PPO_TAGS
    assert not missing, f"metrics lost in migration: {sorted(missing)}"
    assert not extra, f"unexpected metrics logged: {sorted(extra)}"

    writer.close()
    assert fake.run.finished, "close() should finish the run"


def test_padapt_metrics_all_reach_wandb():
    """Replay padapt.log_tensorboard, which fires on every env step."""
    fake = install_fake_wandb()
    wandb_logger = fresh_logger()
    writer = wandb_logger.WandbWriter(run_name="r", group="g", job_type="stage2")

    steps = [16384, 32768]
    for step in steps:
        for tag in EPISODE_TAGS:
            writer.add_scalar(tag, 1.0, step)
        for term in REWARD_TERMS:
            writer.add_scalar(f"Reward/{term}", 2.0, step)
    writer.flush()

    assert len(fake.run.calls) == len(steps), "stage2 must batch per step, not per scalar"

    logged_tags = set()
    for payload, _ in fake.run.calls:
        logged_tags.update(k for k in payload if k != wandb_logger.STEP_METRIC)
    assert logged_tags == EXPECTED_PADAPT_TAGS, (
        f"stage2 mismatch: missing={sorted(EXPECTED_PADAPT_TAGS - logged_tags)} "
        f"extra={sorted(logged_tags - EXPECTED_PADAPT_TAGS)}"
    )


# ---------------------------------------------------------------------------
# Static drift guard: keep the inventory above honest against the real sources.
# ---------------------------------------------------------------------------

ADD_SCALAR_RE = re.compile(r"add_scalar\(\s*f?['\"]([^'\"]*)['\"]")


def _tags_in_source(path):
    """Split add_scalar tag arguments into plain literals and f-string templates."""
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    literals, templates = set(), set()
    for tag in ADD_SCALAR_RE.findall(source):
        (templates if "{" in tag else literals).add(tag)
    return literals, templates


def test_ppo_source_tags_match_inventory():
    literals, templates = _tags_in_source(PPO_SOURCE)
    expected = set(PPO_FIXED_TAGS + EPISODE_TAGS)
    assert literals == expected, (
        f"ppo.py add_scalar tags drifted: missing={sorted(expected - literals)} "
        f"new={sorted(literals - expected)}"
    )
    assert templates == {"Reward/{term}", "{k}"}, f"unexpected templates: {templates}"


def test_padapt_source_tags_match_inventory():
    literals, templates = _tags_in_source(PADAPT_SOURCE)
    assert literals == set(EPISODE_TAGS), (
        f"padapt.py add_scalar tags drifted: {sorted(literals)}"
    )
    assert templates == {"Reward/{term}", "{k}/frame"}, f"unexpected templates: {templates}"


TENSORBOARD_USE_RE = re.compile(
    r"^\s*(?:from\s+tensorboardX|import\s+tensorboardX)|SummaryWriter\(", re.MULTILINE
)


def test_no_tensorboard_usage_remains():
    """Nothing should still import tensorboardX or construct a SummaryWriter."""
    offenders = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in {"logs", ".git", "__pycache__", "assets"}]
        for name in files:
            if not (name.endswith(".py") or name in {"Dockerfile", "pyproject.toml"}):
                continue
            path = os.path.join(root, name)
            if os.path.abspath(path) == os.path.abspath(__file__):
                continue
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            if TENSORBOARD_USE_RE.search(text):
                offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, f"tensorboard usage remains in: {offenders}"


def _find_function(tree, class_name, func_name):
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == func_name:
                    return child
    raise AssertionError(f"{class_name}.{func_name} not found")


def _parse(path):
    import ast

    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read())


def test_writer_only_created_when_output_dir_requested():
    """play.py / deploy.py / replay pass create_output_dir=False and must make no run."""
    import ast

    for path, class_name in ((PPO_SOURCE, "PPO"), (PADAPT_SOURCE, "ProprioAdapt")):
        init = _find_function(_parse(path), class_name, "__init__")
        guarded, unguarded = [], []
        for node in ast.walk(init):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "WandbWriter":
                unguarded.append(node)
        for node in ast.walk(init):
            if isinstance(node, ast.If) and "create_output_dir" in ast.dump(node.test):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "WandbWriter":
                        guarded.append(inner)
        assert len(unguarded) == 1, f"{path}: expected exactly one WandbWriter construction"
        assert len(guarded) == 1, f"{path}: WandbWriter must be inside `if create_output_dir`"
        source = open(path, encoding="utf-8").read()
        assert "self.writer = None" in source, f"{path}: self.writer must default to None"


def test_train_closes_the_writer():
    """A finished run should be flushed and closed, not left dangling."""
    for path, class_name in ((PPO_SOURCE, "PPO"), (PADAPT_SOURCE, "ProprioAdapt")):
        import ast

        train = _find_function(_parse(path), class_name, "train")
        closes = [
            node for node in ast.walk(train)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "close"
            and "writer" in ast.dump(node.func)
        ]
        assert closes, f"{path}: {class_name}.train() never closes the writer"


def test_tensorboardx_not_a_declared_dependency():
    """The Dockerfiles and pyproject should install wandb, not tensorboardX."""
    manifests = [
        os.path.join(REPO_ROOT, "pyproject.toml"),
        os.path.join(REPO_ROOT, "rl_isaaclab", "utils", "docker", "cu124", "Dockerfile"),
        os.path.join(REPO_ROOT, "rl_isaaclab", "utils", "docker", "cu128", "Dockerfile"),
    ]
    for path in manifests:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "tensorboardX" not in text, f"{path} still installs tensorboardX"
        assert "wandb" in text, f"{path} does not install wandb"


# ---------------------------------------------------------------------------
# Degradation: logging problems must never take training down.
# ---------------------------------------------------------------------------

def test_missing_wandb_degrades_to_noop():
    sys.modules["wandb"] = None  # makes `import wandb` raise ImportError
    try:
        wandb_logger = fresh_logger()
        writer = wandb_logger.WandbWriter(run_name="r")
        assert writer.run is None
        writer.add_scalar("losses/actor_loss", 1.0, 100)  # must not raise
        writer.flush()
        writer.close()
    finally:
        sys.modules.pop("wandb", None)


def test_init_failure_degrades_to_noop():
    install_fake_wandb(fail_init=True)
    wandb_logger = fresh_logger()
    writer = wandb_logger.WandbWriter(run_name="r")
    assert writer.run is None
    writer.add_scalar("losses/actor_loss", 1.0, 100)
    writer.close()


def test_disabled_via_config_file(tmp_path=None):
    import json
    import tempfile

    fake = install_fake_wandb()
    wandb_logger = fresh_logger()
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = os.path.join(tmp_dir, "wandb_config.json")
        with open(cfg_path, "w", encoding="utf-8") as handle:
            json.dump({"enabled": False}, handle)
        os.environ[wandb_logger.CONFIG_PATH_ENV_VAR] = cfg_path
        try:
            writer = wandb_logger.WandbWriter(run_name="r")
            assert writer.run is None
            assert fake.init_kwargs is None, "disabled config must not start a run"
            writer.add_scalar("losses/actor_loss", 1.0, 100)
            writer.close()
        finally:
            os.environ.pop(wandb_logger.CONFIG_PATH_ENV_VAR, None)


def test_env_var_overrides_config_file():
    import json
    import tempfile

    wandb_logger = fresh_logger()
    previous = os.environ.get("WANDB_PROJECT")
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = os.path.join(tmp_dir, "wandb_config.json")
        with open(cfg_path, "w", encoding="utf-8") as handle:
            json.dump({"project": "from-file", "api_key": "k"}, handle)
        os.environ[wandb_logger.CONFIG_PATH_ENV_VAR] = cfg_path
        os.environ["WANDB_PROJECT"] = "from-env"
        try:
            settings = wandb_logger.load_settings()
            assert settings["project"] == "from-env", settings
            assert settings["api_key"] == "k", settings
        finally:
            os.environ.pop(wandb_logger.CONFIG_PATH_ENV_VAR, None)
            if previous is None:
                os.environ.pop("WANDB_PROJECT", None)
            else:
                os.environ["WANDB_PROJECT"] = previous


def test_tensor_like_values_are_converted():
    """Values arrive as torch scalars in real runs; conversion goes through .item()."""
    class FakeTensor:
        def item(self):
            return 1.25

    install_fake_wandb()
    wandb_logger = fresh_logger()
    writer = wandb_logger.WandbWriter()
    writer.add_scalar("info/kl", FakeTensor(), 10)
    writer.flush()
    payload, _ = writer_last_payload(writer)
    assert payload["info/kl"] == 1.25, payload


def writer_last_payload(writer):
    return writer.run.calls[-1] if writer.run is not None else ({}, None)


def test_backwards_step_is_clamped_but_value_preserved():
    """wandb rejects a decreasing step; the true value still rides in the payload."""
    fake = install_fake_wandb()
    wandb_logger = fresh_logger()
    writer = wandb_logger.WandbWriter()

    writer.add_scalar("info/kl", 1.0, 500)
    writer.add_scalar("info/kl", 2.0, 100)  # out of order
    writer.flush()

    steps = [step for _, step in fake.run.calls]
    assert steps == sorted(steps), f"log steps must not decrease: {steps}"
    assert fake.run.calls[-1][0][wandb_logger.STEP_METRIC] == 100


def test_shipped_config_file_is_valid():
    """The config committed to the repo must parse and target the right project."""
    import json

    wandb_logger = fresh_logger()
    path = os.path.join(REPO_ROOT, wandb_logger.CONFIG_FILE_NAME)
    assert os.path.isfile(path), f"{wandb_logger.CONFIG_FILE_NAME} is missing"
    with open(path, encoding="utf-8") as handle:
        settings = json.load(handle)
    assert settings["project"] == "sharpa", settings
    for key in wandb_logger.DEFAULT_SETTINGS:
        assert key in settings, f"config file is missing '{key}'"


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        setup_function()
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as error:
            failures.append(name)
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
        finally:
            teardown_function()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
