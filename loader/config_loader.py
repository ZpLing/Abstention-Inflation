from pathlib import Path

import yaml

#: Config-block name used by the current code -> every name that has ever
#: addressed that block, most-preferred first. YAMLs written before the paper's
#: S1–S10 numbering keep working unchanged.
BLOCK_ALIASES = {
    "main_experiment": ("main_experiment", "ab_experiment"),
    "s6_self_diagnosis": ("s6_self_diagnosis", "s5_supplementary"),
    "s9_perception_unknown_labeled_samples": (
        "s9_perception_unknown_labeled_samples",
        "s9_unknown_labeled",
        "supplementary_experiment",
    ),
    "s10_model_sweep": ("s10_model_sweep", "exp2_model_sweep"),
    "s10_difficulty": ("s10_difficulty", "exp3_difficulty"),
    "s4_random_words": ("s4_random_words", "random_perturbation_control"),
}


def block_key(config: dict, name: str) -> str:
    """Return the key under which ``name``'s block lives in ``config``.

    Prefers the current name; falls back to whichever legacy alias the YAML
    actually uses so that reads and writes hit the same dict.
    """
    for candidate in BLOCK_ALIASES.get(name, (name,)):
        if candidate in config:
            return candidate
    return name


def get_block(config: dict, name: str) -> dict:
    """Return the config block for ``name`` (``{}`` when absent)."""
    return config.get(block_key(config, name)) or {}


def load_config(config_path: str) -> dict:
    """Load + merge three sources into one flat dict:

        1. main experiment yaml   (configs/S2_unknown_option/gpt_5.4_nano/FLD.yaml)
        2. API_Config.yaml        (repo root, git-ignored) — credentials

    API_Config.yaml takes either shape. Flat:

        api_key: ...
        base_url: ...

    or nested, which also names the backbone every runner calls:

        llm:
          api_key: ...
          base_url: ...
          model: ...

    Credentials win over the experiment yaml. The model does not when the
    experiment sets ``override_model: true``, which is how several experiments
    share one gateway while pinning different backbones.

    `secrets.yaml` and an extensionless `config` are still read if present, so
    an existing checkout keeps working.

    Args:
        config_path: Path to the main configuration file (e.g., 'configs/S2_unknown_option/gpt_5.4_nano/FLD.yaml').

    Returns:
        A dictionary containing all configuration information.
    """
    # 1. Main experiment yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repo_root = Path(__file__).parent.parent

    # 2. API_Config.yaml, plus the two paths earlier checkouts used.
    secrets_path = repo_root / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)
        if secrets:
            config.update(secrets)
        print("Loaded secrets file (secrets.yaml).")

    # 3. Optional `config` file (gateway-style):
    #        llm: {api_key, base_url, model}   → the backbone every runner uses
    gateway_config = next(
        (
            p
            for p in (repo_root / "API_Config.yaml", repo_root / "config")
            if p.exists() and p.is_file()
        ),
        None,
    )
    if gateway_config is not None:
        with open(gateway_config, "r", encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}

        # ---- llm.* → top-level api_key / base_url / model_name ----
        # `override_model: true` in yaml lets a per-experiment yaml pin its own
        # model while still inheriting api_key / base_url from `config`. This
        # enables running multiple experiments in parallel under the same gateway
        # gateway but with different backbone models, each writing to its own
        # results_dir.
        llm_cfg = extra.get("llm", {}) if isinstance(extra, dict) else {}
        override_model = bool(config.get("override_model", False))
        if "api_key" in llm_cfg:
            config["api_key"] = llm_cfg["api_key"]
        if "base_url" in llm_cfg:
            config["base_url"] = llm_cfg["base_url"]
        # An empty `model` means "let each experiment yaml name its own", which
        # is what the shipped template does; writing it through would blank
        # every model_name.
        if llm_cfg.get("model") and not override_model:
            config["model_name"] = llm_cfg["model"]
        # A flat file (api_key / base_url at top level) is accepted too.
        for k in ("api_key", "base_url"):
            if k in extra and k not in llm_cfg:
                config[k] = extra[k]
        print(
            f"Loaded LLM credentials from {gateway_config.name} "
            f"(model={llm_cfg.get('model')!r}, base_url={llm_cfg.get('base_url')!r})."
        )

    if not config.get("api_key"):
        print("Warning: no api_key resolved (check API_Config.yaml).")

    return config
