import yaml
from pathlib import Path


#: Config-block name used by the current code -> every name that has ever
#: addressed that block, most-preferred first. YAMLs written before the paper's
#: S1–S10 numbering keep working unchanged.
BLOCK_ALIASES = {
    "main_experiment":     ("main_experiment", "ab_experiment"),
    "s6_self_diagnosis":   ("s6_self_diagnosis", "s5_supplementary"),
    "s9_truly_unknown":    ("s9_truly_unknown", "supplementary_experiment"),
    "s10_model_sweep":     ("s10_model_sweep", "exp2_model_sweep"),
    "s10_difficulty":      ("s10_difficulty", "exp3_difficulty"),
    "appendix_mitigation": ("appendix_mitigation", "exp4_mitigation"),
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

        1. main experiment yaml      (configs/C1_structural_trigger/TFQ_n500_GPT_5_4_nano.yaml)
        2. optional secrets.yaml     (top-level api_key / base_url)
        3. optional `config` file    (extensionless YAML at repo root, gateway
                                       schema: llm.{api_key, base_url, model})

    Precedence: experiment.yaml < secrets.yaml < config (later wins).
    Both secrets.yaml and `config` are gitignored.

    Args:
        config_path: Path to the main configuration file (e.g., 'configs/C1_structural_trigger/TFQ_n500_GPT_5_4_nano.yaml').

    Returns:
        A dictionary containing all configuration information.
    """
    # 1. Main experiment yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repo_root = Path(__file__).parent.parent

    # 2. Optional secrets.yaml (legacy path)
    secrets_path = repo_root / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)
        if secrets:
            config.update(secrets)
        print("Loaded secrets file (secrets.yaml).")

    # 3. Optional `config` file (gateway-style):
    #        llm: {api_key, base_url, model}   → the backbone every runner uses
    gateway_config = repo_root / "config"
    if gateway_config.exists() and gateway_config.is_file():
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
        if "model" in llm_cfg and not override_model:
            config["model_name"] = llm_cfg["model"]
        print(
            f"Loaded LLM credentials from `config` "
            f"(model={llm_cfg.get('model')!r}, base_url={llm_cfg.get('base_url')!r})."
        )

    if not config.get("api_key"):
        print("Warning: no api_key resolved (check secrets.yaml or `config`).")

    return config
