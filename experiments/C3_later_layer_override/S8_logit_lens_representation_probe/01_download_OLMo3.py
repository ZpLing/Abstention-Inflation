"""
Download OLMo-3 7B checkpoints for C3 logit-lens experiment.
    Base → Instruct → RL-Zero-General (three alignment stages)
Uses ModelScope as a CDN-mirror fallback (HuggingFace XetHub direct downloads can stall).
Run on the 3090 server: python scripts/download_olmo3.py
"""

import os
from modelscope.hub.snapshot_download import snapshot_download

MODELS = {
    "olmo3-base":     "allenai/OLMo-3-1025-7B",
    "olmo3-instruct": "allenai/OLMo-3-7B-Instruct",
    "olmo3-rl-zero":  "allenai/OLMo-3-7B-RL-Zero-General",
}

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def download_model(name: str, repo_id: str):
    local_path = os.path.join(SAVE_DIR, name)
    if os.path.isdir(local_path) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(local_path)
    ):
        print(f"[skip] {name} already exists at {local_path}")
        return
    print(f"\n[downloading] {repo_id} → {local_path}")
    # ModelScope snapshot_download with local_dir writes files directly (no nesting)
    snapshot_download(
        model_id=repo_id,
        local_dir=local_path,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"[done] {name}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    for name, repo_id in MODELS.items():
        download_model(name, repo_id)
    print("\nAll models ready.")
    print("Paths:")
    for name in MODELS:
        print(f"  {name}: {os.path.join(SAVE_DIR, name)}")
