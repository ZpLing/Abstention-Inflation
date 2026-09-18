"""Check that every model folder under results/ and configs/ is spelled as the model's slug.

The rule (``infra.result_schema.model_slug``): a model's folder is its official
name with "-" and "/" as "_" and the case kept -- ``Olmo-3-7B-Instruct`` ->
``Olmo_3_7B_Instruct``, ``gemma-4-E2B-it`` -> ``gemma_4_E2B_it``, ``gpt-5.4-nano``
-> ``gpt_5.4_nano``. Result files are ``<dataset>_<official name>[...].json``.

This reads the git index (``git ls-files``), not the working tree: on a
case-insensitive filesystem (macOS) a folder whose case differs from the index
looks right locally and is missing on Linux, which is how the misspellings this
guards against went unnoticed.

    python infra/check_names.py    # prints every problem; exit 1 if there is one
"""

import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra.result_schema import (  # noqa: E402
    MODEL_SLUG,
    S8_CHECKPOINTS,
    S8_MODEL,
    SETTING_DIRS,
    model_slug,
    s8_checkpoint_name,
)


def tracked(prefix: str) -> list:
    out = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [p for p in out.stdout.split("\n") if p]


def main() -> int:
    problems = []

    # The slug table may fix a spelling, never change one.
    for name, slug in MODEL_SLUG.items():
        if slug != name.replace("/", "_").replace("-", "_"):
            problems.append(
                f"MODEL_SLUG[{name!r}] = {slug!r} changes the official spelling"
            )

    # configs/: the model folder and the block's model_slug are model_slug(model_name).
    official = {S8_MODEL}
    for p in tracked("configs"):
        if not p.endswith(".yaml") or p.endswith("API_Config.template.yaml"):
            continue
        cfg = yaml.safe_load((ROOT / p).read_text())
        name = cfg["model_name"]
        official.add(name)
        want = model_slug(name)
        folder = pathlib.Path(p).parts[-2]
        block = cfg[cfg["run_tasks"][0]]
        if folder != want:
            problems.append(
                f"{p}: folder {folder!r}, but model_slug({name!r}) is {want!r}"
            )
        if block.get("model_slug") != want:
            problems.append(
                f"{p}: model_slug {block.get('model_slug')!r}, but should be {want!r}"
            )

    # results/: every model folder is the slug of a model a config names, and
    # every file in it carries that model's official name (S8: a checkpoint's).
    slug_to_name = {model_slug(n): n for n in official}
    setting_dirs = set(SETTING_DIRS.values())
    setting_dirs |= {f"{d}/tfq" for d in setting_dirs} | {
        f"{d}/mcq" for d in setting_dirs
    }
    s8_names = {s8_checkpoint_name(k) for k in S8_CHECKPOINTS}
    for p in tracked("results"):
        path = pathlib.Path(p)
        rel_parent = str(path.parent.relative_to("results"))
        if rel_parent in setting_dirs:
            continue  # an aggregate next to the model folders, e.g. S11's summary.json
        folder = path.parent.name
        if folder not in slug_to_name:
            problems.append(
                f"{p}: model folder {folder!r} is no model's slug (known: {sorted(slug_to_name)})"
            )
            continue
        name = slug_to_name[folder]
        stem = path.stem
        if name == S8_MODEL:
            ok = any(
                stem.split("_", 1)[1:] == [c]
                or stem.split("_", 1)[1:] == [f"{c}_inference"]
                for c in s8_names
            )
        else:
            ok = f"_{name}" in stem
        if not ok:
            problems.append(f"{p}: file name does not carry the official name {name!r}")

    if problems:
        print(f"{len(problems)} naming problem(s):")
        for x in problems:
            print("  ", x)
        return 1
    print(
        f"names ok: {len(tracked('configs'))} config files, {len(tracked('results'))} result files, "
        f"{len(official)} models ({', '.join(sorted(official))})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
