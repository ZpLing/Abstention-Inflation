#!/usr/bin/env python3
"""Dispatcher for the Abstention Inflation experiments.

Every setting the paper reports is reached from here. The first argument names
the setting; ``all`` is the only way to run everything. Below the setting the
levels are read top-down -- ``--sub-setting``, ``--model``, ``--dataset`` -- and a
level left out means every value the paper reports for that setting (``all``
at any level says the same explicitly). A setting with more than one experiment
splits into sub-settings; the local settings run in steps, chosen with
``--step``, the same option under its other name.

    python main.py --list                       every setting, its sub-settings or steps, what --model means there
    python main.py all                          every gateway setting on every reported cell
    python main.py all --stage analyze          print every setting's numbers
    python main.py S2                           one setting (collected with its S1 pair)
    python main.py S2 --model gemini-3.1-flash-lite --dataset FLD
    python main.py S4 --sub-setting random_words --model deepseek-v4-flash
    python main.py S9 --sub-setting persistence --model gpt-5.4-nano --dataset FOLIO
    python main.py S2 --model qwen3-max         any model the gateway serves
    python main.py S8 --step logit_lens --model sft
    python main.py S10 --step size_alignment --model gemma-4-E4B-it --model-path <checkout>
    python main.py all --limit 4 --results-root /tmp/smoke --dry-run
    python main.py --config configs/C1_structural_trigger/S1_S3_TFQ_GPT_5_4_nano.yaml

Two kinds of setting. The gateway settings (S1-S6, S9, S10 temperature, S11)
call the OpenAI-compatible endpoint named in API_Config.yaml; they are what
``all`` runs. The local settings (S7, S8, S10 temperature_local and
size_alignment) load a checkpoint from disk; ``all`` prints the command for
each and moves on, and they run when named.

A model outside the three the paper reports borrows gpt-5.4-nano's YAML for its
credentials, keeps its own name, and writes under its own slug
(``results/<setting>/.../<model_slug>/``), so every setting runs on any model
the gateway serves. S1, S2, S3 and S5 come out of one paired pass per cell:
asking for one of them collects what it needs (S1 is always paired with S2).

``--config`` runs one experiment YAML the way this script always has. The
YAML's ``run_tasks`` picks the runner (main_experiment, s6_self_diagnosis,
s9_perception_unknown_labeled_samples, s10_model_sweep); ``--model``,
``--dataset``, ``--limit`` and ``--results-root`` still apply on top of it.
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infra.result_schema import (  # noqa: E402
    S8_CHECKPOINTS,
    cell_path,
    model_slug,
    results_dir,
)
from loader.config_loader import block_key, load_config  # noqa: E402

# =============================================================================
# What the paper reports
# =============================================================================

#: gateway model -> the tag its YAMLs carry under configs/.
GATEWAY_MODELS: Dict[str, str] = {
    "gpt-5.4-nano": "GPT_5_4_nano",
    "gemini-3.1-flash-lite": "Gemini_3_1_Flash_Lite",
    "deepseek-v4-flash": "DeepSeek_V4_Flash",
}
#: A model with no YAML of its own borrows this one's and overrides the name.
TEMPLATE_MODEL = "gpt-5.4-nano"
GATEWAY = tuple(GATEWAY_MODELS)

TFQ_DATASETS = ("FLD", "FOLIO")
MCQ_DATASETS = ("ARC", "MedQA", "MMLU", "LogiQA")
ALL_DATASETS = TFQ_DATASETS + MCQ_DATASETS
#: Collected by one pass of infra/paired_pass.py per (model, dataset) cell.
PAIRED_SETTINGS = ("S1", "S2", "S3", "S5")

S8_REPORTED = ("base", "sft", "rl_zero")
GEMMA_TAGS = tuple(
    f"gemma-4-{size}{arm}"
    for size in ("E2B", "E4B", "26B-A4B", "31B")
    for arm in ("", "-it")
)
OLMO_TAG = "Olmo-3-7B-Instruct"
TEMPERATURES = (0.0, 0.3, 0.7, 1.0, 1.5, 2.0)

CONFIGS = ROOT / "configs"
_C1 = ROOT / "experiments" / "C1_structural_trigger"
_C2 = ROOT / "experiments" / "C2_deny_yet_capable"
_C3 = ROOT / "experiments" / "C3_later_layer_override"
_C4 = ROOT / "experiments" / "C4_stable_bias"
SCRIPT: Dict[str, Path] = {
    "S2/analyze": _C1 / "S2_unknown_option_added" / "analyze_S2.py",
    "S3/analyze": _C1 / "S3_question_format_ablation" / "analyze_S3.py",
    "S4/synonyms": _C1 / "S4_word_content_ablation" / "run_S4_synonyms.py",
    "S4/analyze": _C1 / "S4_word_content_ablation" / "analyze_S4.py",
    "S5/analyze": _C2 / "S5_without_unknown_option_rerun" / "analyze_S5.py",
    "S7/run": _C3 / "S7_reasoning_traces_evaluation" / "run_S7_reasoning_traces_evaluation.py",
    "S8/download": _C3 / "S8_logit_lens_representation_probe" / "run_S8_step1_download.py",
    "S8/inference": _C3 / "S8_logit_lens_representation_probe" / "run_S8_step2_inference.py",
    "S8/logit_lens": _C3 / "S8_logit_lens_representation_probe" / "run_S8_step3_logit_lens.py",
    "S9/persistence": _C4 / "S9_stability" / "run_S9_persistence_across_repeats.py",
    "S10/temperature": _C4 / "S10_factor_analysis" / "run_S10_temperature.py",
    "S10/local_sweep": _C4 / "S10_factor_analysis" / "run_S10_local_sweep.py",
    "S10/analyze_temperature": _C4 / "S10_factor_analysis" / "analyze_S10_temperature.py",
    "S10/analyze_size_alignment": _C4 / "S10_factor_analysis" / "analyze_S10_size_alignment.py",
    "S10/analyze_difficulty": _C4 / "S10_factor_analysis" / "analyze_S10_difficulty.py",
    "S11/run": _C4 / "S11_positional_biases" / "run_S11_positional_biases.py",
    "S11/analyze": _C4 / "S11_positional_biases" / "analyze_S11.py",
}


@dataclass(frozen=True)
class Part:
    """One experiment inside a setting, and what each level below it ranges over."""

    name: str
    doc: str
    datasets: Tuple[str, ...]
    models: Tuple[str, ...] = GATEWAY
    model_kind: str = "gateway model"
    local: bool = False  # loads a checkpoint from disk; `all` prints, does not run
    analyze_only: bool = False  # nothing to collect


@dataclass(frozen=True)
class Setting:
    key: str
    title: str
    parts: Tuple[Part, ...]


SETTINGS: Dict[str, Setting] = {
    "S1": Setting("S1", "Baseline", (
        Part("baseline", "the task's own labels, no extra option; collected in the same pass as S2", ALL_DATASETS),
    )),
    "S2": Setting("S2", "Unknown option added", (
        Part("unknown_option", "the manipulation, scored per item against S1", ALL_DATASETS),
    )),
    "S3": Setting("S3", "Question format ablation", (
        Part("question_format", "the S2 ternary rendered as A / B / C; TFQ only", TFQ_DATASETS),
    )),
    "S4": Setting("S4", "Word content ablation", (
        Part("synonyms", "\"I don't know\" / \"Indeterminate\" in the third slot; reads the S2 cell", TFQ_DATASETS),
        Part("random_words", "\"Triangular\" / \"Cerulean\" in the third slot; reads the S1 and S2 cells", TFQ_DATASETS),
    )),
    "S5": Setting("S5", "Without-Unknown rerun", (
        Part("rerun", "the items S2 abstained on, asked again without the option; collected inside the S1/S2 pass", ALL_DATASETS),
    )),
    "S6": Setting("S6", "Self-diagnosis", (
        Part("self_diagnosis", "the model attributes its own abstention; reads the S1, S2 and S5 cells", TFQ_DATASETS),
    )),
    "S7": Setting("S7", "Reasoning traces", (
        Part("reasoning_traces", "NLI probe over the stored S1/S2 traces; loads the DeBERTa NLI encoder", TFQ_DATASETS, local=True),
    )),
    "S8": Setting("S8", "Logit lens", (
        Part("download", "fetch the Olmo-3-7B checkpoints into models/", ("FLD",), tuple(S8_CHECKPOINTS), "checkpoint key", local=True),
        Part("inference", "S1/S2 answers of the instruct checkpoint, the partition every probe is scored on", ("FLD",), ("instruct",), "checkpoint key", local=True),
        Part("logit_lens", "per-layer UNKNOWN logit on that partition, one checkpoint per call", ("FLD",), S8_REPORTED, "checkpoint key", local=True),
    )),
    "S9": Setting("S9", "Stability", (
        Part("perception", "S1/S2/S3 on the Unknown-labeled items", TFQ_DATASETS),
        Part("persistence", "the same S2 prompt re-drawn three times on the abstaining items; reads the S2 cell", TFQ_DATASETS),
    )),
    "S10": Setting("S10", "Factor analysis", (
        Part("temperature", "S2 at six temperatures on a gateway model whose endpoint applies it", TFQ_DATASETS, ("gemini-3.1-flash-lite",)),
        Part("temperature_local", "the same sweep on a local checkpoint", TFQ_DATASETS, (OLMO_TAG,), "checkpoint tag", local=True),
        Part("size_alignment", "four Gemma-4 scales, base and IT, at T=0; one checkpoint per call", TFQ_DATASETS, GEMMA_TAGS, "checkpoint tag", local=True),
        Part("difficulty", "Abs Rate against FLD proof depth; a stratification of the S2 cells", ("FLD",), analyze_only=True),
    )),
    "S11": Setting("S11", "Positional biases", (
        Part("positional_biases", "the abstain verb in slot A / B / C of the S2 prompt", TFQ_DATASETS),
    )),
}


# =============================================================================
# Steps: what one invocation resolves to
# =============================================================================


@dataclass
class Options:
    limit: Optional[int] = None
    results_root: Optional[Path] = None  # absolute, or None for results/
    model_path: Optional[str] = None
    dry_run: bool = False
    everything: bool = False  # `all` was given

    @property
    def root(self) -> Path:
        return self.results_root or Path("results")


@dataclass
class Step:
    label: str
    shown: str  # the command --dry-run prints; runnable as printed
    argv: Optional[List[str]] = None  # a script, run from the repo root
    coro: Optional[Callable[[], Coroutine[Any, Any, None]]] = None  # in-process
    pre: Optional[Callable[[], None]] = None  # raises when an input is missing
    skip: Optional[str] = None  # printed instead of running
    note: Optional[str] = None  # informational only


def _rel(p: Any) -> str:
    p = Path(str(p))
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _script_cmd(argv: Sequence[Any]) -> str:
    return "python " + " ".join(shlex.quote(_rel(a)) for a in argv)


def _main_cmd(settings, part, models, datasets, o: Options, stage=None, flag="--sub-setting") -> str:
    words = ["python main.py", *settings]
    if part:
        words += [flag, part]
    if models:
        words += ["--model", *(shlex.quote(m) for m in models)]
    if datasets:
        words += ["--dataset", *datasets]
    if stage:
        words += ["--stage", stage]
    if o.limit:
        words += ["--limit", str(o.limit)]
    if o.results_root:
        words += ["--results-root", shlex.quote(str(o.results_root))]
    if o.model_path:
        words += ["--model-path", shlex.quote(o.model_path)]
    return " ".join(words)


def _root_flag(flag: str, o: Options) -> List[str]:
    return [flag, str(o.results_root)] if o.results_root else []


def _limit_flag(flag: str, o: Options) -> List[str]:
    return [flag, str(o.limit)] if o.limit else []


def script_step(label: str, argv: Sequence[Any], pre=None) -> Step:
    return Step(label, _script_cmd(argv), argv=[str(a) for a in argv], pre=pre)


# ---- configs -----------------------------------------------------------------


def yaml_for(kind: str, model: str, dataset: Optional[str] = None) -> Path:
    """The YAML that collects ``kind`` for ``model``; a new model borrows the template's."""
    tag = GATEWAY_MODELS.get(model, GATEWAY_MODELS[TEMPLATE_MODEL])
    return {
        "tfq": CONFIGS / "C1_structural_trigger" / f"S1_S3_TFQ_{tag}.yaml",
        "mcq": CONFIGS / "C1_structural_trigger" / f"S1_S2_{dataset}_{tag}.yaml",
        "s4": CONFIGS / "C1_structural_trigger" / f"S4_random_words_{tag}.yaml",
        "s6": CONFIGS / "C2_deny_yet_capable" / f"S6_{tag}.yaml",
        "s9": CONFIGS / "C4_stable_bias" / f"S9_Perception_Unknown_labeled_Samples_{tag}.yaml",
    }[kind]


def configure(
    yaml_path: Path,
    block_name: str,
    model: str,
    *,
    datasets: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    results_root: Optional[Path] = None,
    **block_overrides: Any,
) -> dict:
    """Load a YAML and point it at ``model``, the datasets, the limit and the root."""
    config = load_config(str(yaml_path))
    config["model_name"] = model
    block = config.setdefault(block_key(config, block_name), None) or {}
    config[block_key(config, block_name)] = block
    block["model_slug"] = model_slug(model)
    if datasets is not None:
        block["datasets"] = list(datasets)
    if results_root is not None:
        block["results_root"] = str(results_root)
    if limit:
        block["sample_limits"] = {ds: limit for ds in block.get("datasets", [])}
    block.update(block_overrides)
    return config


def require_cells(setting: str, model: str, datasets: Sequence[str], o: Options, hint: str):
    """A pre-check: the cells a follow-up setting reads must be on disk."""

    def check() -> None:
        missing = []
        for ds in datasets:
            p = ROOT / cell_path(setting, ds, model, model_slug(model), "tf", o.root)
            if not p.exists():
                missing.append(_rel(p))
        if missing:
            raise FileNotFoundError(
                f"{setting} cell(s) not found: {', '.join(missing)}. Run `{hint}` first."
            )

    return check


# ---- in-process runners --------------------------------------------------------


async def _run(config: dict, runner_cls) -> None:
    """Build the runner on fresh handlers, run it, and close the HTTP client
    while the event loop is still open. A client collected after
    ``asyncio.run()`` has returned schedules its own shutdown on the closed
    loop and prints a traceback for every cell."""
    from infra.evaluator import Evaluator
    from infra.llm_handler import LLMHandler
    from loader.data_handler import DataHandler

    llm = LLMHandler(config)
    try:
        await runner_cls(config, DataHandler(config), llm, Evaluator()).run()
    finally:
        await llm.client.close()


async def run_paired(config: dict) -> None:
    from infra.paired_pass import ABRunner

    await _run(config, ABRunner)


async def run_s6(config: dict) -> None:
    from experiments.C2_deny_yet_capable.S6_self_diagnosis.run_S6_self_diagnosis import (
        S6SelfDiagnosisRunner,
    )

    await _run(config, S6SelfDiagnosisRunner)


async def run_s9_perception(config: dict) -> None:
    from experiments.C4_stable_bias.S9_stability.run_S9_perception_unknown_labeled_samples import (
        PerceptionUnknownLabeledSamplesRunner,
    )

    await _run(config, PerceptionUnknownLabeledSamplesRunner)


async def run_s4_random_words(config: dict) -> None:
    from experiments.C1_structural_trigger.S4_word_content_ablation.run_S4_random_words import (
        run_experiment,
    )

    await run_experiment(config)


# =============================================================================
# Resolving the levels
# =============================================================================


def _is_all(requested: Optional[Sequence[str]]) -> bool:
    return not requested or [r.lower() for r in requested] == ["all"]


def resolve_models(part: Part, requested: Optional[Sequence[str]], key: str) -> List[str]:
    if _is_all(requested):
        return list(part.models)
    if part.model_kind == "checkpoint key":
        bad = [m for m in requested if m not in part.models]
        if bad:
            raise SystemExit(
                f"{key}/{part.name}: --model takes a {part.model_kind} "
                f"({', '.join(part.models)}), not {', '.join(bad)}."
            )
    return list(dict.fromkeys(requested))


def resolve_datasets(part: Part, requested: Optional[Sequence[str]]) -> List[str]:
    if _is_all(requested):
        return list(part.datasets)
    return [d for d in part.datasets if d in requested]


def selected_parts(setting: Setting, requested: Optional[Sequence[str]]) -> List[Part]:
    if _is_all(requested):
        return list(setting.parts)
    return [p for p in setting.parts if p.name in requested]


# ---- collect ---------------------------------------------------------------------


def paired_steps(paired: List[str], model_req, ds_req, o: Options) -> List[Step]:
    """S1/S2/S3/S5 share one pass per cell; run it once with the union of what was asked."""
    models = resolve_models(SETTINGS["S2"].parts[0], model_req, "S2")
    union = list(dict.fromkeys(d for k in paired for d in SETTINGS[k].parts[0].datasets))
    datasets = union if _is_all(ds_req) else [d for d in union if d in ds_req]
    if not datasets:
        return [Step("+".join(paired), "-", note=f"none of the requested datasets belongs to {'/'.join(paired)} ({', '.join(union)})")]

    settings = {"S1"} if paired == ["S1"] else {"S1", "S2"}
    if "S3" in paired:
        settings.add("S3")
    run_s5 = "S5" in paired

    steps: List[Step] = []
    for model in models:
        tfq = [d for d in datasets if d in TFQ_DATASETS]
        mcq = [d for d in datasets if d in MCQ_DATASETS]
        cells = ([("tfq", tfq)] if tfq else []) + [("mcq", [d]) for d in mcq]
        for kind, ds in cells:
            active = [s for s in ("S1", "S2", "S3") if s in settings and not (s == "S3" and kind == "mcq")]

            def make(kind=kind, model=model, ds=tuple(ds), active=tuple(active)):
                return run_paired(
                    configure(
                        yaml_for(kind, model, ds[0]),
                        "main_experiment",
                        model,
                        datasets=ds,
                        limit=o.limit,
                        results_root=o.results_root,
                        settings=list(active),
                        run_s5_rerun=run_s5,
                    )
                )

            what = "+".join(active + (["S5"] if run_s5 else []))
            steps.append(Step(f"{what} · {model} · {'/'.join(ds)}", _main_cmd(paired, None, [model], ds, o), coro=make))
    return steps


def _s4_synonyms(part, models, datasets, o: Options) -> List[Step]:
    return [
        script_step(
            f"S4/synonyms · {m}",
            [SCRIPT["S4/synonyms"], "--models", m, "--datasets", *datasets, *_root_flag("--results-root", o), *_limit_flag("--limit", o)],
            pre=require_cells("S2", m, datasets, o, f"python main.py S2 --model {m}"),
        )
        for m in models
    ]


def _s4_random_words(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for m in models:

        def make(m=m):
            return run_s4_random_words(
                configure(yaml_for("s4", m), "s4_random_words", m, datasets=datasets, results_root=o.results_root)
            )

        steps.append(Step(
            f"S4/random_words · {m}",
            _main_cmd(["S4"], "random_words", [m], datasets, o),
            coro=make,
            pre=require_cells("S2", m, datasets, o, f"python main.py S2 --model {m}"),
        ))
    return steps


def _s6(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for m in models:

        def make(m=m):
            return run_s6(configure(yaml_for("s6", m), "s6_self_diagnosis", m, datasets=datasets, results_root=o.results_root))

        # S5 is written only for items S2 abstained on, so a cell with no
        # abstentions has no S5 file and S6 has nothing to ask; S2 is the input.
        steps.append(Step(
            f"S6 · {m}",
            _main_cmd(["S6"], None, [m], datasets, o),
            coro=make,
            pre=require_cells("S2", m, datasets, o, f"python main.py S5 --model {m}"),
        ))
    return steps


def _s9_perception(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for m in models:

        def make(m=m):
            return run_s9_perception(
                configure(
                    yaml_for("s9", m),
                    "s9_perception_unknown_labeled_samples",
                    m,
                    datasets=datasets,
                    limit=o.limit,
                    results_root=o.results_root,
                )
            )

        steps.append(Step(f"S9/perception · {m}", _main_cmd(["S9"], "perception", [m], datasets, o), coro=make))
    return steps


def _s9_persistence(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for m in models:
        slug = model_slug(m)
        for ds in datasets:
            steps.append(script_step(
                f"S9/persistence · {m} · {ds}",
                [
                    SCRIPT["S9/persistence"],
                    "--summary", cell_path("S2", ds, m, slug, "tf", o.root),
                    "--dataset", ds,
                    "--model", m,
                    "--n_repeats", "3",
                    "--config", yaml_for("tfq", m),
                    "--out", results_dir("S9/Persistence_Across_Repeats", o.root) / slug / f"{ds}_{m}.json",
                ],
                pre=require_cells("S2", m, [ds], o, f"python main.py S2 --model {m} --dataset {ds}"),
            ))
    return steps


def _s10_temperature(part, models, datasets, o: Options) -> List[Step]:
    return [
        script_step(
            f"S10/temperature · {m}",
            [SCRIPT["S10/temperature"], "--model", m, "--datasets", *datasets, *_root_flag("--results-root", o), *_limit_flag("--limit", o)],
        )
        for m in models
    ]


def _one_checkpoint(models: Sequence[str], o: Options) -> Optional[str]:
    """The local sweeps load one checkpoint per call; say so when that is not what was asked."""
    if not o.model_path:
        return "pass --model <tag> --model-path <checkout>, one checkpoint per call"
    if len(models) != 1:
        return "one --model per --model-path; name the checkpoint the path holds"
    return None


def _s10_temperature_local(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for tag in models:
        for T in TEMPERATURES:
            st = script_step(
                f"S10/temperature_local · {tag} · T={T}",
                [
                    SCRIPT["S10/local_sweep"],
                    "--model_path", o.model_path or f"<checkout of {tag}>",
                    "--model_tag", tag,
                    "--use_chat_template",
                    "--settings", "S2",
                    "--n_per_class", "250",
                    "--max_new_tokens", "8192",
                    "--batch_size", "8",
                    "--top_k", "20",
                    "--temperature", str(T),
                    "--datasets", *datasets,
                    "--out_dir", results_dir("S10/temperature", o.root) / model_slug(tag),
                ],
            )
            st.skip = _one_checkpoint(models, o)
            steps.append(st)
    return steps


def _s10_size_alignment(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for tag in models:
        st = script_step(
            f"S10/size_alignment · {tag}",
            [
                SCRIPT["S10/local_sweep"],
                "--model_path", o.model_path or f"<checkout of {tag}>",
                "--model_tag", tag,
                *(["--use_chat_template"] if tag.endswith("-it") else []),
                "--n_per_class", "250",
                "--max_new_tokens", "3072",
                "--batch_size", "8",
                "--datasets", *datasets,
                "--out_dir", results_dir("S10/size_alignment", o.root),
            ],
        )
        st.skip = _one_checkpoint(models, o)
        steps.append(st)
    return steps


def _s11(part, models, datasets, o: Options) -> List[Step]:
    return [
        script_step(
            f"S11 · {m} · {ds}",
            [
                SCRIPT["S11/run"],
                "--model", m,
                "--dataset", ds,
                "--positions", "A", "B", "C",
                "--unified-labels",
                *_limit_flag("--sample-limit", o),
                *_root_flag("--results-root", o),
            ],
        )
        for m in models
        for ds in datasets
    ]


def _s7(part, models, datasets, o: Options) -> List[Step]:
    def pre():
        for m in models:
            require_cells("S2", m, datasets, o, f"python main.py S2 --model {m}")()

    return [script_step(
        "S7 · " + ", ".join(models),
        [SCRIPT["S7/run"], "--models", *models, "--datasets", *datasets, *_root_flag("--results-root", o)],
        pre=pre,
    )]


def _s8_download(part, models, datasets, o: Options) -> List[Step]:
    return [script_step("S8/download · " + ", ".join(models), [SCRIPT["S8/download"], "--models", *models])]


def _s8_inference(part, models, datasets, o: Options) -> List[Step]:
    argv = [SCRIPT["S8/inference"], *(["--model_path", o.model_path] if o.model_path else []), *_root_flag("--results-root", o)]
    return [script_step("S8/inference · instruct", argv)]


def _s8_logit_lens(part, models, datasets, o: Options) -> List[Step]:
    steps = []
    for ckpt in models:
        argv = [SCRIPT["S8/logit_lens"], "--ckpt", ckpt]
        if o.model_path and len(models) == 1:
            argv += ["--model_path", o.model_path]
        argv += _root_flag("--results-root", o)
        st = script_step(f"S8/logit_lens · {ckpt}", argv)
        if o.model_path and len(models) != 1:
            st.skip = "one --model per --model-path; name the checkpoint the path holds"
        steps.append(st)
    return steps


BUILDERS: Dict[Tuple[str, str], Callable[..., List[Step]]] = {
    ("S4", "synonyms"): _s4_synonyms,
    ("S4", "random_words"): _s4_random_words,
    ("S6", "self_diagnosis"): _s6,
    ("S7", "reasoning_traces"): _s7,
    ("S8", "download"): _s8_download,
    ("S8", "inference"): _s8_inference,
    ("S8", "logit_lens"): _s8_logit_lens,
    ("S9", "perception"): _s9_perception,
    ("S9", "persistence"): _s9_persistence,
    ("S10", "temperature"): _s10_temperature,
    ("S10", "temperature_local"): _s10_temperature_local,
    ("S10", "size_alignment"): _s10_size_alignment,
    ("S11", "positional_biases"): _s11,
}

#: Dependency order: the paired pass first, then what reads its cells, then the
#: independent settings, then the local ones.
COLLECT_ORDER = ("S4", "S6", "S9", "S10", "S11", "S7", "S8")


def part_steps(setting: Setting, part: Part, model_req, ds_req, o: Options) -> List[Step]:
    key = setting.key
    tag = f"{key}/{part.name}"
    named = part.name if len(setting.parts) > 1 else None  # a single-experiment setting runs by name alone
    if part.analyze_only:
        return [Step(tag, _main_cmd([key], named, [], [], o, stage="analyze"),
                     note="nothing to collect: this sub-setting stratifies the S2 cells already on disk; run it with --stage analyze")]
    datasets = resolve_datasets(part, ds_req)
    if not datasets:
        return [Step(tag, "-", note=f"none of the requested datasets belongs to {tag} ({', '.join(part.datasets)})")]
    if part.local and o.everything:
        return [Step(tag, _main_cmd([key], named, [], [], Options(), flag=_part_flag(part)),
                     skip="loads a checkpoint from disk; `all` leaves it out, run it by name")]
    models = resolve_models(part, model_req, key)
    return BUILDERS[(key, part.name)](part, models, datasets, o)


def build_collect(keys: List[str], part_req, model_req, ds_req, o: Options) -> List[Step]:
    steps: List[Step] = []
    paired = [k for k in PAIRED_SETTINGS if k in keys]
    if paired:
        steps += paired_steps(paired, model_req, ds_req, o)
    for key in (k for k in COLLECT_ORDER if k in keys):
        setting = SETTINGS[key]
        for part in selected_parts(setting, part_req):
            steps += part_steps(setting, part, model_req, ds_req, o)
    return steps


# ---- analyze ---------------------------------------------------------------------


def build_analyze(keys: List[str], part_req, model_req, ds_req, o: Options) -> List[Step]:
    steps: List[Step] = []
    gateway = resolve_models(SETTINGS["S2"].parts[0], model_req, "S2")
    if "S1" in keys or "S2" in keys:
        steps.append(script_step("S1+S2 · analyze_S2", [SCRIPT["S2/analyze"], *_root_flag("--results_dirs", o)]))
    if "S3" in keys:
        steps.append(script_step("S3 · analyze_S3", [SCRIPT["S3/analyze"], *_root_flag("--results-root", o), "--models", *gateway]))
    if "S4" in keys:
        steps.append(script_step("S4 · analyze_S4", [SCRIPT["S4/analyze"], *_root_flag("--results-root", o), "--models", *gateway]))
    if "S5" in keys:
        steps.append(script_step("S5 · analyze_S5", [SCRIPT["S5/analyze"], *_root_flag("--results_root", o)]))
    for key in ("S6", "S7", "S8", "S9"):
        if key in keys:
            steps.append(Step(f"{key} · analyze", "-", note="no separate analyzer: the runner prints this setting's metrics and stores them in its result file"))
    if "S10" in keys:
        for part in selected_parts(SETTINGS["S10"], part_req):
            if part.name in ("temperature", "temperature_local"):
                for m in resolve_models(part, model_req, "S10"):
                    steps.append(script_step(f"S10/{part.name} · analyze · {m}", [SCRIPT["S10/analyze_temperature"], "--model", m, *_root_flag("--results-root", o)]))
            elif part.name == "size_alignment":
                steps.append(script_step("S10/size_alignment · analyze", [SCRIPT["S10/analyze_size_alignment"], *_root_flag("--results-root", o)]))
            elif part.name == "difficulty":
                steps.append(script_step("S10/difficulty · analyze", [SCRIPT["S10/analyze_difficulty"], *_root_flag("--results-root", o), "--models", *gateway]))
    if "S11" in keys:
        steps.append(script_step(
            "S11 · analyze_S11",
            [SCRIPT["S11/analyze"], "--result-dir", results_dir("S11", o.root), "--models", *gateway, *_limit_flag("--expect-n", o)],
        ))
    return steps


# =============================================================================
# Running the plan
# =============================================================================


def execute(steps: List[Step], o: Options) -> int:
    if not steps:
        print("Nothing to do.")
        return 0
    print(f"{len(steps)} step(s){' -- dry run, nothing is called' if o.dry_run else ''}")
    outcome: List[Tuple[Step, str]] = []
    for i, st in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {st.label}\n    $ {st.shown}")
        if st.note:
            print(f"    note: {st.note}")
            outcome.append((st, "info"))
            continue
        if st.skip:
            print(f"    skipped: {st.skip}")
            outcome.append((st, "skipped"))
            continue
        if o.dry_run:
            outcome.append((st, "planned"))
            continue
        status = "ok"
        try:
            if st.pre:
                st.pre()
            if st.argv is not None:
                rc = subprocess.run([sys.executable, *st.argv], cwd=ROOT).returncode
                if rc != 0:
                    status = f"failed (exit {rc})"
            else:
                asyncio.run(st.coro())
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 -- one failing cell must not stop the rest
            status = f"failed ({type(exc).__name__}: {exc})"
        print(f"    -> {status}")
        outcome.append((st, status))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for st, status in outcome:
        head = status.split(" (")[0]
        print(f"  {head:<8} {st.label}")
        if head == "failed":
            print(f"           {status}")
    failed = [st for st, status in outcome if status.startswith("failed")]
    if failed:
        print(f"\n{len(failed)} step(s) failed.")
        return 1
    return 0


def _part_flag(p: Part) -> str:
    """The local settings run in steps; the gateway ones split into sub-settings."""
    return "--step" if p.local else "--sub-setting"


def _runs(p: Part) -> str:
    return "analysis only" if p.analyze_only else ("local checkpoint" if p.local else "gateway")


def print_list() -> None:
    print("Setting -> --sub-setting (or --step) -> --model -> --dataset. A level left out means")
    print("every value listed for it; `all` at a level says the same. `python main.py all` runs")
    print("the gateway rows; local rows run when named. --stage analyze prints the numbers.\n")
    for s in SETTINGS.values():
        single = len(s.parts) == 1
        print(f"{s.key:<4} {s.title}" + (f"   [{_runs(s.parts[0])}]" if single else ""))
        for p in s.parts:
            if not single:
                print(f"     {_part_flag(p) + ' ' + p.name:<32} {_runs(p)}")
            print(f"       {p.doc}")
            print(f"       --model ({p.model_kind}): {' | '.join(p.models)}")
            print(f"       --dataset: {' '.join(p.datasets)}")
        print()


# ---- --config: one YAML, as before ------------------------------------------------

#: pre-rename run_tasks name -> current name.
LEGACY_TASK_ALIASES = {
    "ab_experiment": "main_experiment",
    "supplementary_experiment": "s9_perception_unknown_labeled_samples",
    "s9_unknown_labeled": "s9_perception_unknown_labeled_samples",
    "s5_supplementary": "s6_self_diagnosis",
    "exp2_model_sweep": "s10_model_sweep",
}

CONFIG_BLOCKS = (
    "main_experiment",
    "s6_self_diagnosis",
    "s9_perception_unknown_labeled_samples",
    "s4_random_words",
    "s10_model_sweep",
)


def _resolve_tasks(config: dict) -> list:
    tasks = []
    for raw in config.get("run_tasks", []) or []:
        if raw in LEGACY_TASK_ALIASES:
            new = LEGACY_TASK_ALIASES[raw]
            print(f"[main] run_tasks: {raw!r} is the pre-rename name; running {new!r}.")
            tasks.append(new)
        else:
            tasks.append(raw)
    return tasks


async def _dispatch(config: dict) -> None:
    tasks = _resolve_tasks(config)
    if not tasks:
        print("[main] run_tasks is empty -- nothing to do.")
        return

    if "main_experiment" in tasks:
        print("\n===== S1 / S2 / S3 + S5 rerun =====")
        await run_paired(config)

    if "s9_perception_unknown_labeled_samples" in tasks:
        print("\n===== S9 Perception of Unknown-labeled Samples =====")
        await run_s9_perception(config)

    if "s6_self_diagnosis" in tasks:
        print("\n===== S6 self-diagnosis =====")
        await run_s6(config)

    if "s10_model_sweep" in tasks:
        print("\n===== S10 alignment & model-size sweep =====")
        from experiments.C4_stable_bias.S10_factor_analysis.run_S10_size_alignment import (
            ModelSweepRunner,
        )

        await _run(config, ModelSweepRunner)


def run_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = None if _is_all(args.model) else args.model
    if model and len(model) != 1:
        raise SystemExit("--config runs one YAML on one model; give a single --model.")
    if model:
        config["model_name"] = model[0]
    for name in CONFIG_BLOCKS:
        key = block_key(config, name)
        if key not in config:
            continue
        block = config[key] or {}
        config[key] = block
        if model:
            block["model_slug"] = model_slug(model[0])
        if not _is_all(args.dataset):
            block["datasets"] = [d for d in block.get("datasets", []) if d in args.dataset]
        if args.results_root:
            block["results_root"] = str(Path(args.results_root).resolve())
        if args.limit:
            block["sample_limits"] = {ds: args.limit for ds in block.get("datasets", [])}
    if args.dry_run:
        print(f"would run run_tasks={_resolve_tasks(config)} from {args.config} "
              f"on model={config.get('model_name')!r}")
        return 0
    asyncio.run(_dispatch(config))
    return 0


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Abstention Inflation settings: a setting (or `all`), then --sub-setting / --step, --model, --dataset.",
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("setting", nargs="*", metavar="SETTING",
                   help="S1 ... S11 (several allowed) or `all`. Required unless --list or --config.")
    p.add_argument("--sub-setting", "--step", dest="sub_setting", nargs="+", metavar="NAME",
                   help="which sub-setting(s) of a setting with more than one, or which step(s) of a local setting; default all of them")
    p.add_argument("--model", nargs="+", metavar="MODEL",
                   help="gateway model name(s), or for S8 / local S10 the checkpoint key or tag; default every reported one")
    p.add_argument("--dataset", nargs="+", metavar="DATASET",
                   help="dataset name(s) within the setting's own list; default every one")
    p.add_argument("--stage", choices=("collect", "analyze", "both"), default="collect",
                   help="collect the cells (default), print the numbers, or both")
    p.add_argument("--limit", type=int, metavar="N", help="items per cell, for a smoke run")
    p.add_argument("--results-root", metavar="DIR", help="read and write under DIR instead of results/")
    p.add_argument("--model-path", metavar="PATH", help="local checkout for S8 and the local S10 steps")
    p.add_argument("--dry-run", action="store_true", help="print every command without calling anything")
    p.add_argument("--list", action="store_true", help="show every setting, its sub-settings or steps, and their levels")
    p.add_argument("--config", metavar="YAML", help="run one experiment YAML instead of naming a setting")
    return p


def main() -> int:
    # Step headers must land in a redirected log before the runner's own
    # output, not when the buffer happens to fill.
    sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_list()
        return 0
    if args.config:
        if args.setting:
            raise SystemExit("--config runs one YAML; give either a setting or --config, not both.")
        return run_config(args)
    if not args.setting:
        parser.print_help()
        print("\nName a setting (S1 ... S11) or `all`.", file=sys.stderr)
        return 2

    requested = [s.upper() if s.lower() != "all" else "all" for s in args.setting]
    everything = "all" in requested
    keys = list(SETTINGS) if everything else list(dict.fromkeys(requested))
    unknown = [k for k in keys if k not in SETTINGS]
    if unknown:
        raise SystemExit(f"unknown setting(s): {', '.join(unknown)}. Choose from {', '.join(SETTINGS)} or `all`.")

    if not _is_all(args.sub_setting):
        have = {p.name for k in keys for p in SETTINGS[k].parts}
        bad = [p for p in args.sub_setting if p not in have]
        if bad:
            raise SystemExit(f"{', '.join(bad)}: no such sub-setting or step in {', '.join(keys)}. Choose from: {', '.join(sorted(have))}.")
    if not _is_all(args.dataset):
        bad = [d for d in args.dataset if d not in ALL_DATASETS]
        if bad:
            raise SystemExit(f"--dataset {', '.join(bad)}: unknown. Datasets: {', '.join(ALL_DATASETS)}.")

    o = Options(
        limit=args.limit,
        results_root=Path(args.results_root).resolve() if args.results_root else None,
        model_path=args.model_path,
        dry_run=args.dry_run,
        everything=everything,
    )
    steps: List[Step] = []
    if args.stage in ("collect", "both"):
        steps += build_collect(keys, args.sub_setting, args.model, args.dataset, o)
    if args.stage in ("analyze", "both"):
        steps += build_analyze(keys, args.sub_setting, args.model, args.dataset, o)
    try:
        return execute(steps, o)
    except KeyboardInterrupt:
        print("\n[main] interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
