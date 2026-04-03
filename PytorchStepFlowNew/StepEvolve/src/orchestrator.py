"""Outer loop orchestrator: schedules kernels, runs OpenEvolve, harvests results.

Key lifecycle: KernelLib target → evolve → on success, graduate to StepDB.
Graduated kernels become context for solving remaining targets.
"""

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

from src.config import StepEvolveConfig, KernelTarget
from src.experience_store import Experience, ExperienceStore
from src.seed_loader import seed_from_stepdb, extract_op_tags
from src.seed_builder import build_seed
from src.evaluator_factory import write_evaluator
from src.prompt_builder import build_system_message
from src.kernel_curator import curate_experiences

logger = logging.getLogger(__name__)

# Ensure openevolve is importable
_OE_PATH = str(Path(__file__).resolve().parent.parent.parent / "openevolve")
if _OE_PATH not in sys.path:
    sys.path.insert(0, _OE_PATH)

from openevolve.api import _run_evolution_async
from openevolve.config import Config, LLMModelConfig


def _resolve_api_key(se_config: StepEvolveConfig) -> str:
    """Resolve API key: env var overrides direct value."""
    if se_config.llm_api_key_env:
        return os.environ.get(se_config.llm_api_key_env, se_config.llm_api_key)
    return se_config.llm_api_key


def build_oe_config(se_config: StepEvolveConfig, system_message: str) -> Config:
    """Build an OpenEvolve Config from StepEvolve settings."""
    oe_cfg = Config()

    oe_cfg.max_iterations = se_config.oe_iterations_per_round
    oe_cfg.diff_based_evolution = True
    oe_cfg.max_code_length = 8000
    oe_cfg.language = "python"

    # LLM
    api_key = _resolve_api_key(se_config)
    oe_cfg.llm.api_base = se_config.llm_api_base
    oe_cfg.llm.api_key = api_key
    oe_cfg.llm.temperature = 0.7
    oe_cfg.llm.max_tokens = 4096
    oe_cfg.llm.models = [LLMModelConfig(name=se_config.llm_model, weight=1.0)]
    oe_cfg.llm.update_model_params({
        "api_base": se_config.llm_api_base,
        "api_key": api_key,
        "temperature": 0.7,
        "max_tokens": 4096,
    })

    # Prompt -- inject curated context via system message
    oe_cfg.prompt.system_message = system_message
    oe_cfg.prompt.num_top_programs = 3
    oe_cfg.prompt.num_diverse_programs = 2

    # Database / MAP-Elites
    oe_cfg.database.num_islands = se_config.oe_num_islands
    oe_cfg.database.population_size = se_config.oe_population_size
    oe_cfg.database.feature_dimensions = list(se_config.oe_feature_dimensions)

    # Evaluator -- MUST be sequential (STeP simulator uses os.chdir)
    oe_cfg.evaluator.cascade_evaluation = True
    oe_cfg.evaluator.cascade_thresholds = [0.2, 0.5, 0.9]
    oe_cfg.evaluator.timeout = 120
    oe_cfg.evaluator.parallel_evaluations = 1

    return oe_cfg


def _load_reference_spec(kernel: KernelTarget, se_config: StepEvolveConfig) -> str:
    """Load the PyTorch reference source code for a kernel."""
    ref_path = Path(kernel.reference_path)
    if not ref_path.is_absolute():
        ref_path = Path(se_config.stepdb_path) / kernel.reference_path
    assert ref_path.exists(), f"Reference not found: {ref_path}"
    return ref_path.read_text()


# ---------------------------------------------------------------------------
# Graduation: on success, copy kernel into StepDB so it becomes reusable context
# ---------------------------------------------------------------------------

def graduate_to_stepdb(
    kernel: KernelTarget,
    step_impl_code: str,
    se_config: StepEvolveConfig,
) -> None:
    """Copy a solved kernel's reference + STeP implementation into StepDB.

    Creates:
      StepDB/kernels/{name}/reference.py  (copied from KernelLib)
      StepDB/kernels/{name}/step_impl.py  (the evolved code)
    Appends entry to StepDB/bench_config.yaml.
    """
    stepdb = Path(se_config.stepdb_path).resolve()
    kernel_dir = stepdb / "kernels" / kernel.name
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference
    src_ref = Path(kernel.reference_path).resolve()
    assert src_ref.exists(), f"Reference not found: {src_ref}"
    shutil.copy2(str(src_ref), str(kernel_dir / "reference.py"))

    # Write step_impl
    (kernel_dir / "step_impl.py").write_text(step_impl_code)

    # Append to bench_config.yaml
    config_path = stepdb / "bench_config.yaml"
    assert config_path.exists(), f"bench_config.yaml not found: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if kernel.name not in config:
        config[kernel.name] = {
            "problem": f"kernels/{kernel.name}/reference.py",
            "step_impl": f"kernels/{kernel.name}/step_impl.py",
            "params": list(kernel.dims.keys()),
            "presets": {kernel.preset: dict(kernel.dims)},
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Graduated {kernel.name}/{kernel.preset} to StepDB")


# ---------------------------------------------------------------------------
# Per-kernel round
# ---------------------------------------------------------------------------

async def run_kernel_round(
    kernel: KernelTarget,
    store: ExperienceStore,
    se_config: StepEvolveConfig,
    round_idx: int,
) -> bool:
    """Run one round of OpenEvolve for a single kernel. Returns True if new success found."""
    logger.info(f"Round {round_idx}: evolving {kernel.name} / {kernel.preset}")

    prompts_dir = str(Path(__file__).resolve().parent.parent / "prompts")
    work_dir = Path(se_config.work_dir) / kernel.name / f"round_{round_idx}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load target kernel reference (used by both curator and prompt builder)
    kernel_spec = _load_reference_spec(kernel, se_config)

    # 2. Curate cross-kernel examples (curator sees PyTorch references for informed selection)
    api_key = _resolve_api_key(se_config)
    curated = await curate_experiences(
        store=store,
        target_kernel=kernel.name,
        target_reference=kernel_spec,
        model=se_config.curator_model,
        api_base=se_config.llm_api_base,
        api_key=api_key,
        prompts_dir=prompts_dir,
        stepdb_path=se_config.stepdb_path,
        max_examples=se_config.max_curated_examples,
        kernellib_path=se_config.kernellib_path,
    )
    logger.info(f"  Curated {len(curated)} cross-kernel examples")

    # 3. Build seed program
    seed_code = build_seed(store, kernel.name, kernel.dims)
    seed_path = str(work_dir / "seed.py")
    Path(seed_path).write_text(seed_code)

    # 4. Build evaluator (now takes dims + reference_path directly)
    evaluator_path = str(work_dir / "evaluator.py")
    write_evaluator(
        kernel.name, kernel.preset, kernel.dims,
        kernel.reference_path, se_config.stepdb_path, evaluator_path,
    )

    # 5. Build system message with curated context
    system_message = build_system_message(kernel_spec, curated, prompts_dir)

    # 6. Build OpenEvolve config
    oe_cfg = build_oe_config(se_config, system_message)

    # 7. Run OpenEvolve (use async version since we're already in an event loop)
    logger.info(f"  Running OpenEvolve for {se_config.oe_iterations_per_round} iterations")
    result = await _run_evolution_async(
        initial_program=seed_path,
        evaluator=evaluator_path,
        config=oe_cfg,
        iterations=se_config.oe_iterations_per_round,
        output_dir=str(work_dir / "oe_output"),
        cleanup=False,
    )

    # 8. Harvest results
    if result.best_program and result.metrics.get("success", False):
        best_code = result.best_code
        cycles = result.metrics.get("cycles", 0.0)
        max_diff = result.metrics.get("max_diff", 0.0)

        store.add(Experience(
            kernel_name=kernel.name,
            code=best_code,
            cycle_time=cycles,
            max_diff=max_diff,
            dims=kernel.dims,
            tags=extract_op_tags(best_code),
        ))

        # Graduate to StepDB
        graduate_to_stepdb(kernel, best_code, se_config)

        logger.info(f"  SUCCESS: {kernel.name} cycles={cycles}")
        return True

    logger.info(f"  No correct implementation found (best_score={result.best_score:.4f})")
    return False


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------

async def run(se_config: StepEvolveConfig) -> ExperienceStore:
    """Main orchestration loop.

    - Seeds ExperienceStore from existing StepDB implementations
    - Each round, attempts all unsolved kernels
    - On success, graduates kernel to StepDB (skip in future rounds)
    - Terminates early if all kernels are solved
    """
    store = ExperienceStore()

    # Seed from StepDB (existing implementations become context)
    stepdb_path = str(Path(se_config.stepdb_path).resolve())
    seed_from_stepdb(store, stepdb_path)
    logger.info(f"Seeded store with {len(store.get_summary())} kernels from StepDB")

    Path(se_config.work_dir).mkdir(parents=True, exist_ok=True)

    # Track which kernels have been solved
    solved = set()

    for round_idx in range(se_config.num_rounds):
        pending = [k for k in se_config.kernels if k.name not in solved]
        if not pending:
            logger.info("All kernels solved! Terminating early.")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_idx + 1} / {se_config.num_rounds} "
                     f"({len(pending)} pending, {len(solved)} solved)")
        logger.info(f"{'='*60}")

        for kernel in pending:
            success = await run_kernel_round(kernel, store, se_config, round_idx)
            if success:
                solved.add(kernel.name)

        # Save store after each round
        store_path = str(Path(se_config.work_dir) / "experience_store.json")
        store.save(store_path)
        logger.info(f"Store saved: {len(store.get_summary())} kernels "
                     f"({len(solved)} solved, {len(se_config.kernels) - len(solved)} remaining)")

    return store
