import asyncio
import importlib.util
import json
import os
import time
from datetime import datetime

import yaml

from src.agent_config import ExperimentConfig, KernelState, Experience
from src.beam_search import select_top_k
from src.evaluation import evaluate_kernel
from src.experience_store import ExperienceStore
from src.agents.common import load_prompt
from src.agents.kernel_curator import curate_experiences
from src.agents.kernel_curator import SYSTEM_PROMPT_PATH as CURATOR_SYSTEM_PATH
from src.agents.kernel_curator import build_curator_user_prompt
from src.agents.planner import generate_plans
from src.agents.planner import SYSTEM_PROMPT_PATH as PLANNER_SYSTEM_PATH
from src.agents.planner import build_planner_user_prompt
from src.agents.executor import generate_implementations
from src.agents.executor import SYSTEM_PROMPT_PATH as EXECUTOR_SYSTEM_PATH
from src.agents.executor import build_executor_user_prompt
from src.agents.error_curator import diagnose_error
from src.agents.error_curator import SYSTEM_PROMPT_PATH as ERROR_CURATOR_SYSTEM_PATH
from src.agents.error_curator import build_error_user_prompt


def _save_prompts(directory: str, system_prompt: str, user_prompt: str):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "system_prompt.txt"), "w") as f:
        f.write(system_prompt)
    with open(os.path.join(directory, "user_prompt.txt"), "w") as f:
        f.write(user_prompt)


def load_experiment_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_dims(bench_name: str, preset: str) -> dict:
    bench_config_path = os.path.join(os.path.dirname(__file__), "..", "StepBench", "bench_config.yaml")
    with open(bench_config_path) as f:
        bench = yaml.safe_load(f)
    assert bench_name in bench, f"Unknown benchmark: {bench_name}"
    assert preset in bench[bench_name]["presets"], (
        f"Unknown preset '{preset}' for {bench_name}. Available: {list(bench[bench_name]['presets'].keys())}"
    )
    return bench[bench_name]["presets"][preset]


def load_problem_module(problem_path: str):
    abs_path = os.path.abspath(problem_path)
    assert os.path.exists(abs_path), f"Problem file not found: {abs_path}"
    spec = importlib.util.spec_from_file_location("problem_mod", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _run_kernel_iteration(
    config: ExperimentConfig,
    kc: dict,
    ks: KernelState,
    problem_mod,
    dims: dict,
    store: ExperienceStore,
    checkpoint_dir: str,
    iteration: int,
    n_plans: int,
    n_impls: int,
    top_k: int,
    eval_lock: asyncio.Lock,
):
    """Run one kernel's full curate → plan → execute → evaluate → beam pipeline."""
    name = kc["name"]
    kernel_spec = open(kc["problem"]).read()
    iter_dir = os.path.join(checkpoint_dir, name, f"iter_{iteration}")
    plans_dir = os.path.join(iter_dir, "plans")
    impls_dir = os.path.join(iter_dir, "implementations")
    os.makedirs(plans_dir, exist_ok=True)
    os.makedirs(impls_dir, exist_ok=True)

    print(f"\n--- {name} (iter {iteration}) ---")

    # Step 1: Curate cross-kernel experiences
    curator_user_prompt = build_curator_user_prompt(store, name)
    if curator_user_prompt:
        _save_prompts(
            os.path.join(iter_dir, "curator"),
            load_prompt(CURATOR_SYSTEM_PATH),
            curator_user_prompt,
        )
    ks.curated_experiences, curator_output = await curate_experiences(config, store, name)
    if curator_output:
        with open(os.path.join(iter_dir, "curator", "response.txt"), "w") as f:
            f.write(curator_output)
    print(f"  [{name}] Curated {len(ks.curated_experiences)} cross-kernel experiences")

    # Step 2: Generate plans
    _save_prompts(
        os.path.join(iter_dir, "planner"),
        load_prompt(PLANNER_SYSTEM_PATH),
        build_planner_user_prompt(kernel_spec, ks.top_k_candidates, ks.curated_experiences),
    )
    plans = await generate_plans(
        config, kernel_spec, ks.top_k_candidates, ks.curated_experiences, n_plans,
    )
    for i, plan in enumerate(plans):
        with open(os.path.join(plans_dir, f"plan_{i}.txt"), "w") as f:
            f.write(plan)
    print(f"  [{name}] Generated {len(plans)} plans")

    # Step 3: Generate implementations for ALL plans in parallel
    executor_system_prompt = load_prompt(EXECUTOR_SYSTEM_PATH)
    for plan_idx, plan in enumerate(plans):
        _save_prompts(
            os.path.join(impls_dir, f"plan_{plan_idx}_prompts"),
            executor_system_prompt,
            build_executor_user_prompt(kernel_spec, plan, ks.diagnosed_errors),
        )

    all_plan_codes = await asyncio.gather(*[
        generate_implementations(config, kernel_spec, plan, ks.diagnosed_errors, n_impls)
        for plan in plans
    ])
    print(f"  [{name}] Generated implementations for {len(plans)} plans")

    # Step 4: Evaluate all implementations (sequential — simulator uses os.chdir)
    iter_successes = []
    iter_failures = 0
    failures_to_diagnose = []
    error_curator_system_prompt = load_prompt(ERROR_CURATOR_SYSTEM_PATH)

    for plan_idx, codes in enumerate(all_plan_codes):
        for impl_idx, code in enumerate(codes):
            work_dir = os.path.join(impls_dir, f"plan_{plan_idx}_impl_{impl_idx}")
            # Lock evaluation so concurrent kernels don't collide on os.chdir
            async with eval_lock:
                result = evaluate_kernel(code, problem_mod, dims, work_dir)

            if result.success:
                store.add(name, code, result.cycle_time)
                iter_successes.append(result)
                print(f"    [{name}] plan_{plan_idx}_impl_{impl_idx}: SUCCESS (cycles={result.cycle_time})")
            else:
                iter_failures += 1
                print(f"    [{name}] plan_{plan_idx}_impl_{impl_idx}: FAIL at {result.stage}")
                if result.stage in ("exec", "simulate") and result.error_message:
                    _save_prompts(
                        os.path.join(work_dir, "error_curator"),
                        error_curator_system_prompt,
                        build_error_user_prompt(name, code, result.error_message),
                    )
                    failures_to_diagnose.append((work_dir, code, result))

    # Step 5: Diagnose ALL errors in parallel
    if failures_to_diagnose:
        diagnoses = await asyncio.gather(*[
            diagnose_error(config, name, code, result.error_message)
            for _, code, result in failures_to_diagnose
        ])
        for (work_dir, _, _), diagnosis in zip(failures_to_diagnose, diagnoses):
            with open(os.path.join(work_dir, "error_curator", "diagnosis.txt"), "w") as f:
                f.write(diagnosis.diagnosis)
        ks.diagnosed_errors.extend(diagnoses)
        print(f"  [{name}] Diagnosed {len(diagnoses)} errors in parallel")

    # Step 6: Beam search — combine this iter's successes with prior top_k
    new_experiences = [
        Experience(kernel_name=name, code=r.code, cycle_time=r.cycle_time)
        for r in iter_successes
    ]
    all_candidates = ks.top_k_candidates + new_experiences
    ks.top_k_candidates = select_top_k(all_candidates, top_k)

    best = ks.top_k_candidates[0].cycle_time if ks.top_k_candidates else None
    print(f"  [{name}] Results: {len(iter_successes)} successes, {iter_failures} failures, best={best}")


async def run_experiment(
    experiment_cfg: dict,
    config: ExperimentConfig,
    checkpoint_dir: str,
):
    """Run the full optimization loop."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(checkpoint_dir, "experiment.yaml"), "w") as f:
        yaml.dump(experiment_cfg, f)

    # Initialize global state
    store = ExperienceStore()
    kernel_states: dict[str, KernelState] = {}
    problem_modules = {}
    dims_map = {}

    for kc in experiment_cfg["kernels"]:
        name = kc["name"]
        kernel_states[name] = KernelState(
            kernel_name=name,
            kernel_spec_path=kc["problem"],
        )
        problem_modules[name] = load_problem_module(kc["problem"])
        bench_name = os.path.splitext(os.path.basename(kc["problem"]))[0]
        dims_map[name] = resolve_dims(bench_name, kc["preset"])

    iterations = config.iterations
    n_plans = config.plans_per_kernel
    n_impls = config.implementations_per_plan
    top_k = config.top_k

    # Lock for evaluate_kernel which uses os.chdir (process-global)
    eval_lock = asyncio.Lock()

    for iteration in range(iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # Run ALL kernels in parallel within each iteration
        await asyncio.gather(*[
            _run_kernel_iteration(
                config, kc, kernel_states[kc["name"]],
                problem_modules[kc["name"]], dims_map[kc["name"]],
                store, checkpoint_dir, iteration,
                n_plans, n_impls, top_k, eval_lock,
            )
            for kc in experiment_cfg["kernels"]
        ])

        elapsed = time.time() - iter_start
        print(f"\nIteration {iteration} complete in {elapsed:.1f}s")

    # Final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    for name, ks in kernel_states.items():
        if ks.top_k_candidates:
            best = ks.top_k_candidates[0].cycle_time
            print(f"  {name}: best cycle_time = {best}")
        else:
            print(f"  {name}: no successful implementations")
