#!/usr/bin/env python3
"""CLI entry point for PytorchStepFlow experiments.

Usage:
    python run_experiment.py --config configs/experiment.yaml
"""
import argparse
import asyncio
import io
import os
import sys
from datetime import datetime


class _Tee(io.TextIOBase):
    """Write to both the original stream and a log file."""

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOBase):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self):
        self._original.flush()
        self._log_file.flush()

# ---------- Environment bootstrap ----------
_CONDA_ENV = "step-perf"
_CONDA_PYTHON = os.path.expanduser(f"~/miniconda3/envs/{_CONDA_ENV}/bin/python3")

# If we're not running under the conda env Python, re-exec under it.
if os.path.realpath(sys.executable) != os.path.realpath(_CONDA_PYTHON):
    print(f"Re-executing under conda env '{_CONDA_ENV}' ({_CONDA_PYTHON})")
    os.execv(_CONDA_PYTHON, [_CONDA_PYTHON] + sys.argv)

# Scrub any Neuron venv paths from sys.path to avoid protobuf version conflicts.
sys.path = [p for p in sys.path if "/aws_neuronx_venv" not in p]

# Bootstrap step_tl packages (step_py, sim, utils, graph) onto sys.path
# so that exec'd kernel code and the simulator can import them.
_base = os.path.dirname(os.path.abspath(__file__))
_step_tl_src = os.path.join(_base, "step_tl", "src")
for _subdir in ("", "step_py", "sim", "proto"):
    _p = os.path.join(_step_tl_src, _subdir) if _subdir else _step_tl_src
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Also ensure the project root is on the path
if _base not in sys.path:
    sys.path.insert(0, _base)

from src.agent_config import ExperimentConfig
from src.loop import load_experiment_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run PytorchStepFlow optimization experiment")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (default: checkpoints/<datetime>)")
    args = parser.parse_args()

    experiment_cfg = load_experiment_config(args.config)

    # Build ExperimentConfig from experiment.yaml + llm.json
    llm_config_path = experiment_cfg.get("llm_config", "configs/llm.json")
    config = ExperimentConfig.from_files(
        llm_config_path=llm_config_path,
        iterations=experiment_cfg["iterations"],
        plans_per_kernel=experiment_cfg["plans_per_kernel"],
        implementations_per_plan=experiment_cfg["implementations_per_plan"],
        top_k=experiment_cfg["top_k"],
    )

    # Create checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        checkpoint_dir = os.path.join("checkpoints", timestamp)

    # Tee stdout/stderr to a log file in the checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "experiment.log")
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print(f"Experiment config: {args.config}")
    print(f"LLM: {config.model} @ {config.url}")
    print(f"Kernels: {[k['name'] for k in experiment_cfg['kernels']]}")
    print(f"Iterations: {config.iterations}, Plans: {config.plans_per_kernel}, "
          f"Impls/plan: {config.implementations_per_plan}, Top-K: {config.top_k}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    asyncio.run(run_experiment(experiment_cfg, config, checkpoint_dir))

    log_file.close()


if __name__ == "__main__":
    main()
