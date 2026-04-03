"""Thin loader for KernelLib bench_config.yaml — mirrors StepDB loader interface."""

import importlib.util
from pathlib import Path

import yaml

BENCH_CONFIG_PATH = Path(__file__).parent / "bench_config.yaml"
KERNELLIB_ROOT = Path(__file__).parent


def load_config():
    """Return the full config dict from bench_config.yaml."""
    with open(BENCH_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_dims(kernel_name, preset):
    """Return the dims dict for a kernel + preset name."""
    config = load_config()
    return config[kernel_name]["presets"][preset]


def list_presets(kernel_name):
    """Return list of preset names for a kernel."""
    config = load_config()
    return list(config[kernel_name]["presets"].keys())


def list_kernels():
    """Return list of all kernel names."""
    config = load_config()
    return list(config.keys())


def _load_module(rel_path, module_name):
    """Import a module from a path relative to KernelLib root."""
    full_path = KERNELLIB_ROOT / rel_path
    assert full_path.exists(), f"Module not found: {full_path}"
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_problem(kernel_name):
    """Import and return the reference (problem) module."""
    config = load_config()
    return _load_module(config[kernel_name]["problem"], f"{kernel_name}.reference")
