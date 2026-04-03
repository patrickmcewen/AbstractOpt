"""Configuration for StepEvolve."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class KernelTarget:
    name: str
    reference_path: str
    preset: str
    dims: dict


@dataclass
class StepEvolveConfig:
    kernels: list[KernelTarget]

    # Outer loop
    num_rounds: int = 5
    oe_iterations_per_round: int = 50

    # Paths
    stepdb_path: str = "StepDB"
    kernellib_path: str = ""  # optional: path to KernelLib for target kernels
    work_dir: str = "step_evolve_output"

    # OpenEvolve defaults
    oe_num_islands: int = 3
    oe_population_size: int = 200
    oe_feature_dimensions: list[str] = field(default_factory=lambda: ["complexity"])

    # LLM
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_base: str = "http://172.17.0.1:31001/v1"
    llm_api_key: str = "None"
    llm_api_key_env: str = ""  # if set, overrides llm_api_key with env var value
    curator_model: str = "openai/gpt-oss-120b"

    # Context limits
    max_curated_examples: int = 4
    max_operator_ref_chars: int = 8000


def load_config(path: str) -> StepEvolveConfig:
    """Load StepEvolveConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())

    kernels = []
    for k in raw.pop("kernels", []):
        kernels.append(KernelTarget(**k))

    return StepEvolveConfig(kernels=kernels, **raw)
