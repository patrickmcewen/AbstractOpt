# src/agent_config.py
import json
from dataclasses import dataclass, field


@dataclass
class Experience:
    kernel_name: str
    code: str
    cycle_time: float


@dataclass
class BuildError:
    kernel_name: str
    code: str
    error_message: str
    diagnosis: str


@dataclass
class KernelState:
    kernel_name: str
    kernel_spec_path: str
    top_k_candidates: list[Experience] = field(default_factory=list)
    curated_experiences: list[Experience] = field(default_factory=list)
    diagnosed_errors: list[BuildError] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    url: str
    model: str
    api_key: str
    iterations: int
    plans_per_kernel: int
    implementations_per_plan: int
    top_k: int

    @classmethod
    def from_files(cls, llm_config_path: str, **kwargs) -> "ExperimentConfig":
        with open(llm_config_path) as f:
            llm = json.load(f)
        return cls(
            url=llm["url"],
            model=llm["model"],
            api_key=llm["api_key"],
            **kwargs,
        )
