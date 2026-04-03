from src.experience_store import ExperienceStore
from src.agent_config import Experience, ExperimentConfig
from src.agents.common import make_agent, load_prompt, render_prompt
from agents import Runner
import asyncio


SYSTEM_PROMPT_PATH = "prompts/kernel_curator/system_prompt.txt"


def build_curator_user_prompt(store: ExperienceStore, target_kernel: str) -> str:
    summary = store.get_summary()
    # Exclude the target kernel from candidates
    candidates = [s for s in summary if s["kernel_name"] != target_kernel]
    if not candidates:
        return ""
    lines = [f"Target kernel: {target_kernel}", "", "Available kernels:"]
    for c in candidates:
        lines.append(f"  - {c['kernel_name']} (best cycle time: {c['cycle_time']})")
    return "\n".join(lines)


def parse_curator_response(response: str, available_kernels: list[str]) -> list[str]:
    if response.strip().lower() == "none":
        return []
    names = [n.strip().lower() for n in response.split(",")]
    return [n for n in names if n in available_kernels]


async def curate_experiences(
    config: ExperimentConfig,
    store: ExperienceStore,
    target_kernel: str,
) -> tuple[list[Experience], str]:
    user_prompt = build_curator_user_prompt(store, target_kernel)
    if not user_prompt:
        return [], ""

    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
    agent = make_agent(config, name="KernelCurator", system_prompt=system_prompt)

    result = await Runner.run(agent, user_prompt)
    raw_output = result.final_output

    best_per_kernel = store.get_best_per_kernel()
    available = [k for k in best_per_kernel if k != target_kernel]
    selected_names = parse_curator_response(raw_output, available)

    experiences = [best_per_kernel[name] for name in selected_names if name in best_per_kernel]
    return experiences, raw_output
