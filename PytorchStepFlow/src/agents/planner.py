from src.agent_config import Experience, ExperimentConfig
from src.agents.common import make_agent, load_prompt, render_prompt
from agents import Runner
import asyncio


SYSTEM_PROMPT_PATH = "prompts/planner/system_prompt.txt"
USER_PROMPT_TEMPLATE_PATH = "prompts/planner/user_prompt_template.txt"
OPERATOR_REFERENCE_PATH = "prompts/operator_reference.txt"


def render_top_k_context(candidates: list[Experience]) -> str:
    if not candidates:
        return ""
    lines = ["## Prior Successful Implementations"]
    for exp in candidates:
        lines.append(f"\n### {exp.kernel_name} (cycle time: {exp.cycle_time})")
        lines.append(f"```python\n{exp.code}\n```")
    return "\n".join(lines)


def render_curated_context(experiences: list[Experience]) -> str:
    if not experiences:
        return ""
    lines = ["## Relevant Implementations From Other Kernels"]
    for exp in experiences:
        lines.append(f"\n### {exp.kernel_name} (cycle time: {exp.cycle_time})")
        lines.append(f"```python\n{exp.code}\n```")
    return "\n".join(lines)


def build_planner_user_prompt(
    kernel_spec: str,
    top_k_candidates: list[Experience],
    curated_experiences: list[Experience],
) -> str:
    template = load_prompt(USER_PROMPT_TEMPLATE_PATH)
    operator_ref = load_prompt(OPERATOR_REFERENCE_PATH)
    return render_prompt(
        template,
        kernel_spec=kernel_spec,
        operator_reference=operator_ref,
        top_k_context=render_top_k_context(top_k_candidates),
        curated_experiences_context=render_curated_context(curated_experiences),
    )


async def generate_plans(
    config: ExperimentConfig,
    kernel_spec: str,
    top_k_candidates: list[Experience],
    curated_experiences: list[Experience],
    n_plans: int,
) -> list[str]:
    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
    agent = make_agent(config, name="Planner", system_prompt=system_prompt)
    user_prompt = build_planner_user_prompt(kernel_spec, top_k_candidates, curated_experiences)

    results = await asyncio.gather(*[
        Runner.run(agent, user_prompt) for _ in range(n_plans)
    ])

    return [r.final_output for r in results if r is not None]
