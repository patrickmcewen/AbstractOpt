import re
from src.agent_config import BuildError, ExperimentConfig
from src.agents.common import make_agent, load_prompt, render_prompt
from src.agents.error_curator import render_diagnosed_errors
from agents import Runner
import asyncio


SYSTEM_PROMPT_PATH = "prompts/executor/system_prompt.txt"
USER_PROMPT_TEMPLATE_PATH = "prompts/executor/user_prompt_template.txt"
OPERATOR_REFERENCE_PATH = "prompts/operator_reference.txt"


def extract_code(response: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def build_executor_user_prompt(
    kernel_spec: str,
    plan: str,
    diagnosed_errors: list[BuildError],
) -> str:
    template = load_prompt(USER_PROMPT_TEMPLATE_PATH)
    operator_ref = load_prompt(OPERATOR_REFERENCE_PATH)
    return render_prompt(
        template,
        kernel_spec=kernel_spec,
        plan=plan,
        operator_reference=operator_ref,
        diagnosed_errors_context=render_diagnosed_errors(diagnosed_errors),
    )


async def generate_implementations(
    config: ExperimentConfig,
    kernel_spec: str,
    plan: str,
    diagnosed_errors: list[BuildError],
    n_implementations: int,
) -> list[str]:
    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
    agent = make_agent(config, name="Executor", system_prompt=system_prompt)
    user_prompt = build_executor_user_prompt(kernel_spec, plan, diagnosed_errors)

    results = await asyncio.gather(*[
        Runner.run(agent, user_prompt) for _ in range(n_implementations)
    ])

    codes = []
    for r in results:
        if r is None:
            continue
        code = extract_code(r.final_output)
        if code is not None:
            codes.append(code)
    return codes
