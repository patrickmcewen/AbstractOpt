from src.agent_config import BuildError, ExperimentConfig
from src.agents.common import make_agent, load_prompt
from agents import Runner
import asyncio


SYSTEM_PROMPT_PATH = "prompts/error_curator/system_prompt.txt"


def build_error_user_prompt(kernel_name: str, code: str, error_message: str) -> str:
    lines = [
        f"Kernel: {kernel_name}",
        "",
        "Implementation:",
        f"```python\n{code}\n```",
        "",
        f"Error:\n{error_message}",
    ]
    return "\n".join(lines)


def render_diagnosed_errors(errors: list[BuildError]) -> str:
    if not errors:
        return ""
    lines = ["# Common Build Errors to Avoid"]
    for err in errors:
        lines.append(f"- {err.diagnosis}")
    return "\n".join(lines)


async def diagnose_error(
    config: ExperimentConfig,
    kernel_name: str,
    code: str,
    error_message: str,
) -> BuildError:
    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
    agent = make_agent(config, name="ErrorCurator", system_prompt=system_prompt)

    user_prompt = build_error_user_prompt(kernel_name, code, error_message)
    result = await Runner.run(agent, user_prompt)

    return BuildError(
        kernel_name=kernel_name,
        code=code,
        error_message=error_message,
        diagnosis=result.final_output,
    )
