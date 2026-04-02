from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
from src.agent_config import ExperimentConfig


def make_client(config: ExperimentConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.url, api_key=config.api_key, timeout=600)


def make_agent(config: ExperimentConfig, name: str, system_prompt: str) -> Agent:
    client = make_client(config)
    model = OpenAIChatCompletionsModel(model=config.model, openai_client=client)
    return Agent(name=name, instructions=system_prompt, model=model)


def load_prompt(path: str) -> str:
    with open(path) as f:
        return f.read()


def render_prompt(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result
