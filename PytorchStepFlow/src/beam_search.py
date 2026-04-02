from src.agent_config import Experience


def select_top_k(experiences: list[Experience], k: int) -> list[Experience]:
    sorted_exps = sorted(experiences, key=lambda e: e.cycle_time)
    return sorted_exps[:k]
