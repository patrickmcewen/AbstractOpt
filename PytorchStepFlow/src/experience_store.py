from src.agent_config import Experience


class ExperienceStore:
    def __init__(self):
        self._experiences: list[Experience] = []

    def add(self, kernel_name: str, code: str, cycle_time: float):
        self._experiences.append(Experience(kernel_name=kernel_name, code=code, cycle_time=cycle_time))

    def get_all_for_kernel(self, kernel_name: str) -> list[Experience]:
        matches = [e for e in self._experiences if e.kernel_name == kernel_name]
        matches.sort(key=lambda e: e.cycle_time)
        return matches

    def get_best_per_kernel(self) -> dict[str, Experience]:
        best: dict[str, Experience] = {}
        for e in self._experiences:
            if e.kernel_name not in best or e.cycle_time < best[e.kernel_name].cycle_time:
                best[e.kernel_name] = e
        return best

    def get_summary(self) -> list[dict]:
        best = self.get_best_per_kernel()
        return [{"kernel_name": k, "cycle_time": v.cycle_time} for k, v in best.items()]
