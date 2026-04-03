"""ExperienceStore: cross-kernel knowledge base of successful STeP implementations."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Experience:
    kernel_name: str
    code: str
    cycle_time: float
    max_diff: float
    dims: dict
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        return cls(**d)


class ExperienceStore:
    def __init__(self):
        self._experiences: list[Experience] = []

    def add(self, exp: Experience) -> None:
        self._experiences.append(exp)

    def get_best(self, kernel_name: str) -> Experience | None:
        candidates = self.get_all_for_kernel(kernel_name)
        return candidates[0] if candidates else None

    def get_all_for_kernel(self, kernel_name: str) -> list[Experience]:
        matches = [e for e in self._experiences if e.kernel_name == kernel_name]
        matches.sort(key=lambda e: e.cycle_time)
        return matches

    def get_best_per_kernel(self) -> dict[str, Experience]:
        best: dict[str, Experience] = {}
        for exp in self._experiences:
            if exp.kernel_name not in best or exp.cycle_time < best[exp.kernel_name].cycle_time:
                best[exp.kernel_name] = exp
        return best

    def get_summary(self) -> list[dict]:
        best_map = self.get_best_per_kernel()
        return [
            {"kernel_name": name, "cycle_time": exp.cycle_time, "tags": exp.tags}
            for name, exp in sorted(best_map.items())
        ]

    def save(self, path: str) -> None:
        data = [exp.to_dict() for exp in self._experiences]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self._experiences = [Experience.from_dict(d) for d in data]
