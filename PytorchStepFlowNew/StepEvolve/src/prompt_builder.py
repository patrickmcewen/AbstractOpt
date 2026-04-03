"""Build system messages for OpenEvolve with curated cross-kernel context."""

from pathlib import Path

from src.experience_store import Experience

# STeP reference lives one level up from StepEvolve, in PytorchStepFlowNew/
_STEP_REFERENCE_PATH = Path(__file__).resolve().parent.parent.parent / "STeP_reference.txt"


def build_system_message(
    kernel_spec: str,
    curated_experiences: list[Experience],
    prompts_dir: str,
) -> str:
    """Build a system message containing STeP reference, critical rules, kernel spec, and curated examples."""
    step_rules = (Path(prompts_dir) / "step_rules.txt").read_text()
    step_reference = _STEP_REFERENCE_PATH.read_text()

    parts = [
        step_rules,
        "\n## Full STeP IR API Reference\n\n" + step_reference,
    ]

    parts.append(f"\n## Target Kernel (PyTorch Reference)\n\n```python\n{kernel_spec}\n```")

    if curated_experiences:
        parts.append("\n## Reference Implementations from Related Kernels")
        for exp in curated_experiences:
            parts.append(
                f"\n### {exp.kernel_name} (cycle_time: {exp.cycle_time}, tags: {', '.join(exp.tags)})\n"
                f"```python\n{exp.code}\n```"
            )

    return "\n".join(parts)
