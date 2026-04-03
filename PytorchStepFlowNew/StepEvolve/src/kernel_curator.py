"""LLM-based kernel curator: selects relevant cross-kernel examples."""

import asyncio
from pathlib import Path

from openai import OpenAI

from src.experience_store import Experience, ExperienceStore


def build_curator_user_prompt(
    store: ExperienceStore,
    target_kernel: str,
    target_reference: str,
    candidate_references: dict[str, str],
) -> str:
    """Build the user prompt with target + candidate PyTorch references.

    Args:
        store: ExperienceStore with kernel summaries
        target_kernel: name of the kernel we're implementing
        target_reference: PyTorch reference source code for the target
        candidate_references: {kernel_name: reference_source} for each candidate
    """
    summary = store.get_summary()
    candidates = [s for s in summary if s["kernel_name"] != target_kernel]
    if not candidates:
        return ""

    lines = [
        f"## Target Kernel: {target_kernel}",
        "",
        "```python",
        target_reference.strip(),
        "```",
        "",
        "## Available Kernels",
        "",
    ]

    for c in candidates:
        name = c["kernel_name"]
        tags_str = ", ".join(c["tags"]) if c["tags"] else "none"
        ref_code = candidate_references.get(name, "")

        lines.append(f"### {name} (cycle_time: {c['cycle_time']}, tags: {tags_str})")
        if ref_code:
            lines.append("")
            lines.append("```python")
            lines.append(ref_code.strip())
            lines.append("```")
        lines.append("")

    return "\n".join(lines)


def parse_curator_response(response: str | None, available_kernels: list[str]) -> list[str]:
    """Parse the curator's comma-separated response into valid kernel names."""
    if not response or response.strip().lower() == "none":
        return []
    names = [n.strip().lower() for n in response.split(",")]
    return [n for n in names if n in available_kernels]


def load_candidate_references(
    store: ExperienceStore,
    stepdb_path: str,
    exclude_kernel: str,
    kernellib_path: str = "",
) -> dict[str, str]:
    """Load PyTorch reference source code for all kernels in the store except the target.

    Searches both StepDB and KernelLib for reference files.
    """
    import importlib.util

    refs = {}

    # Load configs from available sources
    sources = []  # list of (root_path, config_dict)

    stepdb = Path(stepdb_path).resolve()
    loader_path = stepdb / "loader.py"
    if loader_path.exists():
        spec = importlib.util.spec_from_file_location("stepdb_loader", str(loader_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sources.append((stepdb, mod.load_config()))

    if kernellib_path:
        klib = Path(kernellib_path).resolve()
        loader_path = klib / "loader.py"
        if loader_path.exists():
            spec = importlib.util.spec_from_file_location("kernellib_loader", str(loader_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sources.append((klib, mod.load_config()))

    for entry in store.get_summary():
        name = entry["kernel_name"]
        if name == exclude_kernel:
            continue
        for root, config in sources:
            if name in config:
                ref_path = root / config[name]["problem"]
                if ref_path.exists():
                    refs[name] = ref_path.read_text()
                    break
    return refs


async def curate_experiences(
    store: ExperienceStore,
    target_kernel: str,
    target_reference: str,
    model: str,
    api_base: str,
    api_key: str,
    prompts_dir: str,
    stepdb_path: str,
    max_examples: int = 4,
    kernellib_path: str = "",
) -> list[Experience]:
    """Call the curator LLM to select relevant cross-kernel experiences."""
    candidate_refs = load_candidate_references(store, stepdb_path, target_kernel, kernellib_path)
    user_prompt = build_curator_user_prompt(store, target_kernel, target_reference, candidate_refs)
    if not user_prompt:
        return []

    system_prompt = (Path(prompts_dir) / "curator_system.txt").read_text()

    def _call_llm():
        client = OpenAI(api_key=api_key, base_url=api_base)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    raw_output = await asyncio.to_thread(_call_llm)
    best_per_kernel = store.get_best_per_kernel()
    available = [k for k in best_per_kernel if k != target_kernel]
    selected_names = parse_curator_response(raw_output, available)

    return [best_per_kernel[name] for name in selected_names[:max_examples] if name in best_per_kernel]
