"""Load existing StepDB kernel implementations into ExperienceStore."""

import json
import re
from pathlib import Path

from src.experience_store import Experience, ExperienceStore


# Patterns to detect operator usage in STeP code
_TAG_PATTERNS = {
    "matmul": re.compile(r"BinaryMapAccum.*Matmul|MapAccumMatmul|MapAccumDynMatmul|map_accum_fn\.Matmul", re.DOTALL),
    "reduction": re.compile(r"RowWiseSum|Accum\("),
    "element_wise": re.compile(r"BinaryMap\(.*fn=(?:Mul|Add|Div|IsEqual)\("),
    "activation": re.compile(r"fn=Silu\(\)"),
    "unary": re.compile(r"UnaryMap\("),
    "tiling": re.compile(r"tile_[a-z]"),
    "broadcast": re.compile(r"Broadcast\("),
    "parallelize": re.compile(r"Parallelize\("),
    "bufferize": re.compile(r"Bufferize\(|Streamify\("),
}


def extract_op_tags(code: str) -> list[str]:
    """Scan STeP code for operator patterns and return matching tags."""
    return [tag for tag, pattern in _TAG_PATTERNS.items() if pattern.search(code)]


def seed_from_stepdb(store: ExperienceStore, stepdb_path: str) -> None:
    """Load all StepDB kernels with evaluation results into the store."""
    stepdb = Path(stepdb_path)

    # Scan all kernel directories: seed_kernels/ and solved_kernels/
    kernel_dirs = []
    for subdir_name in ("seed_kernels", "solved_kernels"):
        subdir = stepdb / subdir_name
        if subdir.is_dir():
            kernel_dirs.extend(sorted(subdir.iterdir()))
    assert kernel_dirs, f"No kernel directories found in {stepdb}"

    for kernel_dir in kernel_dirs:
        if not kernel_dir.is_dir() or kernel_dir.name.startswith("_"):
            continue

        step_impl = kernel_dir / "step_impl.py"
        if not step_impl.exists():
            continue

        code = step_impl.read_text()
        tags = extract_op_tags(code)

        # Find best result across all _work_* directories
        best_cycle = float("inf")
        best_dims = {}
        best_max_diff = 0.0

        for work_dir in kernel_dir.glob("_work_*"):
            result_file = work_dir / "result.json"
            if not result_file.exists():
                continue
            result = json.loads(result_file.read_text())
            if not result.get("success"):
                continue
            cycle = result.get("cycle_time", float("inf"))
            if cycle < best_cycle:
                best_cycle = cycle
                best_dims = result.get("dims", {})
                best_max_diff = result.get("max_diff", 0.0)

        if best_cycle == float("inf"):
            best_cycle = 0.0

        store.add(Experience(
            kernel_name=kernel_dir.name,
            code=code,
            cycle_time=best_cycle,
            max_diff=best_max_diff,
            dims=best_dims,
            tags=tags,
        ))
