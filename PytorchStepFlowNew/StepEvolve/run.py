#!/usr/bin/env python3
"""StepEvolve CLI: cross-task evolutionary STeP implementation.

Usage:
    python run.py --auto-all --rounds 3 --iterations 20
    python run.py --kernellib ../KernelLib --rounds 5
    python run.py --list-kernels
    python run.py --config configs/default.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure local src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure openevolve is importable
_OE_PATH = str(Path(__file__).resolve().parent.parent / "openevolve")
if _OE_PATH not in sys.path:
    sys.path.insert(0, _OE_PATH)

from src.config import StepEvolveConfig, KernelTarget, load_config


def _get_loader(path: str):
    """Import the loader module from a StepDB or KernelLib directory."""
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    import importlib
    spec = importlib.util.spec_from_file_location("loader", str(Path(resolved) / "loader.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_kernels(path: str, label: str = ""):
    """List available kernels from a StepDB or KernelLib directory."""
    loader = _get_loader(path)
    header = f"Available kernels ({label}):" if label else "Available kernels:"
    print(header)
    for name in loader.list_kernels():
        presets = ", ".join(loader.list_presets(name))
        print(f"  {name}: [{presets}]")


def auto_kernels(path: str, preset: str = "small") -> list[KernelTarget]:
    """Build KernelTarget list from all kernels in a directory using a given preset."""
    loader = _get_loader(path)
    resolved = Path(path).resolve()
    config = loader.load_config()
    targets = []
    for name in loader.list_kernels():
        presets = loader.list_presets(name)
        p = preset if preset in presets else presets[0]
        dims = loader.get_dims(name, p)
        ref_path = config[name]["problem"]
        targets.append(KernelTarget(
            name=name,
            reference_path=str(resolved / ref_path),
            preset=p,
            dims=dims,
        ))
    return targets


def main():
    parser = argparse.ArgumentParser(description="StepEvolve: cross-task evolutionary STeP implementation")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--rounds", type=int, help="Override number of rounds")
    parser.add_argument("--iterations", type=int, help="Override OpenEvolve iterations per round")
    parser.add_argument("--list-kernels", action="store_true", help="List available kernels")
    parser.add_argument("--auto-all", action="store_true", help="Auto-generate targets from all StepDB kernels")
    parser.add_argument("--kernellib", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "KernelLib"),
                        help="Path to KernelLib directory (target kernels without STeP impls)")
    parser.add_argument("--preset", type=str, default="small", help="Preset for auto targets (default: small)")
    parser.add_argument("--stepdb", type=str, default=str(Path(__file__).resolve().parent.parent / "StepDB"),
                        help="Path to StepDB directory (seed kernels with STeP impls)")
    parser.add_argument("--work-dir", type=str, help="Override work directory")
    parser.add_argument("--only", type=str, help="Comma-separated list of kernel names to target (filter)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.list_kernels:
        list_kernels(args.stepdb, "StepDB")
        kernellib = Path(args.kernellib)
        if kernellib.exists():
            print()
            list_kernels(str(kernellib), "KernelLib")
        return

    # Load or create config
    if args.config:
        se_config = load_config(args.config)
    else:
        se_config = StepEvolveConfig(kernels=[], stepdb_path=args.stepdb)

    se_config.kernellib_path = str(Path(args.kernellib).resolve())

    # Auto-populate kernels
    if args.auto_all:
        # StepDB kernels as targets (re-evolve)
        se_config.kernels = auto_kernels(se_config.stepdb_path, args.preset)
    elif not se_config.kernels:
        # Default: target KernelLib kernels (the main use case)
        kernellib = Path(args.kernellib)
        if kernellib.exists():
            se_config.kernels = auto_kernels(str(kernellib), args.preset)
            logging.info(f"Loaded {len(se_config.kernels)} target kernels from KernelLib")
        else:
            se_config.kernels = auto_kernels(se_config.stepdb_path, args.preset)

    # Filter to specific kernels if requested
    if args.only:
        allowed = {n.strip() for n in args.only.split(",")}
        se_config.kernels = [k for k in se_config.kernels if k.name in allowed]
        assert se_config.kernels, f"No kernels matched --only={args.only}"

    # Apply CLI overrides
    if args.rounds:
        se_config.num_rounds = args.rounds
    if args.iterations:
        se_config.oe_iterations_per_round = args.iterations
    if args.work_dir:
        se_config.work_dir = args.work_dir

    logging.info(f"StepEvolve starting: {len(se_config.kernels)} kernels, "
                 f"{se_config.num_rounds} rounds, "
                 f"{se_config.oe_iterations_per_round} iterations/round")

    from src.orchestrator import run
    store = asyncio.run(run(se_config))

    # Print summary
    print(f"\n{'='*60}")
    print("StepEvolve Complete")
    print(f"{'='*60}")
    for entry in store.get_summary():
        print(f"  {entry['kernel_name']}: cycle_time={entry['cycle_time']}, tags={entry['tags']}")


if __name__ == "__main__":
    main()
