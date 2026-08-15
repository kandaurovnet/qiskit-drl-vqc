#!/usr/bin/env python3
"""
Regenerate everything under ``results_drl/``: the four-arm benchmark at several
VQC depths, with and without reward shaping.

This is the driver that produced the committed sweeps. It shells out to
run_experiment.py once per (depth, shaping) cell rather than importing it, so a
crash in one cell cannot poison the others and each cell's stdout lands in its
own log next to its results.

    ./run_sweep.py                       # full sweep: depths 1,2,3,5 x shaping on/off
    ./run_sweep.py --n-layers 3 5        # only these depths
    ./run_sweep.py --shaping on          # only the shaped half
    ./run_sweep.py --dry-run             # print the plan, run nothing
    ./run_sweep.py --resume              # skip cells whose benchmark.json exists

Every cell runs under CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
OMP_WAIT_POLICY=PASSIVE, which this script sets itself -- the workload is 8 KB
of strictly sequential gates, so a GPU and extra cores both cost more than they
return. See the README section "Runs are slow?".

Cost warning: the full default sweep is 8 cells x 4 arms x 10 seeds x 100k
steps. The quantum arms dominate at roughly 10-11 minutes per seed, so budget
on the order of a day. Narrow it with --n-layers/--shaping/--seeds first.

The classical-small arm is NOT a fixed width here. run_experiment.py sizes it
from the depth of the cell, as the narrowest hidden layer whose parameter count
just exceeds that depth's VQC. That is the whole point of the arm -- if the
classical net were the smaller of the two, the quantum arm would hold exactly
the parameter advantage the arm exists to remove.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Matches the committed sweeps. results_drl/nlayer<N>[-no-rwshp]/
DEPTHS = (1, 2, 3, 5)
SEEDS = tuple(range(10))
TOTAL_STEPS = 100_000

# The workload is latency-bound, not compute-bound: a GPU sits idle between
# ~3 us kernel launches, and OpenMP threads spin-wait around 8 KB tensors.
RUN_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OMP_WAIT_POLICY": "PASSIVE",
    "MKL_NUM_THREADS": "1",
}

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "results_drl"


def cell_dir(root: Path, n_layers: int, shaping: bool) -> Path:
    """Output directory for one sweep cell, matching the committed naming."""
    return root / f"nlayer{n_layers}{'' if shaping else '-no-rwshp'}"


def build_command(args, n_layers: int, shaping: bool, out_dir: Path) -> list[str]:
    cmd = [
        # -u: the cell's stdout is a file, so without it Python block-buffers
        # and sweep.log stays empty for hours -- no way to see progress or spot
        # a cell that is failing.
        sys.executable, "-u", str(HERE / "run_experiment.py"),
        "--arms", *args.arms,
        "--seeds", *(str(s) for s in args.seeds),
        "--n-layers", str(n_layers),
        "--total-steps", str(args.total_steps),
        "--out-dir", str(out_dir),
    ]
    # classical-small width is left to run_experiment.py so it tracks n_layers,
    # unless the caller pinned it explicitly.
    if args.classical_small_hidden:
        cmd += ["--classical-small-hidden", *(str(h) for h in args.classical_small_hidden)]
    cmd.append("--reward-shaping" if shaping else "--no-reward-shaping")
    return cmd


def run_cell(args, n_layers: int, shaping: bool) -> tuple[str, bool, float]:
    """Run one (depth, shaping) cell. Returns (label, ok, seconds)."""
    out_dir = cell_dir(Path(args.out_root), n_layers, shaping)
    label = out_dir.name
    cmd = build_command(args, n_layers, shaping, out_dir)

    if args.resume and (out_dir / "benchmark.json").exists():
        print(f"[skip]  {label}: benchmark.json already present")
        return label, True, 0.0

    if args.dry_run:
        print(f"[plan]  {label}\n        {' '.join(cmd)}")
        return label, True, 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sweep.log"
    env = {**os.environ, **RUN_ENV}

    print(f"[run ]  {label}  -> {log_path}")
    tic = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log,
                              stderr=subprocess.STDOUT, cwd=HERE)
    elapsed = time.time() - tic

    ok = proc.returncode == 0
    print(f"[{'done' if ok else 'FAIL'}]  {label}  {elapsed / 60:.1f} min"
          + ("" if ok else f"  (exit {proc.returncode}, see {log_path})"))
    return label, ok, elapsed


def main() -> int:
    p = argparse.ArgumentParser(
        description="Regenerate the results_drl/ sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-layers", nargs="+", type=int, default=list(DEPTHS),
                   metavar="L", help=f"VQC depths to sweep (default {list(DEPTHS)})")
    p.add_argument("--shaping", choices=("on", "off", "both"), default="both",
                   help="Reward-shaping halves to run (default both)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                   help=f"Seeds per cell (default {list(SEEDS)})")
    p.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--arms", nargs="+",
                   default=["classical", "classical-small", "quantum", "quantum-noisy"],
                   help="Arms to run in every cell")
    p.add_argument("--classical-small-hidden", nargs="+", type=int, default=None,
                   metavar="H",
                   help="Pin the classical-small width instead of deriving it "
                        "per-depth. Only for reproducing an old sweep -- it "
                        "breaks the parameter-matching the arm exists for.")
    p.add_argument("--out-root", default=str(DEFAULT_ROOT),
                   help=f"Parent directory for the cells (default {DEFAULT_ROOT})")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells that already have a benchmark.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit")
    p.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                   help="Run N cells concurrently (default 1). Cells are "
                        "independent and each is pinned to a single thread, so "
                        "N costs about N cores. Do not raise this above the "
                        "free core count: oversubscribing reintroduces exactly "
                        "the spin-wait contention OMP_NUM_THREADS=1 avoids.")
    args = p.parse_args()

    shapings = {"on": [True], "off": [False], "both": [True, False]}[args.shaping]
    cells = [(L, s) for L in args.n_layers for s in shapings]

    print(f"Sweep: {len(cells)} cells x {len(args.arms)} arms x "
          f"{len(args.seeds)} seeds x {args.total_steps:,} steps")
    print(f"Output root: {args.out_root}")
    print("Env: " + " ".join(f"{k}={v!r}" for k, v in RUN_ENV.items()))
    print(f"Concurrency: {args.jobs} cell(s) at a time")
    print()

    tic = time.time()
    if args.jobs > 1 and not args.dry_run:
        # Threads, not processes: each worker only waits on a subprocess, so
        # the GIL is released for the entire runtime of the cell.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_cell, args, L, s): (L, s) for L, s in cells}
            results = [f.result() for f in as_completed(futures)]
    else:
        results = [run_cell(args, L, s) for L, s in cells]

    if args.dry_run:
        return 0

    total = time.time() - tic
    failed = [label for label, ok, _ in results if not ok]
    print(f"\n{'=' * 60}")
    print(f"{len(results) - len(failed)}/{len(results)} cells completed in "
          f"{total / 3600:.1f} h")
    for label, ok, secs in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<24}{secs / 60:>8.1f} min")
    if failed:
        print(f"\nFailed cells: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
