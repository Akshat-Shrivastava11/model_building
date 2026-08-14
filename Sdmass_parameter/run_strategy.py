#!/usr/bin/env python3
"""Orchestrator: for one strategy, pull EOS data (skipping masses already
cached), run the (beta, z_cut) grid scan (skipping grid points already
cached), and generate all plots. Each step is independently re-runnable
and idempotent -- see data_pull.py / scan_softdrop.py for cache locations.
"""
import argparse

import config
import data_pull
import scan_softdrop as scan
import plotting


def run_strategy(strategy, masses=config.TARGET_MASSES, force_pull=False, force_scan=False, workers=None):
    print(f"\n{'=' * 70}\n{strategy}\n{'=' * 70}")
    print("[RUN] step 1/3: data pull")
    data_pull.pull_all(target_masses=masses, strategies=[strategy], force=force_pull, workers=workers)
    print("[RUN] step 2/3: (beta, z_cut) grid scan")
    scan.scan_all(strategies=[strategy], masses=masses, force=force_scan)
    print("[RUN] step 3/3: plots")
    plotting.make_all_plots(strategy)
    print(f"[RUN] {strategy} done. Plots in {config.plots_dir_for(strategy)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", type=str, required=True, choices=config.STRATEGY_ORDER)
    ap.add_argument("--masses", type=str, default=",".join(f"{m:g}" for m in config.TARGET_MASSES))
    ap.add_argument("--force-pull", action="store_true")
    ap.add_argument("--force-scan", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    masses = [float(x) for x in args.masses.split(",") if x.strip()]
    run_strategy(args.strategy, masses=masses, force_pull=args.force_pull, force_scan=args.force_scan,
                 workers=args.workers)


if __name__ == "__main__":
    main()
