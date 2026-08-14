#!/usr/bin/env python3
"""Adaptive-window EOS data puller for the soft drop parameter study.

Reuses StrategiesForTrainingData.py's own per-file processing function
(process_one_point) unmodified -- it already builds the PF-constituent `X`
array, the strategy pass/fail flags, and the Delphes-computed jet_sdmass
ground truth in one pass over a ROOT file. We just point it at files chosen
by proximity to each of our 8 target masses (instead of the global
mass-stratified scan StrategiesForTrainingData.py itself does), uncapped
(no 20-jets-per-point limit), and stop early per (mass, strategy) once
enough jets have been collected.

Output: one cached npz per (strategy, target mass) under
config.RAW_CACHE_DIR, with the exact same schema as today's strategy
datasets (X, kinematics, macro, masses, ...) -- so every existing helper in
StrategiesForTrainingData.py keeps working on it unmodified. Already-cached
(strategy, mass) pairs are skipped unless --force is passed.
"""
import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import config
import StrategiesForTrainingData as std


def candidate_points_for_mass(target_mass, points, window_frac, floor_gev):
    window = max(target_mass * window_frac, floor_gev)
    cands = [p for p in points if abs(p["mpi"] - target_mass) <= window]
    cands.sort(key=lambda p: abs(p["mpi"] - target_mass))
    return cands, window


def pull_target_mass(
    target_mass,
    strategies=config.STRATEGY_ORDER,
    workers=None,
    min_jets_target=config.MIN_JETS_TARGET,
    widest_window_frac=config.WINDOW_FRACTIONS[-1],
    floor_gev=config.WINDOW_ABS_FLOOR_GEV,
    max_files=config.MAX_FILES_PER_MASS,
    verbose=True,
):
    """Pull and classify jets from EOS files near `target_mass`, proximity
    order, stopping once every strategy in `strategies` has >= min_jets_target
    jets (or the file budget runs out). Returns (arrays_by_strategy, summary).
    """
    all_points = std.read_mass_points(std.MASS_POINTS_FILE)
    candidates, window_used = candidate_points_for_mass(target_mass, all_points, widest_window_frac, floor_gev)
    if not candidates:
        raise RuntimeError(f"No EOS mass points found within +/-{widest_window_frac*100:.0f}% of {target_mass} GeV")
    candidates = candidates[:max_files]

    buckets = {s: std.empty_bucket() for s in strategies}
    counts = {s: 0 for s in strategies}
    n_files_used = 0
    n_files_failed = 0
    n_events_total = 0

    workers = workers or os.cpu_count()
    batch_size = max(workers * 4, 16)
    # 'spawn' avoids the fork-after-XRootD-threads deadlock -- see
    # StrategiesForTrainingData.run_sampling for the original rationale.
    mp_ctx = multiprocessing.get_context("spawn")

    if verbose:
        print(
            f"[PULL] target={target_mass:g} GeV: {len(candidates)} candidate files within "
            f"+/-{window_used:.3f} GeV, min_jets_target={min_jets_target}, workers={workers}"
        )

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
        idx_ptr = 0
        pending = {}

        def submit_more(n):
            nonlocal idx_ptr
            submitted = 0
            while submitted < n and idx_ptr < len(candidates):
                point = candidates[idx_ptr]
                idx_ptr += 1
                url = std.point_events_root_url(point, std.EOS_BASE_DEFAULT, std.EOS_HOST_DEFAULT)
                worker_args = (
                    url, point["point_id"], point["mpi"],
                    std.FATJET_R_DEFAULT, std.MATCHING_R_DEFAULT,
                    std.MAX_CONSTITUENTS_DEFAULT, std.MAX_DARK_HADRONS_DEFAULT,
                    std.MAX_INVISIBLE_DARK_HADRONS_DEFAULT,
                    True,            # require_status_8384
                    1_000_000_000,   # max_jets_per_point -- uncapped
                    std.MAX_OUTSIDE_CONSTITUENTS_DEFAULT,
                )
                fut = executor.submit(std.process_one_point, worker_args)
                pending[fut] = point
                submitted += 1

        submit_more(batch_size)
        while pending:
            done = next(as_completed(pending))
            point = pending.pop(done)
            res = done.result()

            if res["ok"] and res["n_events"] > 0:
                n_files_used += 1
                n_events_total += res["n_events"]
                for s in strategies:
                    std.merge_bucket(buckets[s], res["bucket"][s])
                    counts[s] = len(buckets[s]["masses"])
            elif not res["ok"]:
                n_files_failed += 1
                if verbose:
                    print(f"[PULL][WARN] point {point['point_id']} (mpi={point['mpi']:g}) failed: "
                          f"{(res['error'] or '').splitlines()[0] if res['error'] else 'unknown error'}")

            if verbose and (n_files_used % 20 == 0 or all(counts[s] >= min_jets_target for s in strategies)):
                print(f"[PULL]   files={n_files_used}/{len(candidates)} events={n_events_total:,} | "
                      + " ".join(f"{s}={counts[s]:,}" for s in strategies))

            if all(counts[s] >= min_jets_target for s in strategies):
                break
            if idx_ptr < len(candidates):
                submit_more(1)

    arrays = {}
    for s in strategies:
        a = std.bucket_to_arrays(
            buckets[s], std.MAX_CONSTITUENTS_DEFAULT, std.MAX_DARK_HADRONS_DEFAULT,
            std.MAX_INVISIBLE_DARK_HADRONS_DEFAULT, std.MAX_OUTSIDE_CONSTITUENTS_DEFAULT,
        )
        arrays[s] = a

    summary = {
        "target_mass": target_mass,
        "window_gev": window_used,
        "n_files_used": n_files_used,
        "n_files_failed": n_files_failed,
        "n_files_available": len(candidates),
        "n_events_total": n_events_total,
        "counts": dict(counts),
        "min_jets_target": min_jets_target,
        "shortfall": {s: max(0, min_jets_target - counts[s]) for s in strategies},
    }
    if verbose:
        short = {s: v for s, v in summary["shortfall"].items() if v > 0}
        if short:
            print(f"[PULL][WARN] target={target_mass:g} GeV did not reach min_jets_target for: {short} "
                  f"(exhausted {n_files_used}/{len(candidates)} available files)")
        else:
            print(f"[PULL] target={target_mass:g} GeV done: " + " ".join(f"{s}={counts[s]:,}" for s in strategies))

    return arrays, summary


def save_arrays(strategy, target_mass, arrays, force=False):
    path = config.raw_cache_path(strategy, target_mass)
    if path.exists() and not force:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays is None or len(arrays.get("masses", [])) == 0:
        print(f"[PULL][WARN] {strategy} @ {target_mass:g} GeV: 0 jets collected, not writing a cache file.")
        return path, False
    np.savez_compressed(path, **arrays)
    return path, True


def load_cached(strategy, target_mass):
    path = config.raw_cache_path(strategy, target_mass)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def pull_all(target_masses=config.TARGET_MASSES, strategies=config.STRATEGY_ORDER, force=False, workers=None):
    summaries = []
    for m in target_masses:
        missing = [s for s in strategies if force or not config.raw_cache_path(s, m).exists()]
        if not missing:
            print(f"[PULL] target={m:g} GeV: all strategies already cached, skipping (--force to redo).")
            continue
        arrays, summary = pull_target_mass(m, strategies=strategies, workers=workers)
        for s in strategies:
            save_arrays(s, m, arrays.get(s), force=force)
        summaries.append(summary)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", type=str, default=",".join(config.STRATEGY_ORDER))
    ap.add_argument("--masses", type=str, default=",".join(f"{m:g}" for m in config.TARGET_MASSES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    masses = [float(x) for x in args.masses.split(",") if x.strip()]
    pull_all(target_masses=masses, strategies=strategies, force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
