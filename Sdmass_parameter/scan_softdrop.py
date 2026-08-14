#!/usr/bin/env python3
"""Apply the (beta, z_cut) grid from config.py to the raw per-(strategy,mass)
jet caches built by data_pull.py, recomputing soft drop mass at each grid
point via softdrop_recluster.batch_softdrop_mass. Results are cached to
config.scan_cache_path(strategy, mass) so replotting or extending the grid
later never re-runs fastjet clustering for points already computed.
"""
import argparse

import numpy as np

import config
import data_pull
import softdrop_recluster as sdr


def grid_points():
    """All (beta, z_cut) points to compute: the beta scan at fixed
    z_cut=ZCUT_DEFAULT, plus the z_cut scan at fixed beta=BETA_DEFAULT,
    de-duplicated (they share the (BETA_DEFAULT, ZCUT_DEFAULT) point)."""
    pts = {(round(b, 6), config.ZCUT_DEFAULT) for b in config.BETA_GRID}
    pts |= {(config.BETA_DEFAULT, round(z, 6)) for z in config.ZCUT_GRID}
    return sorted(pts)


def grid_key(beta, zcut):
    return f"beta_{beta:g}__zcut_{zcut:g}".replace("-", "m").replace(".", "p")


def scan_mass(strategy, mass, force=False, verbose=True):
    raw = data_pull.load_cached(strategy, mass)
    if raw is None:
        raise RuntimeError(f"No raw cache for {strategy} @ {mass:g} GeV -- run data_pull.py first.")
    X = raw["X"]
    kin = raw["kinematics"]
    n = X.shape[0]

    path = config.scan_cache_path(strategy, mass)
    out = {}
    if path.exists() and not force:
        with np.load(path) as f:
            out = {k: f[k] for k in f.files}

    out["true_mass"] = raw["masses"][:, 0]
    changed = False
    for beta, zcut in grid_points():
        key = grid_key(beta, zcut)
        if key in out and not force:
            continue
        changed = True
        if verbose:
            print(f"[SCAN] {strategy} @ {mass:g} GeV: beta={beta:g} z_cut={zcut:g} ({n} jets)...")
        out[key] = sdr.batch_softdrop_mass(X, kin, z_cut=zcut, beta=beta)

    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        if verbose:
            print(f"[SCAN] wrote {path} ({len(grid_points())} grid points, {n} jets)")
    elif verbose:
        print(f"[SCAN] {strategy} @ {mass:g} GeV: full grid already cached, nothing to do.")
    return out


def load_scan(strategy, mass):
    path = config.scan_cache_path(strategy, mass)
    if not path.exists():
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def scan_all(strategies=config.STRATEGY_ORDER, masses=config.TARGET_MASSES, force=False):
    for s in strategies:
        for m in masses:
            scan_mass(s, m, force=force)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", type=str, default=",".join(config.STRATEGY_ORDER))
    ap.add_argument("--masses", type=str, default=",".join(f"{m:g}" for m in config.TARGET_MASSES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    masses = [float(x) for x in args.masses.split(",") if x.strip()]
    scan_all(strategies=strategies, masses=masses, force=args.force)


if __name__ == "__main__":
    main()
