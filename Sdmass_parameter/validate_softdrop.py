#!/usr/bin/env python3
"""Validation gate: pull a modest real sample, recompute soft drop mass at
exactly (beta=0, z_cut=0.1, R0=0.8) from the stored PF constituents, and
compare against Delphes' own FatJet.SoftDroppedJet.Mass (macro['jet_sdmass'])
for the same jets. Also cross-checks the plain (ungroomed) constituent-sum
mass against macro['jet_mass'], to confirm the stored X constituent set
faithfully represents the jet fastjet itself would reconstruct.

Must show good agreement before the (beta, z_cut) grid scan is trusted.
"""
import numpy as np

import config
import data_pull
import softdrop_recluster as sdr
from StrategiesForTrainingData import MACRO_NAMES


def run_validation(target_mass=50.0, strategy="strat1_leading_full", min_jets_target=200):
    print(f"[VALIDATE] pulling a small sample at {target_mass:g} GeV for {strategy} "
          f"(min_jets_target={min_jets_target}) ...")
    arrays, summary = data_pull.pull_target_mass(
        target_mass, strategies=[strategy], min_jets_target=min_jets_target,
    )
    a = arrays[strategy]
    X = a["X"]
    kin = a["kinematics"]  # [pt, eta, phi, mass] per jet
    macro = a["macro"]
    n = X.shape[0]
    print(f"[VALIDATE] got {n} jets. Recomputing plain mass + soft drop (beta=0,z_cut=0.1)...")

    jm_idx = MACRO_NAMES.index("jet_mass")
    sd_idx = MACRO_NAMES.index("jet_sdmass")

    plain_recomputed = np.array([
        sdr.plain_jet_mass_from_constituents(X[i], float(kin[i, 1]), float(kin[i, 2])) for i in range(n)
    ])
    sd_recomputed = np.array([
        sdr.jet_softdrop_mass(X[i], float(kin[i, 1]), float(kin[i, 2]), z_cut=0.1, beta=0.0, R0=config.R0)
        for i in range(n)
    ])

    plain_stored = macro[:, jm_idx]
    sd_stored = macro[:, sd_idx]

    def report(name, stored, recomputed):
        valid = np.isfinite(stored) & np.isfinite(recomputed) & (stored > 0.5)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            print(f"  {name}: no valid jets to compare")
            return
        rel_diff = np.abs(recomputed[valid] - stored[valid]) / stored[valid]
        print(f"  {name}: n={n_valid}  median rel.diff={np.median(rel_diff):.4f}  "
              f"90th pct={np.percentile(rel_diff, 90):.4f}  "
              f"frac within 5%={(np.mean(rel_diff < 0.05)):.3f}  "
              f"frac within 10%={(np.mean(rel_diff < 0.10)):.3f}")
        return rel_diff

    print("[VALIDATE] Plain (ungroomed) constituent-sum mass vs stored macro jet_mass:")
    report("plain_mass", plain_stored, plain_recomputed)

    print("[VALIDATE] Soft drop (beta=0, z_cut=0.1) recomputed vs Delphes macro jet_sdmass:")
    rel_diff_sd = report("softdrop_mass(0,0.1)", sd_stored, sd_recomputed)

    return {
        "n": n, "plain_stored": plain_stored, "plain_recomputed": plain_recomputed,
        "sd_stored": sd_stored, "sd_recomputed": sd_recomputed, "rel_diff_sd": rel_diff_sd,
    }


if __name__ == "__main__":
    run_validation()
