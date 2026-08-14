#!/usr/bin/env python3
"""Plots for the soft drop (beta, z_cut) parameter study:
  - 2D heatmaps: true mDark vs sdmass, one per beta (z_cut fixed) and one
    per z_cut (beta fixed)
  - Kendall tau(sdmass, true mDark) per grid point, + vs-beta / vs-z_cut
    curves
  - 1D sdmass histograms per target mass, overlaid across the beta (or
    z_cut) grid
  - Event displays extending StrategiesForTrainingData.draw_event_display
    with soft-drop-dropped constituents outlined, one panel per
    representative (beta, z_cut) point

All read from the caches built by data_pull.py / scan_softdrop.py -- no
fastjet reclustering happens in this file.
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

import config
import data_pull
import scan_softdrop as scan
import softdrop_recluster as sdr
import StrategiesForTrainingData as std


def _select_at_mass(true_mass, target, tol_frac=config.MASS_SELECT_TOL_FRAC, tol_min=config.MASS_SELECT_TOL_MIN):
    tol = max(target * tol_frac, tol_min)
    return np.abs(true_mass - target) <= tol


def _fname(tag, val, ext="png"):
    s = f"{tag}_{val:g}".replace("-", "m").replace(".", "p")
    return f"{s}.{ext}"


def pooled_scan_data(strategy, masses=config.TARGET_MASSES):
    """Load and concatenate the scanned sdmass grid across all target masses
    for one strategy. Returns dict: grid_key -> concatenated sdmass array,
    plus 'true_mass' (each jet's own exact true mDark) and 'target_mass'
    (the nominal target it was pooled under). None if nothing is cached."""
    pooled = {}
    target_mass_col = []
    any_loaded = False
    for m in masses:
        s = scan.load_scan(strategy, m)
        if s is None:
            print(f"[PLOT][WARN] no scan cache for {strategy} @ {m:g} GeV, skipping (run scan_softdrop.py first).")
            continue
        any_loaded = True
        n = len(s["true_mass"])
        target_mass_col.append(np.full(n, m))
        for k, v in s.items():
            pooled.setdefault(k, []).append(v)
    if not any_loaded:
        return None
    out = {k: np.concatenate(v) for k, v in pooled.items()}
    out["target_mass"] = np.concatenate(target_mass_col)
    return out


# =============================================================================
# 2D heatmaps
# =============================================================================
def make_heatmaps(strategy, outdir):
    pooled = pooled_scan_data(strategy)
    if pooled is None:
        return
    true_mass = pooled["true_mass"]

    for beta in config.BETA_GRID:
        key = scan.grid_key(beta, config.ZCUT_DEFAULT)
        if key not in pooled:
            continue
        std.plot_heatmap(
            true_mass, pooled[key], "true mDark [GeV]", "sdmass [GeV]",
            f"{strategy}: sdmass vs true mDark  ($\\beta$={beta:g}, z_cut={config.ZCUT_DEFAULT:g})",
            outdir / _fname("heatmap_beta", beta),
        )
    for zcut in config.ZCUT_GRID:
        key = scan.grid_key(config.BETA_DEFAULT, zcut)
        if key not in pooled:
            continue
        std.plot_heatmap(
            true_mass, pooled[key], "true mDark [GeV]", "sdmass [GeV]",
            f"{strategy}: sdmass vs true mDark  ($\\beta$={config.BETA_DEFAULT:g}, z_cut={zcut:g})",
            outdir / _fname("heatmap_zcut", zcut),
        )
    print(f"[PLOT] {strategy}: heatmaps written to {outdir}")


# =============================================================================
# Kendall tau
# =============================================================================
def compute_kendall_grid(strategy, masses=config.TARGET_MASSES):
    pooled = pooled_scan_data(strategy, masses)
    if pooled is None:
        return None
    true_mass = pooled["true_mass"]
    rows = []

    def _tau_row(scan_name, beta, zcut, key):
        if key not in pooled:
            return
        sd = pooled[key]
        valid = np.isfinite(sd) & (sd >= 0) & np.isfinite(true_mass)
        if np.sum(valid) < 10:
            return
        tau, pval = kendalltau(sd[valid], true_mass[valid])
        rows.append({"scan": scan_name, "beta": beta, "zcut": zcut, "n": int(np.sum(valid)),
                     "kendall_tau": float(tau), "pvalue": float(pval)})

    for beta in config.BETA_GRID:
        _tau_row("beta", beta, config.ZCUT_DEFAULT, scan.grid_key(beta, config.ZCUT_DEFAULT))
    for zcut in config.ZCUT_GRID:
        _tau_row("zcut", config.BETA_DEFAULT, zcut, scan.grid_key(config.BETA_DEFAULT, zcut))
    return rows


def write_kendall_csv(rows, outpath):
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scan", "beta", "zcut", "n", "kendall_tau", "pvalue"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[PLOT] wrote {outpath}")


def plot_kendall_vs_beta(rows, outpath):
    sub = sorted((r for r in rows if r["scan"] == "beta"), key=lambda r: r["beta"])
    if not sub:
        return
    betas = [r["beta"] for r in sub]
    taus = [r["kendall_tau"] for r in sub]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, taus, "o-", color="#1b9e77")
    ax.axvline(0, color="0.5", ls="--", lw=1, label="mMDT ($\\beta$=0)")
    ax.set_xlabel(r"soft drop $\beta$")
    ax.set_ylabel(r"Kendall $\tau$(sdmass, true $m_{Dark}$)")
    ax.set_title(f"z_cut fixed at {config.ZCUT_DEFAULT:g}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[PLOT] wrote {outpath}")


def plot_kendall_vs_zcut(rows, outpath):
    sub = sorted((r for r in rows if r["scan"] == "zcut"), key=lambda r: r["zcut"])
    if not sub:
        return
    zcuts = [r["zcut"] for r in sub]
    taus = [r["kendall_tau"] for r in sub]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(zcuts, taus, "o-", color="#d95f02")
    ax.axvline(0.1, color="0.5", ls="--", lw=1, label="default z_cut=0.1")
    ax.set_xlabel(r"soft drop $z_{\rm cut}$")
    ax.set_ylabel(r"Kendall $\tau$(sdmass, true $m_{Dark}$)")
    ax.set_title(f"$\\beta$ fixed at {config.BETA_DEFAULT:g}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[PLOT] wrote {outpath}")


# =============================================================================
# 1D sdmass histograms per target mass, overlaid across the grid
# =============================================================================
def _overlay_1d(strategy, outdir, grid_values, key_fn, label_fn, fixed_desc, fname_prefix, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    for m in config.TARGET_MASSES:
        s = scan.load_scan(strategy, m)
        if s is None:
            continue
        true_mass = s["true_mass"]
        sel = _select_at_mass(true_mass, m)
        if np.sum(sel) < 5:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        n_lines = 0
        for i, val in enumerate(grid_values):
            key = key_fn(val)
            if key not in s:
                continue
            sd = s[key][sel]
            sd = sd[np.isfinite(sd) & (sd >= 0)]
            if sd.size == 0:
                continue
            color = cmap(i / max(len(grid_values) - 1, 1))
            ax.hist(sd, bins=40, range=(0, max(2.5 * m, 20)), histtype="step", density=True,
                    linewidth=1.6, color=color, label=f"{label_fn(val)}  (N={sd.size})")
            n_lines += 1
        if n_lines == 0:
            plt.close(fig)
            continue
        ax.axvline(m, color="red", ls="--", lw=1.2, label=f"true mDark = {m:g} GeV")
        ax.set_yscale("log")
        ax.set_xlabel("sdmass [GeV]")
        ax.set_ylabel("normalized entries (log scale)")
        ax.set_title(f"{strategy}: sdmass near true mDark={m:g} GeV ({fixed_desc})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"{fname_prefix}_{config.mass_label(m)}.png", dpi=150)
        plt.close(fig)
    print(f"[PLOT] {strategy}: 1D overlays ({fname_prefix}) written to {outdir}")


def make_1d_overlays(strategy, outdir):
    _overlay_1d(
        strategy, outdir, config.BETA_GRID,
        key_fn=lambda b: scan.grid_key(b, config.ZCUT_DEFAULT),
        label_fn=lambda b: f"$\\beta$={b:g}",
        fixed_desc=f"z_cut={config.ZCUT_DEFAULT:g}", fname_prefix="hist1d_beta_scan_mass",
        cmap_name="viridis",
    )
    _overlay_1d(
        strategy, outdir, config.ZCUT_GRID,
        key_fn=lambda z: scan.grid_key(config.BETA_DEFAULT, z),
        label_fn=lambda z: f"z_cut={z:g}",
        fixed_desc=f"$\\beta$={config.BETA_DEFAULT:g}", fname_prefix="hist1d_zcut_scan_mass",
        cmap_name="plasma",
    )


# =============================================================================
# Event displays with soft-drop-dropped constituents outlined
# =============================================================================
def make_event_displays_softdrop(strategy, outdir, masses=config.TARGET_MASSES,
                                  n_per_mass=config.N_EVENT_DISPLAY_JETS_PER_MASS, seed=137):
    rng = np.random.default_rng(seed)
    scan_points = ([(b, config.ZCUT_DEFAULT) for b in config.EVENT_DISPLAY_BETAS]
                   + [(config.BETA_DEFAULT, z) for z in config.EVENT_DISPLAY_ZCUTS])
    ncols = 4
    nrows = int(np.ceil(len(scan_points) / ncols))

    for m in masses:
        raw = data_pull.load_cached(strategy, m)
        if raw is None:
            continue
        true_mass = raw["masses"][:, 0]
        sel = np.nonzero(_select_at_mass(true_mass, m))[0]
        if sel.size == 0:
            continue
        chosen = rng.choice(sel, size=min(n_per_mass, sel.size), replace=False)

        for idx in chosen:
            idx = int(idx)
            j_eta = float(raw["kinematics"][idx, 1])
            j_phi = float(raw["kinematics"][idx, 2])
            X_idx = raw["X"][idx]

            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
            for ip, (beta, zcut) in enumerate(scan_points):
                ax = axes[ip // ncols][ip % ncols]
                std.draw_event_display(ax, raw, idx, std.MAX_DARK_HADRONS_DEFAULT, fatjet_r=std.FATJET_R_DEFAULT)
                _, sd_mass, _, dropped = sdr.jet_softdrop_full(X_idx, j_eta, j_phi, zcut, beta)
                if dropped:
                    d = np.asarray(dropped, dtype=int)
                    ax.scatter(X_idx[d, 2], X_idx[d, 3], s=230, facecolors="none", edgecolors="crimson",
                               linewidths=1.8, marker="o", zorder=6)
                ax.text(0.02, 0.02, f"red ring = dropped by soft drop ({len(dropped)})",
                        transform=ax.transAxes, fontsize=6, color="crimson", va="bottom")
                ax.set_title(f"$\\beta$={beta:g}, z_cut={zcut:g}   sdmass={sd_mass:.1f} GeV", fontsize=8)

            for ip in range(len(scan_points), nrows * ncols):
                axes[ip // ncols][ip % ncols].axis("off")

            fig.suptitle(f"{strategy}: true mDark={m:g} GeV, jet #{idx} -- "
                         f"soft-drop-dropped constituents per scan point", y=1.0)
            fig.tight_layout()
            fig.savefig(outdir / f"event_display_softdrop_{config.mass_label(m)}_idx{idx}.png", dpi=140)
            plt.close(fig)
    print(f"[PLOT] {strategy}: soft-drop event displays written to {outdir}")


# =============================================================================
# Driver
# =============================================================================
def make_all_plots(strategy):
    outdir = config.plots_dir_for(strategy)
    make_heatmaps(strategy, outdir)
    rows = compute_kendall_grid(strategy)
    if rows:
        write_kendall_csv(rows, outdir / "kendall_tau_grid.csv")
        plot_kendall_vs_beta(rows, outdir / "kendall_tau_vs_beta.png")
        plot_kendall_vs_zcut(rows, outdir / "kendall_tau_vs_zcut.png")
    make_1d_overlays(strategy, outdir)
    make_event_displays_softdrop(strategy, outdir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", type=str, default=",".join(config.STRATEGY_ORDER))
    args = ap.parse_args()
    for s in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        print(f"\n==== plotting {s} ====")
        make_all_plots(s)


if __name__ == "__main__":
    main()
