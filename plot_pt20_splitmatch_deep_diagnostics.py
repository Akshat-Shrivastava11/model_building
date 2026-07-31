#!/usr/bin/env python3
"""
plot_visible_dh_pt_ratio_softdrop_by_ndark.py

Plot-only study for visible dark-hadron multiplicity and SoftDrop mass performance.

What it does
------------
Loops directly over Delphes ROOT files, without writing a new NPZ, and for each
approved reco FatJet:

  1. Finds visible dark hadrons inside the reco FatJet cone.
     - dark hadron PID: pi_D/rho_D = {4900111, +/-4900211, 4900113, +/-4900213}
     - default status: 83 or 84
     - "visible" means the dark hadron has at least one visible stable descendant.

  2. Sorts those visible DHs by generator-level pT.
     DH1 = highest-pT visible DH, DH2 = second-highest-pT visible DH.
     ratio = pT(DH2) / pT(DH1).

  3. Makes plots to test whether cutting on this ratio changes SoftDrop mass behavior:
     - ratio distribution for N_DH^vis >= 2
     - SoftDrop residual overlays for ratio cuts
     - summary curves vs ratio cut: N, Kendall tau, median rel residual, 68%-width
     - multipage PDF: SoftDrop mass vs true dark-hadron mass for each ratio cut

  4. Makes a multipage PDF: SoftDrop mass vs true dark-hadron mass for
     N_DH^vis = 1, 2, ..., 10, with Kendall tau printed on every page.

This intentionally mirrors the ROOT-level matching logic used in the DRACON dataset
builders, but only writes plots and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

try:
    import awkward as ak
    import uproot
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script needs uproot and awkward in the environment. "
        "Activate your DRACON/darknet environment first."
    ) from exc

try:
    from scipy.stats import kendalltau
except Exception:  # pragma: no cover
    kendalltau = None


# =============================================================================
# Physics constants / dataset conventions
# =============================================================================
DARK_HADRONS = {
    4900111,    # piD0
    4900211,    # piD+
    -4900211,   # piD-
    4900113,    # rhoD0
    4900213,    # rhoD+
    -4900213,   # rhoD-
}

DARK_HADRON_COUNT_STATUSES = {83, 84}

INVISIBLE_PIDS_ABS = {
    12, 14, 16, 18,
    4900101, 4900102, 4900103, 4900104, 4900105, 4900106,
}

CATEGORY_RANK = {"none": 0, "partial": 1, "full": 2}


# =============================================================================
# General helpers
# =============================================================================
def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "savefig.bbox": "tight",
    })


def calc_dphi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2.0 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + calc_dphi(phi1, phi2) ** 2)


def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, pad=8)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)


def safe_name(x: float) -> str:
    return f"{x:.3f}".replace(".", "p").replace("-", "m")


def finite_mask(*arrays) -> np.ndarray:
    n = min(len(np.asarray(a).reshape(-1)) for a in arrays)
    mask = np.ones(n, dtype=bool)
    for a in arrays:
        aa = np.asarray(a).reshape(-1)[:n]
        mask &= np.isfinite(aa)
    return mask


def compute_tau(x, y) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(x), len(y))
    if n < 3 or kendalltau is None:
        return float("nan")
    x = x[:n]
    y = y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return float("nan")
    val = kendalltau(x[m], y[m], nan_policy="omit").correlation
    return float(val) if val is not None and np.isfinite(val) else float("nan")


def robust_rel_width(rel: np.ndarray) -> float:
    rel = np.asarray(rel).reshape(-1)
    rel = rel[np.isfinite(rel)]
    if len(rel) < 5:
        return float("nan")
    q16, q84 = np.nanpercentile(rel, [16, 84])
    return float(0.5 * (q84 - q16))


def parse_mpi_from_path(path: str) -> Optional[float]:
    """Parse target m_pi/m_dark from common SVJ path formats."""
    patterns = [
        r"point_mPI_([0-9]+(?:\.[0-9]+)?)",
        r"mpi-([0-9]+(?:\.[0-9]+)?)",
        r"mpiEqMrho_([0-9]+(?:p[0-9]+)?)",
        r"mDark_([0-9]+(?:\.[0-9]+)?)",
        r"mdark[_-]([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(path), flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace("p", "."))
    return None


def read_filelist(path: str, eos_host: str = "") -> List[str]:
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(normalize_root_path(s, eos_host=eos_host))
    return out


def normalize_root_path(path: str, eos_host: str = "") -> str:
    path = path.strip()
    if path.startswith("root://"):
        return path
    if eos_host and path.startswith("/store/"):
        return eos_host.rstrip("/") + "/" + path.lstrip("/")
    return path


def collect_input_files(args) -> List[str]:
    files: List[str] = []
    if args.filelist:
        files.extend(read_filelist(args.filelist, eos_host=args.eos_host))
    if args.input_glob:
        for g in args.input_glob:
            files.extend(sorted(glob.glob(g)))
    # preserve order, remove duplicates
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    if args.max_files is not None and args.max_files > 0:
        uniq = uniq[: args.max_files]
    return uniq


# =============================================================================
# ROOT helpers copied/adapted from the DRACON builder workflow
# =============================================================================
def get_delphes_tree(root_file: str):
    f = uproot.open(root_file)
    if "Delphes;1" in f:
        return f, f["Delphes;1"]
    if "Delphes" in f:
        return f, f["Delphes"]
    f.close()
    raise RuntimeError("Could not find Delphes tree")


def get_branch(tree, possible_names: Sequence[str], required: bool = True):
    keys = set(tree.keys())
    for name in possible_names:
        if name in keys:
            return tree[name].array(library="ak")
    if required:
        raise KeyError(f"Could not find any branch from: {possible_names}")
    return None


def safe_softdrop_mass(rf_sdmass, event_idx: int, jet_idx: int) -> float:
    if rf_sdmass is None:
        return -1.0
    try:
        if len(rf_sdmass[event_idx]) <= jet_idx:
            return -1.0
        return float(rf_sdmass[event_idx][jet_idx])
    except Exception:
        return -1.0


def safe_softdrop_pt(rf_sdpt, event_idx: int, jet_idx: int) -> float:
    if rf_sdpt is None:
        return -1.0
    try:
        if len(rf_sdpt[event_idx]) <= jet_idx:
            return -1.0
        return float(rf_sdpt[event_idx][jet_idx])
    except Exception:
        return -1.0


def build_softdrop_from_packed(tree):
    """Fallback for FatJet/FatJet.SoftDroppedP4[5]."""
    rf_sdp4_raw = get_branch(
        tree,
        ["FatJet/FatJet.SoftDroppedP4[5]", "FatJet.SoftDroppedP4[5]"],
        required=False,
    )
    if rf_sdp4_raw is None:
        return None, None

    tmp_pt = []
    tmp_mass = []
    for i_evt in range(len(rf_sdp4_raw)):
        evt_pt = []
        evt_mass = []
        for j in range(len(rf_sdp4_raw[i_evt])):
            jet_entry = rf_sdp4_raw[i_evt][j]
            if len(jet_entry) == 0:
                evt_pt.append(-1.0)
                evt_mass.append(-1.0)
                continue
            vec = jet_entry[0]
            px = vec["fP"]["fX"]
            py = vec["fP"]["fY"]
            pz = vec["fP"]["fZ"]
            e = vec["fE"]
            pt = np.sqrt(px**2 + py**2)
            m2 = e**2 - px**2 - py**2 - pz**2
            mass = np.sqrt(max(m2, 0.0))
            evt_pt.append(float(pt))
            evt_mass.append(float(mass))
        tmp_pt.append(evt_pt)
        tmp_mass.append(evt_mass)
    return ak.Array(tmp_mass), ak.Array(tmp_pt)


def should_keep_category(cat: str, keep_mode: str) -> bool:
    if keep_mode == "full_only":
        return cat == "full"
    if keep_mode == "partial_only":
        return cat == "partial"
    if keep_mode == "none_only":
        return cat == "none"
    if keep_mode == "full_plus_partial":
        return cat in ["full", "partial"]
    if keep_mode == "partial_plus_none":
        return cat in ["partial", "none"]
    if keep_mode == "not_full":
        return cat in ["partial", "none"]
    if keep_mode == "all":
        return True
    raise ValueError(f"Unknown keep_mode: {keep_mode}")


def is_visible_final_state(pid: int, status: int) -> bool:
    apid = abs(int(pid))
    if int(status) != 1:
        return False
    if apid in INVISIBLE_PIDS_ABS:
        return False
    return True


def get_descendants(idx: int, d1s, d2s, n_parts: int, visited=None) -> List[int]:
    if visited is None:
        visited = set()
    out: List[int] = []
    if idx < 0 or idx >= n_parts or idx in visited:
        return out
    visited.add(idx)
    d1 = int(d1s[idx])
    d2 = int(d2s[idx])
    if d1 < 0:
        return out
    if d2 < d1:
        d2 = d1
    for child in range(d1, d2 + 1):
        if 0 <= child < n_parts:
            out.append(child)
            out.extend(get_descendants(child, d1s, d2s, n_parts, visited))
    return out


def build_genfatjet_containment_map(
    pids_evt,
    status_evt,
    eta_evt,
    phi_evt,
    mass_evt,
    d1_evt,
    d2_evt,
    gf_eta_evt,
    gf_phi_evt,
    fatjet_r: float,
) -> Dict[int, dict]:
    """
    Same idea as the dataset builder: for each dark hadron, find nearest GenFatJet,
    then classify visible descendants as full/partial/none contained.
    Keep best dark-hadron candidate per GenFatJet.
    """
    genfatjet_best: Dict[int, dict] = {}
    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]

    for dh_idx in dh_indices:
        dh_eta = eta_evt[dh_idx]
        dh_phi = phi_evt[dh_idx]
        dh_mass = mass_evt[dh_idx]

        if not np.isfinite(dh_eta) or not np.isfinite(dh_phi):
            continue
        if len(gf_eta_evt) == 0:
            continue

        drs_dh_gf = calc_dr(dh_eta, dh_phi, gf_eta_evt, gf_phi_evt)
        best_gf_idx = int(np.argmin(drs_dh_gf))
        best_dr = float(drs_dh_gf[best_gf_idx])
        if best_dr >= fatjet_r:
            continue

        desc = sorted(set(get_descendants(int(dh_idx), d1_evt, d2_evt, len(pids_evt))))
        if len(desc) == 0:
            continue

        vis_desc = []
        for child_idx in desc:
            if is_visible_final_state(pids_evt[child_idx], status_evt[child_idx]):
                if np.isfinite(eta_evt[child_idx]) and np.isfinite(phi_evt[child_idx]):
                    vis_desc.append(child_idx)
        if len(vis_desc) == 0:
            continue

        drs_desc = calc_dr(
            eta_evt[vis_desc],
            phi_evt[vis_desc],
            gf_eta_evt[best_gf_idx],
            gf_phi_evt[best_gf_idx],
        )
        inside = drs_desc < fatjet_r
        n_inside = int(np.sum(inside))
        n_total = len(vis_desc)
        frac_inside = n_inside / max(n_total, 1)

        if n_inside == n_total:
            cat = "full"
        elif n_inside > 0:
            cat = "partial"
        else:
            cat = "none"

        candidate = {
            "dh_idx": int(dh_idx),
            "dh_mass": float(dh_mass),
            "gf_idx": int(best_gf_idx),
            "category": cat,
            "rank": int(CATEGORY_RANK[cat]),
            "frac_inside": float(frac_inside),
            "n_vis_desc": int(n_total),
        }

        if best_gf_idx not in genfatjet_best:
            genfatjet_best[best_gf_idx] = candidate
        else:
            old = genfatjet_best[best_gf_idx]
            old_key = (old["rank"], old["frac_inside"], old["n_vis_desc"])
            new_key = (candidate["rank"], candidate["frac_inside"], candidate["n_vis_desc"])
            if new_key > old_key:
                genfatjet_best[best_gf_idx] = candidate

    return genfatjet_best


def visible_dark_hadron_pts_in_reco_jet(
    pids_evt,
    status_evt,
    pt_evt,
    eta_evt,
    phi_evt,
    d1_evt,
    d2_evt,
    j_eta: float,
    j_phi: float,
    fatjet_r: float,
    require_status_8384: bool = True,
    visible_desc_must_be_inside_jet: bool = False,
) -> List[float]:
    """
    Return sorted visible dark-hadron pTs inside the reco FatJet.

    A DH is counted if:
      - PID is one of the dark pion/rho mesons
      - status is 83/84 by default
      - DH axis is within DeltaR < fatjet_r of the reco FatJet axis
      - it has at least one visible stable descendant

    If visible_desc_must_be_inside_jet is enabled, at least one visible stable descendant
    must also lie within the reco FatJet cone.
    """
    finite = np.isfinite(eta_evt) & np.isfinite(phi_evt) & np.isfinite(pt_evt)
    pid_mask = np.isin(pids_evt, list(DARK_HADRONS))
    if require_status_8384:
        status_mask = np.isin(status_evt, list(DARK_HADRON_COUNT_STATUSES))
    else:
        status_mask = np.ones_like(pid_mask, dtype=bool)

    dh_indices = np.where(finite & pid_mask & status_mask)[0]
    pts = []

    for dh_idx in dh_indices:
        dr = calc_dr(eta_evt[dh_idx], phi_evt[dh_idx], j_eta, j_phi)
        if dr >= fatjet_r:
            continue

        desc = sorted(set(get_descendants(int(dh_idx), d1_evt, d2_evt, len(pids_evt))))
        has_visible_desc = False
        for child_idx in desc:
            if not is_visible_final_state(pids_evt[child_idx], status_evt[child_idx]):
                continue
            if not (np.isfinite(eta_evt[child_idx]) and np.isfinite(phi_evt[child_idx])):
                continue
            if visible_desc_must_be_inside_jet:
                child_dr = calc_dr(eta_evt[child_idx], phi_evt[child_idx], j_eta, j_phi)
                if child_dr >= fatjet_r:
                    continue
            has_visible_desc = True
            break

        if has_visible_desc and pt_evt[dh_idx] > 0:
            pts.append(float(pt_evt[dh_idx]))

    pts.sort(reverse=True)
    return pts


def empty_columns() -> Dict[str, list]:
    return {
        "true_mass": [],
        "sdmass": [],
        "sdpt": [],
        "reco_pt": [],
        "reco_mass": [],
        "n_visible_dh": [],
        "dh1_pt": [],
        "dh2_pt": [],
        "dh2_over_dh1": [],
        "containment_category": [],
        "containment_fraction": [],
        "root_file": [],
    }


def process_one_file(task) -> Dict[str, list]:
    (
        root_file,
        keep_mode,
        max_events_per_file,
        fatjet_r,
        matching_r,
        pt_cut,
        require_status_8384,
        visible_desc_must_be_inside_jet,
    ) = task

    cols = empty_columns()
    target_mpi = parse_mpi_from_path(root_file)
    if target_mpi is None:
        print(f"[WARN] Could not parse target mass from path, skipping: {root_file}", flush=True)
        return cols

    f = None
    try:
        f, tree = get_delphes_tree(root_file)

        # Gen particles
        gen_pid = get_branch(tree, ["GenParticle/GenParticle.PID", "GenParticle.PID"])
        gen_pt = get_branch(tree, ["GenParticle/GenParticle.PT", "GenParticle.PT"])
        gen_eta = get_branch(tree, ["GenParticle/GenParticle.Eta", "GenParticle.Eta"])
        gen_phi = get_branch(tree, ["GenParticle/GenParticle.Phi", "GenParticle.Phi"])
        gen_mass = get_branch(tree, ["GenParticle/GenParticle.Mass", "GenParticle.Mass"])
        gen_status = get_branch(tree, ["GenParticle/GenParticle.Status", "GenParticle.Status"])
        gen_d1 = get_branch(tree, ["GenParticle/GenParticle.D1", "GenParticle.D1"])
        gen_d2 = get_branch(tree, ["GenParticle/GenParticle.D2", "GenParticle.D2"])

        # GenFatJets
        gf_eta = get_branch(tree, ["GenFatJet/GenFatJet.Eta", "GenFatJet.Eta"])
        gf_phi = get_branch(tree, ["GenFatJet/GenFatJet.Phi", "GenFatJet.Phi"])

        # Reco FatJets
        rf_pt = get_branch(tree, ["FatJet/FatJet.PT", "FatJet.PT"])
        rf_eta = get_branch(tree, ["FatJet/FatJet.Eta", "FatJet.Eta"])
        rf_phi = get_branch(tree, ["FatJet/FatJet.Phi", "FatJet.Phi"])
        rf_m = get_branch(tree, ["FatJet/FatJet.Mass", "FatJet.Mass"])

        # SoftDrop
        rf_sdmass = get_branch(
            tree,
            ["FatJet/FatJet.SoftDroppedJet.Mass", "FatJet.SoftDroppedJet.Mass"],
            required=False,
        )
        rf_sdpt = get_branch(
            tree,
            ["FatJet/FatJet.SoftDroppedJet.PT", "FatJet.SoftDroppedJet.PT"],
            required=False,
        )
        if rf_sdmass is None or rf_sdpt is None:
            packed_sdmass, packed_sdpt = build_softdrop_from_packed(tree)
            if rf_sdmass is None:
                rf_sdmass = packed_sdmass
            if rf_sdpt is None:
                rf_sdpt = packed_sdpt

        n_events = len(rf_pt)
        if max_events_per_file is not None and max_events_per_file > 0:
            n_events = min(n_events, max_events_per_file)

        for iev in range(n_events):
            pids_evt = ak.to_numpy(gen_pid[iev])
            pt_evt = ak.to_numpy(gen_pt[iev])
            eta_evt = ak.to_numpy(gen_eta[iev])
            phi_evt = ak.to_numpy(gen_phi[iev])
            mass_evt = ak.to_numpy(gen_mass[iev])
            status_evt = ak.to_numpy(gen_status[iev])
            d1_evt = ak.to_numpy(gen_d1[iev])
            d2_evt = ak.to_numpy(gen_d2[iev])

            gf_eta_evt = ak.to_numpy(gf_eta[iev])
            gf_phi_evt = ak.to_numpy(gf_phi[iev])

            rfs_pt = ak.to_numpy(rf_pt[iev])
            rfs_eta = ak.to_numpy(rf_eta[iev])
            rfs_phi = ak.to_numpy(rf_phi[iev])
            rfs_m = ak.to_numpy(rf_m[iev])

            genfatjet_best = build_genfatjet_containment_map(
                pids_evt=pids_evt,
                status_evt=status_evt,
                eta_evt=eta_evt,
                phi_evt=phi_evt,
                mass_evt=mass_evt,
                d1_evt=d1_evt,
                d2_evt=d2_evt,
                gf_eta_evt=gf_eta_evt,
                gf_phi_evt=gf_phi_evt,
                fatjet_r=fatjet_r,
            )

            for r_idx in range(len(rfs_pt)):
                j_pt = float(rfs_pt[r_idx])
                j_eta = float(rfs_eta[r_idx])
                j_phi = float(rfs_phi[r_idx])
                j_m = float(rfs_m[r_idx])

                if not (np.isfinite(j_pt) and np.isfinite(j_eta) and np.isfinite(j_phi) and np.isfinite(j_m)):
                    continue
                if j_pt < pt_cut:
                    continue
                if len(gf_eta_evt) == 0:
                    continue

                drs_rg = calc_dr(j_eta, j_phi, gf_eta_evt, gf_phi_evt)
                closest_gen = int(np.argmin(drs_rg))
                closest_dr = float(drs_rg[closest_gen])
                if closest_dr >= matching_r:
                    continue
                if closest_gen not in genfatjet_best:
                    continue

                match_info = genfatjet_best[closest_gen]
                cat = match_info["category"]
                if not should_keep_category(cat, keep_mode):
                    continue

                dh_pts = visible_dark_hadron_pts_in_reco_jet(
                    pids_evt=pids_evt,
                    status_evt=status_evt,
                    pt_evt=pt_evt,
                    eta_evt=eta_evt,
                    phi_evt=phi_evt,
                    d1_evt=d1_evt,
                    d2_evt=d2_evt,
                    j_eta=j_eta,
                    j_phi=j_phi,
                    fatjet_r=fatjet_r,
                    require_status_8384=require_status_8384,
                    visible_desc_must_be_inside_jet=visible_desc_must_be_inside_jet,
                )

                nvdh = len(dh_pts)
                dh1_pt = dh_pts[0] if nvdh >= 1 else np.nan
                dh2_pt = dh_pts[1] if nvdh >= 2 else np.nan
                ratio = dh2_pt / dh1_pt if nvdh >= 2 and dh1_pt > 0 else np.nan

                cols["true_mass"].append(float(target_mpi))
                cols["sdmass"].append(safe_softdrop_mass(rf_sdmass, iev, r_idx))
                cols["sdpt"].append(safe_softdrop_pt(rf_sdpt, iev, r_idx))
                cols["reco_pt"].append(j_pt)
                cols["reco_mass"].append(j_m)
                cols["n_visible_dh"].append(int(nvdh))
                cols["dh1_pt"].append(float(dh1_pt) if np.isfinite(dh1_pt) else np.nan)
                cols["dh2_pt"].append(float(dh2_pt) if np.isfinite(dh2_pt) else np.nan)
                cols["dh2_over_dh1"].append(float(ratio) if np.isfinite(ratio) else np.nan)
                cols["containment_category"].append(CATEGORY_RANK.get(cat, -1))
                cols["containment_fraction"].append(float(match_info.get("frac_inside", np.nan)))
                cols["root_file"].append(root_file)

    except Exception as exc:
        print(f"[WARN] Failed file: {root_file}\n{exc}\n{traceback.format_exc()}", flush=True)
    finally:
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    return cols


def merge_columns(results: Iterable[Dict[str, list]]) -> Dict[str, np.ndarray]:
    merged = empty_columns()
    for res in results:
        for key in merged:
            merged[key].extend(res.get(key, []))
    out = {}
    for key, vals in merged.items():
        if key == "root_file":
            out[key] = np.asarray(vals, dtype=object)
        elif key in ["n_visible_dh", "containment_category"]:
            out[key] = np.asarray(vals, dtype=np.int16)
        else:
            out[key] = np.asarray(vals, dtype=np.float32)
    return out


# =============================================================================
# Plotting
# =============================================================================
def valid_base_mask(data: Dict[str, np.ndarray], mass_range, sdmass_range) -> np.ndarray:
    true_m = data["true_mass"]
    sd = data["sdmass"]
    m = np.isfinite(true_m) & np.isfinite(sd) & (true_m > 0) & (sd > 0)
    if mass_range is not None:
        m &= (true_m >= mass_range[0]) & (true_m <= mass_range[1])
    if sdmass_range is not None:
        m &= (sd >= sdmass_range[0]) & (sd <= sdmass_range[1])
    return m


def draw_heatmap_page(
    pdf: PdfPages,
    true_m: np.ndarray,
    sd_m: np.ndarray,
    title: str,
    mass_range: Tuple[float, float],
    sdmass_range: Tuple[float, float],
    bins: int,
    extra_text: str = "",
) -> None:
    true_m = np.asarray(true_m).reshape(-1)
    sd_m = np.asarray(sd_m).reshape(-1)
    m = np.isfinite(true_m) & np.isfinite(sd_m) & (true_m > 0) & (sd_m > 0)
    true_m = true_m[m]
    sd_m = sd_m[m]
    tau = compute_tau(true_m, sd_m)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    if len(true_m) == 0:
        ax.text(0.5, 0.5, "No valid entries", ha="center", va="center", transform=ax.transAxes)
    else:
        h = ax.hist2d(
            true_m,
            sd_m,
            bins=bins,
            range=[mass_range, sdmass_range],
            norm=LogNorm(),
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")
        ax.plot(mass_range, mass_range, linestyle="--", linewidth=1.2, color="black", label="y = x")
        ax.legend(loc="upper left", frameon=True)

    text = f"N = {len(true_m):,}\nKendall tau = {tau:.4f}"
    if extra_text:
        text += "\n" + extra_text
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    style_ax(
        ax,
        xlabel=r"True dark-hadron mass $m_{\mathrm{DH}}$ [GeV]",
        ylabel=r"SoftDrop mass $m_{\mathrm{SD}}$ [GeV]",
        title=title,
    )
    ax.set_xlim(*mass_range)
    ax.set_ylim(*sdmass_range)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_ratio_distribution(data, outdir: Path, pdf: PdfPages, ratio_bins: int = 60) -> None:
    ratio = data["dh2_over_dh1"]
    nvdh = data["n_visible_dh"]
    m = np.isfinite(ratio) & (nvdh >= 2) & (ratio >= 0) & (ratio <= 1.5)
    x = ratio[m]

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(x) > 0:
        ax.hist(x, bins=ratio_bins, range=(0, 1), histtype="step", linewidth=2.2, density=True)
        ax.axvline(np.nanmedian(x), linestyle="--", linewidth=1.5, label=f"median={np.nanmedian(x):.3f}")
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, "No N_DH^vis >= 2 entries", ha="center", va="center", transform=ax.transAxes)
    style_ax(
        ax,
        xlabel=r"$p_T(\mathrm{DH}_2) / p_T(\mathrm{DH}_1)$",
        ylabel="Density",
        title=r"Visible-DH pT balance for $N_{\mathrm{DH}}^{vis}\geq2$",
    )
    fig.tight_layout()
    fig.savefig(outdir / "dh2_over_dh1_ratio_distribution.pdf")
    pdf.savefig(fig)
    plt.close(fig)


def plot_residual_overlays(data, outdir: Path, pdf: PdfPages, ratio_cuts: Sequence[float]) -> None:
    true_m = data["true_mass"]
    sd = data["sdmass"]
    ratio = data["dh2_over_dh1"]
    nvdh = data["n_visible_dh"]

    base = np.isfinite(true_m) & np.isfinite(sd) & np.isfinite(ratio) & (true_m > 0) & (sd > 0) & (nvdh >= 2)
    rel = (sd - true_m) / true_m

    fig, ax = plt.subplots(figsize=(8, 6))
    for cut in ratio_cuts:
        m = base & (ratio >= cut)
        if np.sum(m) < 5:
            continue
        ax.hist(
            rel[m],
            bins=80,
            range=(-2, 5),
            density=True,
            histtype="step",
            linewidth=2.0,
            label=fr"$r_{{21}}>={cut:.2f}$, N={np.sum(m):,}",
        )
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    style_ax(
        ax,
        xlabel=r"$(m_{\mathrm{SD}}-m_{\mathrm{DH}})/m_{\mathrm{DH}}$",
        ylabel="Density",
        title=r"SoftDrop relative residual after $p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$ cuts",
    )
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "sdmass_relative_residual_ratio_cut_overlay.pdf")
    pdf.savefig(fig)
    plt.close(fig)


def plot_ratio_cut_summary(data, outdir: Path, pdf: PdfPages, ratio_cuts: Sequence[float]) -> List[dict]:
    true_m = data["true_mass"]
    sd = data["sdmass"]
    ratio = data["dh2_over_dh1"]
    nvdh = data["n_visible_dh"]

    rows = []
    for cut in ratio_cuts:
        m = (
            np.isfinite(true_m) & np.isfinite(sd) & np.isfinite(ratio)
            & (true_m > 0) & (sd > 0) & (nvdh >= 2) & (ratio >= cut)
        )
        rel = (sd[m] - true_m[m]) / true_m[m] if np.sum(m) else np.array([])
        row = {
            "ratio_cut_min": float(cut),
            "entries": int(np.sum(m)),
            "kendall_tau_sdmass_true_mass": compute_tau(true_m[m], sd[m]) if np.sum(m) >= 3 else float("nan"),
            "median_rel_residual": float(np.nanmedian(rel)) if len(rel) else float("nan"),
            "mean_rel_residual": float(np.nanmean(rel)) if len(rel) else float("nan"),
            "std_rel_residual": float(np.nanstd(rel)) if len(rel) else float("nan"),
            "iqr68_rel_residual": robust_rel_width(rel),
            "median_sdmass": float(np.nanmedian(sd[m])) if np.sum(m) else float("nan"),
            "median_true_mass": float(np.nanmedian(true_m[m])) if np.sum(m) else float("nan"),
        }
        rows.append(row)

    cuts = np.array([r["ratio_cut_min"] for r in rows], dtype=float)
    entries = np.array([r["entries"] for r in rows], dtype=float)
    tau = np.array([r["kendall_tau_sdmass_true_mass"] for r in rows], dtype=float)
    med = np.array([r["median_rel_residual"] for r in rows], dtype=float)
    width = np.array([r["iqr68_rel_residual"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cuts, entries, marker="o", linewidth=2.0)
    style_ax(
        ax,
        xlabel=r"Minimum $p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$",
        ylabel="Selected jets",
        title=r"Yield vs visible-DH pT-balance cut, $N_{\mathrm{DH}}^{vis}\geq2$",
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(outdir / "ratio_cut_yield_curve.pdf")
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cuts, tau, marker="o", linewidth=2.0)
    style_ax(
        ax,
        xlabel=r"Minimum $p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$",
        ylabel=r"Kendall $\tau(m_{SD},m_{DH})$",
        title="SoftDrop rank correlation vs visible-DH pT-balance cut",
    )
    fig.tight_layout()
    fig.savefig(outdir / "ratio_cut_kendalltau_curve.pdf")
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cuts, med, marker="o", linewidth=2.0, label="median")
    ax.plot(cuts, width, marker="s", linewidth=2.0, label="68% half-width")
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    style_ax(
        ax,
        xlabel=r"Minimum $p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$",
        ylabel=r"Relative residual metric",
        title="SoftDrop residual metrics vs visible-DH pT-balance cut",
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "ratio_cut_residual_metrics_curve.pdf")
    pdf.savefig(fig)
    plt.close(fig)

    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_ndark_pages(data, outdir: Path, args) -> List[dict]:
    rows = []
    pdf_path = outdir / "sdmass_vs_true_mass_by_nvisibleDH_pages.pdf"
    with PdfPages(pdf_path) as pdf:
        for n in range(args.ndark_min, args.ndark_max + 1):
            m = valid_base_mask(data, args.mass_range, args.sdmass_range) & (data["n_visible_dh"] == n)
            true_m = data["true_mass"][m]
            sd = data["sdmass"][m]
            rel = (sd - true_m) / true_m if len(true_m) else np.array([])
            rows.append({
                "n_visible_dh": int(n),
                "entries": int(len(true_m)),
                "kendall_tau_sdmass_true_mass": compute_tau(true_m, sd),
                "median_rel_residual": float(np.nanmedian(rel)) if len(rel) else float("nan"),
                "iqr68_rel_residual": robust_rel_width(rel),
            })
            draw_heatmap_page(
                pdf=pdf,
                true_m=true_m,
                sd_m=sd,
                title=fr"SoftDrop mass vs true mass, $N_{{\mathrm{{DH}}}}^{{vis}}={n}$",
                mass_range=args.mass_range,
                sdmass_range=args.sdmass_range,
                bins=args.bins,
                extra_text="visible DHs ranked by gen-pT",
            )
    print(f"[SAVED] {pdf_path}")
    write_csv(outdir / "sdmass_vs_true_mass_by_nvisibleDH_summary.csv", rows)
    return rows


def make_ratio_cut_pages(data, outdir: Path, args, ratio_rows: List[dict]) -> None:
    pdf_path = outdir / "sdmass_vs_true_mass_by_DH2overDH1_cut_pages.pdf"
    with PdfPages(pdf_path) as pdf:
        for cut in args.ratio_cuts:
            m = (
                valid_base_mask(data, args.mass_range, args.sdmass_range)
                & (data["n_visible_dh"] >= 2)
                & np.isfinite(data["dh2_over_dh1"])
                & (data["dh2_over_dh1"] >= cut)
            )
            true_m = data["true_mass"][m]
            sd = data["sdmass"][m]
            rel = (sd - true_m) / true_m if len(true_m) else np.array([])
            extra = ""
            if len(rel):
                extra = f"median rel = {np.nanmedian(rel):.3f}\n68% half-width = {robust_rel_width(rel):.3f}"
            draw_heatmap_page(
                pdf=pdf,
                true_m=true_m,
                sd_m=sd,
                title=fr"$m_{{SD}}$ vs $m_{{DH}}$, $N_{{DH}}^{{vis}}\geq2$, $p_T(DH_2)/p_T(DH_1)\geq{cut:.2f}$",
                mass_range=args.mass_range,
                sdmass_range=args.sdmass_range,
                bins=args.bins,
                extra_text=extra,
            )
    print(f"[SAVED] {pdf_path}")


def make_summary_pdf(data, outdir: Path, args) -> List[dict]:
    pdf_path = outdir / "visibleDH_ratio_softdrop_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_ratio_distribution(data, outdir, pdf)
        plot_residual_overlays(data, outdir, pdf, args.ratio_cuts)
        ratio_rows = plot_ratio_cut_summary(data, outdir, pdf, args.ratio_cuts)
    print(f"[SAVED] {pdf_path}")
    write_csv(outdir / "DH2overDH1_ratio_cut_softdrop_summary.csv", ratio_rows)
    return ratio_rows


def write_overall_summary(data, outdir: Path, args, files: Sequence[str]) -> None:
    rows = [{
        "n_input_files": len(files),
        "n_selected_jets": int(len(data["true_mass"])),
        "n_valid_sdmass": int(np.sum(np.isfinite(data["sdmass"]) & (data["sdmass"] > 0))),
        "n_visible_dh_ge_2": int(np.sum(data["n_visible_dh"] >= 2)),
        "pt_cut": args.pt_cut,
        "fatjet_r": args.fatjet_r,
        "matching_r": args.matching_r,
        "keep_mode": args.keep_mode,
        "require_status_8384": int(args.require_status_8384),
        "visible_desc_must_be_inside_jet": int(args.visible_desc_must_be_inside_jet),
        "mass_min": args.mass_range[0],
        "mass_max": args.mass_range[1],
        "sdmass_min": args.sdmass_range[0],
        "sdmass_max": args.sdmass_range[1],
    }]
    write_csv(outdir / "overall_plot_summary.csv", rows)


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot visible-DH pT balance and SoftDrop mass behavior vs N_visible_DH, without saving a new dataset."
    )
    parser.add_argument("--filelist", default=None, help="Text file with ROOT paths, one per line.")
    parser.add_argument("--input-glob", nargs="*", default=None, help="Alternative: local ROOT glob(s).")
    parser.add_argument("--eos-host", default="", help="Prefix for /store paths, e.g. root://cmseos.fnal.gov")
    parser.add_argument("--outdir", required=True, help="Directory where plots and CSV summaries are written.")

    parser.add_argument("--pt-cut", type=float, default=20.0, help="Reco FatJet pT cut.")
    parser.add_argument("--fatjet-r", type=float, default=0.8, help="Reco FatJet cone used for DH counting.")
    parser.add_argument("--matching-r", type=float, default=0.4, help="Reco FatJet to GenFatJet matching cone.")
    parser.add_argument(
        "--keep-mode",
        default="full_only",
        choices=["full_only", "partial_only", "none_only", "full_plus_partial", "partial_plus_none", "not_full", "all"],
        help="Containment category selection, matching your dataset-builder convention.",
    )
    parser.add_argument(
        "--no-require-status-8384",
        dest="require_status_8384",
        action="store_false",
        help="Count all dark hadrons regardless of status. Default counts status 83/84 only.",
    )
    parser.set_defaults(require_status_8384=True)
    parser.add_argument(
        "--visible-desc-must-be-inside-jet",
        action="store_true",
        help="Require a visible stable descendant inside the reco FatJet cone. Default: DH axis inside jet + any visible descendant.",
    )

    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of files.")
    parser.add_argument("--max-events-per-file", type=int, default=None, help="Optional cap on events per file.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel ROOT files to process.")

    parser.add_argument("--mass-min", type=float, default=1.0)
    parser.add_argument("--mass-max", type=float, default=200.0)
    parser.add_argument("--sdmass-min", type=float, default=0.0)
    parser.add_argument("--sdmass-max", type=float, default=250.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--ndark-min", type=int, default=1)
    parser.add_argument("--ndark-max", type=int, default=10)
    parser.add_argument(
        "--ratio-cuts",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Minimum DH2/DH1 pT-ratio cuts to scan.",
    )

    args = parser.parse_args()
    args.mass_range = (args.mass_min, args.mass_max)
    args.sdmass_range = (args.sdmass_min, args.sdmass_max)
    if not args.filelist and not args.input_glob:
        parser.error("Provide --filelist or --input-glob")
    return args


def main() -> None:
    args = parse_args()
    setup_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(args)
    if not files:
        raise RuntimeError("No input ROOT files found.")

    print("==========================================================")
    print("Visible-DH pT ratio / SoftDrop plotting study")
    print(f"Input files      : {len(files):,}")
    print(f"Outdir           : {outdir}")
    print(f"keep_mode        : {args.keep_mode}")
    print(f"pt_cut           : {args.pt_cut}")
    print(f"workers          : {args.workers}")
    print("==========================================================")

    tasks = [
        (
            f,
            args.keep_mode,
            args.max_events_per_file,
            args.fatjet_r,
            args.matching_r,
            args.pt_cut,
            args.require_status_8384,
            args.visible_desc_must_be_inside_jet,
        )
        for f in files
    ]

    results = []
    if args.workers <= 1:
        for i, task in enumerate(tasks, start=1):
            if i % 50 == 0 or i == 1 or i == len(tasks):
                print(f"[INFO] Processing file {i}/{len(tasks)}", flush=True)
            results.append(process_one_file(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one_file, task) for task in tasks]
            for i, fut in enumerate(as_completed(futs), start=1):
                if i % 50 == 0 or i == 1 or i == len(futs):
                    print(f"[INFO] Finished file {i}/{len(futs)}", flush=True)
                results.append(fut.result())

    data = merge_columns(results)
    print(f"[INFO] Selected jets: {len(data['true_mass']):,}")
    print(f"[INFO] N_visible_DH >= 2: {np.sum(data['n_visible_dh'] >= 2):,}")

    write_overall_summary(data, outdir, args, files)

    # Main requested outputs
    make_ndark_pages(data, outdir, args)
    ratio_rows = make_summary_pdf(data, outdir, args)
    make_ratio_cut_pages(data, outdir, args, ratio_rows)

    print("==========================================================")
    print("Done. Key outputs:")
    print(f"  {outdir / 'sdmass_vs_true_mass_by_nvisibleDH_pages.pdf'}")
    print(f"  {outdir / 'visibleDH_ratio_softdrop_summary.pdf'}")
    print(f"  {outdir / 'sdmass_vs_true_mass_by_DH2overDH1_cut_pages.pdf'}")
    print(f"  {outdir / 'DH2overDH1_ratio_cut_softdrop_summary.csv'}")
    print(f"  {outdir / 'sdmass_vs_true_mass_by_nvisibleDH_summary.csv'}")
    print("==========================================================")


if __name__ == "__main__":
    main()
