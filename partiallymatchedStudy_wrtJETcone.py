#!/usr/bin/env python3
"""
partial_match_study_root_lpc.py

ROOT-level matched-fraction study for DRACON/SVJ Delphes files.

This version loops over Delphes ROOT files directly and makes ROOT-level
2D jet/decay containment displays. It does not need a prebuilt NPZ.

Main outputs in --outdir:
  partial_match_root_summary.pdf
  sdmass_vs_mdark_by_partial_fraction.pdf
  sdmass_vs_mdark_unmatched.pdf
  sdmass_vs_mdark_fully_matched.pdf
  example_partial_fraction_jets_root_2d.pdf
  example_partial_fraction_subplots_2d.pdf
  partial_fraction_summary.csv
  representative_jets.csv
  selected_jets_summary.csv
  failed_files.csv

Display convention:
  - gray points are PF candidates inside the reco AK8 cone only
  - green circles are visible DH descendants inside the reco jet
  - red x markers are visible DH descendants outside the reco jet
  - gold stars mark visible dark hadrons inside the reco jet

Default physics choices:
  - reco FatJet pT > 20 GeV
  - reco FatJet matched to nearest GenFatJet within DeltaR < 0.4
  - matched fraction = visible final-state descendants of the matched dark
    hadron inside the RECO FatJet cone / all visible final-state descendants
  - --keep-mode all by default so unmatched, partial, and fully matched jets
    are available for the requested comparison pages.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

try:
    import awkward as ak
    import uproot
except Exception as exc:
    raise RuntimeError(
        "This script needs uproot and awkward. Activate your LPC python/conda "
        "environment first, or install them with pip --user."
    ) from exc

try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None


# =============================================================================
# Physics conventions
# =============================================================================
DARK_HADRONS = {
    4900111,     # piD0
    4900211,     # piD+
    -4900211,    # piD-
    4900113,     # rhoD0
    4900213,     # rhoD+
    -4900213,    # rhoD-
}

DARK_HADRON_COUNT_STATUSES = {83, 84}

INVISIBLE_PIDS_ABS = {
    12, 14, 16, 18,
    4900101, 4900102, 4900103, 4900104, 4900105, 4900106,
}

PID_NAMES = {
    1: "d", -1: "dbar", 2: "u", -2: "ubar", 3: "s", -3: "sbar",
    4: "c", -4: "cbar", 5: "b", -5: "bbar", 6: "t", -6: "tbar",
    11: "e-", -11: "e+", 12: "nu_e", -12: "nu_e_bar",
    13: "mu-", -13: "mu+", 14: "nu_mu", -14: "nu_mu_bar",
    15: "tau-", -15: "tau+", 16: "nu_tau", -16: "nu_tau_bar",
    21: "g", 22: "gamma", 111: "pi0", 211: "pi+", -211: "pi-",
    321: "K+", -321: "K-", 130: "K_L", 310: "K_S",
    2212: "p", -2212: "pbar", 2112: "n", -2112: "nbar",
    4900111: "piD0", 4900211: "piD+", -4900211: "piD-",
    4900113: "rhoD0", 4900213: "rhoD+", -4900213: "rhoD-",
    4900101: "dark_q1", -4900101: "dark_q1bar",
    4900102: "dark_q2", -4900102: "dark_q2bar",
    4900103: "dark_q3", -4900103: "dark_q3bar",
    4900104: "dark_q4", -4900104: "dark_q4bar",
    4900105: "dark_q5", -4900105: "dark_q5bar",
    4900106: "dark_q6", -4900106: "dark_q6bar",
    4900021: "dark_g",
}

CATEGORY_CODE = {"none": 0, "partial": 1, "full": 2}


# =============================================================================
# General helpers
# =============================================================================
def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
    })


def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, pad=8)
    try:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    except Exception:
        pass
    if grid:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)


def calc_dphi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2.0 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + calc_dphi(phi1, phi2) ** 2)


def center_coords(etas, phis, center_eta, center_phi):
    return np.asarray(etas) - center_eta, calc_dphi(np.asarray(phis), center_phi)


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
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    m = finite_mask(x, y)
    x = x[m]
    y = y[m]
    if len(x) < 3 or kendalltau is None:
        return float("nan")
    val = kendalltau(x, y, nan_policy="omit").correlation
    return float(val) if val is not None and np.isfinite(val) else float("nan")


def rel_metrics(true_m, pred_m) -> Dict[str, float]:
    true_m = np.asarray(true_m).reshape(-1)
    pred_m = np.asarray(pred_m).reshape(-1)
    if len(true_m) == 0 or len(pred_m) == 0:
        return {"rel_median": np.nan, "rel_mean": np.nan, "rel_width68": np.nan}
    m = finite_mask(true_m, pred_m) & (true_m > 0) & (pred_m > 0)
    if not np.any(m):
        return {"rel_median": np.nan, "rel_mean": np.nan, "rel_width68": np.nan}
    rel = (pred_m[m] - true_m[m]) / true_m[m]
    if len(rel) < 5:
        width = np.nan
    else:
        q16, q84 = np.nanpercentile(rel, [16, 84])
        width = 0.5 * (q84 - q16)
    return {
        "rel_median": float(np.nanmedian(rel)),
        "rel_mean": float(np.nanmean(rel)),
        "rel_width68": float(width),
    }


def fraction_slices() -> List[Tuple[float, float, float]]:
    out = []
    for c in np.arange(0.1, 1.0, 0.1):
        lo = max(0.0, float(c - 0.05))
        hi = min(1.0, float(c + 0.05))
        out.append((float(round(c, 1)), lo, hi))
    return out


def safe_name(x: float) -> str:
    return f"{x:.1f}".replace(".", "p")


def pid_name(pid: int) -> str:
    return PID_NAMES.get(int(pid), str(int(pid)))


# =============================================================================
# Input path helpers
# =============================================================================
def normalize_root_path(path: str, eos_host: str = "") -> str:
    path = path.strip()
    if not path:
        return path

    # LPC/XRootD wants root://cmseos.fnal.gov//store/..., not one slash.
    if path.startswith("root://"):
        m = re.match(r"^(root://[^/]+)/(store/.*)$", path)
        if m:
            return m.group(1).rstrip("/") + "//" + m.group(2)
        return path

    if eos_host and path.startswith("/store/"):
        return eos_host.rstrip("/") + "//" + path.lstrip("/")

    return path


def read_filelist(path: str, eos_host: str = "") -> List[str]:
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(normalize_root_path(s, eos_host=eos_host))
    return out


def collect_input_files(args) -> List[str]:
    files: List[str] = []
    if args.filelist:
        files.extend(read_filelist(args.filelist, eos_host=args.eos_host))
    if args.input_glob:
        for g in args.input_glob:
            files.extend(sorted(glob.glob(g)))
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    if args.max_files is not None and args.max_files > 0:
        uniq = uniq[: args.max_files]
    return uniq


def parse_mpi_from_path(path: str) -> Optional[float]:
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


# =============================================================================
# ROOT branch helpers
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


# =============================================================================
# Matching and decay helpers
# =============================================================================
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


def get_visible_descendants(dh_idx, pids_evt, status_evt, eta_evt, phi_evt, d1_evt, d2_evt) -> List[int]:
    desc = sorted(set(get_descendants(int(dh_idx), d1_evt, d2_evt, len(pids_evt))))
    vis = []
    for child_idx in desc:
        if is_visible_final_state(pids_evt[child_idx], status_evt[child_idx]):
            if np.isfinite(eta_evt[child_idx]) and np.isfinite(phi_evt[child_idx]):
                vis.append(int(child_idx))
    return vis


def fraction_category(frac: float) -> str:
    if not np.isfinite(frac) or frac <= 0.0:
        return "none"
    if frac >= 1.0:
        return "full"
    return "partial"


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


def build_genfatjet_dh_map(
    pids_evt,
    status_evt,
    pt_evt,
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
    For each GenFatJet, keep the dark hadron with the largest visible-descendant
    multiplicity, then largest pT. The actual matched fraction is recomputed
    later relative to reco or genfatjet depending on --fraction-reference.
    """
    out: Dict[int, dict] = {}
    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]

    for dh_idx in dh_indices:
        dh_eta = eta_evt[dh_idx]
        dh_phi = phi_evt[dh_idx]
        if not np.isfinite(dh_eta) or not np.isfinite(dh_phi):
            continue
        if len(gf_eta_evt) == 0:
            continue

        drs_dh_gf = calc_dr(dh_eta, dh_phi, gf_eta_evt, gf_phi_evt)
        best_gf_idx = int(np.argmin(drs_dh_gf))
        best_dr = float(drs_dh_gf[best_gf_idx])
        if best_dr >= fatjet_r:
            continue

        vis_desc = get_visible_descendants(dh_idx, pids_evt, status_evt, eta_evt, phi_evt, d1_evt, d2_evt)
        if len(vis_desc) == 0:
            continue

        candidate = {
            "dh_idx": int(dh_idx),
            "dh_pid": int(pids_evt[dh_idx]),
            "dh_pt": float(pt_evt[dh_idx]) if np.isfinite(pt_evt[dh_idx]) else -1.0,
            "dh_mass": float(mass_evt[dh_idx]) if np.isfinite(mass_evt[dh_idx]) else -1.0,
            "gf_idx": int(best_gf_idx),
            "dr_dh_gf": float(best_dr),
            "vis_desc": vis_desc,
            "n_vis_desc": int(len(vis_desc)),
        }
        old = out.get(best_gf_idx)
        if old is None:
            out[best_gf_idx] = candidate
        else:
            old_key = (old["n_vis_desc"], old["dh_pt"])
            new_key = (candidate["n_vis_desc"], candidate["dh_pt"])
            if new_key > old_key:
                out[best_gf_idx] = candidate
    return out


def compute_visible_desc_fraction(vis_desc: Sequence[int], eta_evt, phi_evt, axis_eta, axis_phi, radius) -> Tuple[float, int, int, np.ndarray]:
    n_total = len(vis_desc)
    if n_total == 0:
        return 0.0, 0, 0, np.zeros(0, dtype=bool)
    desc_eta = eta_evt[list(vis_desc)]
    desc_phi = phi_evt[list(vis_desc)]
    dr = calc_dr(desc_eta, desc_phi, axis_eta, axis_phi)
    inside = dr < radius
    n_inside = int(np.sum(inside))
    return float(n_inside / max(n_total, 1)), n_inside, n_total, np.asarray(inside, dtype=bool)


def count_dark_hadrons_in_reco_jet(
    pids_evt,
    status_evt,
    eta_evt,
    phi_evt,
    d1_evt,
    d2_evt,
    jet_eta,
    jet_phi,
    fatjet_r,
    require_status_8384=True,
) -> Tuple[int, int]:
    n_dark = 0
    n_visible_dark = 0
    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]
    for dh_idx in dh_indices:
        if require_status_8384 and int(status_evt[dh_idx]) not in DARK_HADRON_COUNT_STATUSES:
            continue
        if not np.isfinite(eta_evt[dh_idx]) or not np.isfinite(phi_evt[dh_idx]):
            continue
        if float(calc_dr(eta_evt[dh_idx], phi_evt[dh_idx], jet_eta, jet_phi)) >= fatjet_r:
            continue
        n_dark += 1
        vis_desc = get_visible_descendants(dh_idx, pids_evt, status_evt, eta_evt, phi_evt, d1_evt, d2_evt)
        if len(vis_desc) > 0:
            n_visible_dark += 1
    return n_dark, n_visible_dark


def get_visible_dark_hadrons_in_reco_jet(
    pids_evt,
    status_evt,
    pt_evt,
    eta_evt,
    phi_evt,
    d1_evt,
    d2_evt,
    jet_eta,
    jet_phi,
    fatjet_r,
    require_status_8384=True,
) -> List[dict]:
    """Return visible dark hadrons whose axes are inside the reco jet cone."""
    out = []
    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]
    for dh_idx in dh_indices:
        if require_status_8384 and int(status_evt[dh_idx]) not in DARK_HADRON_COUNT_STATUSES:
            continue
        if not (np.isfinite(pt_evt[dh_idx]) and np.isfinite(eta_evt[dh_idx]) and np.isfinite(phi_evt[dh_idx])):
            continue
        if float(calc_dr(eta_evt[dh_idx], phi_evt[dh_idx], jet_eta, jet_phi)) >= fatjet_r:
            continue
        vis_desc = get_visible_descendants(dh_idx, pids_evt, status_evt, eta_evt, phi_evt, d1_evt, d2_evt)
        if len(vis_desc) == 0:
            continue
        de, dp = center_coords(np.array([eta_evt[dh_idx]]), np.array([phi_evt[dh_idx]]), jet_eta, jet_phi)
        out.append({
            "idx": int(dh_idx),
            "pid": int(pids_evt[dh_idx]),
            "name": pid_name(int(pids_evt[dh_idx])),
            "pt": float(pt_evt[dh_idx]),
            "eta": float(eta_evt[dh_idx]),
            "phi": float(phi_evt[dh_idx]),
            "deta": float(de[0]),
            "dphi": float(dp[0]),
        })
    out.sort(key=lambda x: x["pt"], reverse=True)
    return out


# =============================================================================
# Example display extraction
# =============================================================================
def make_display_payload(
    root_file,
    file_idx,
    event_idx,
    jet_idx,
    center,
    target_mpi,
    frac,
    frac_reco,
    frac_genfatjet,
    category,
    sdmass,
    sdpt,
    n_dark,
    n_visible_dark,
    n_inside,
    n_total,
    closest_gen,
    closest_dr,
    dh_info,
    pids_evt,
    status_evt,
    pt_evt,
    eta_evt,
    phi_evt,
    d1_evt,
    d2_evt,
    gf_eta_evt,
    gf_phi_evt,
    rfs_pt,
    rfs_eta,
    rfs_phi,
    rfs_m,
    pfs_pt,
    pfs_eta,
    pfs_phi,
    fatjet_r,
    max_pf_display,
    max_decay_nodes,
    require_status_8384=True,
) -> dict:
    j_pt = float(rfs_pt[jet_idx])
    j_eta = float(rfs_eta[jet_idx])
    j_phi = float(rfs_phi[jet_idx])
    j_m = float(rfs_m[jet_idx])

    # PF candidates shown in the display are ONLY the PF candidates inside the reco jet cone.
    pf_payload = {"deta": [], "dphi": [], "pt": []}
    if pfs_pt is not None and len(pfs_pt) > 0:
        pdr = calc_dr(pfs_eta, pfs_phi, j_eta, j_phi)
        pmask = np.isfinite(pfs_pt) & np.isfinite(pfs_eta) & np.isfinite(pfs_phi) & (pfs_pt > 0) & (pdr < fatjet_r)
        idx = np.where(pmask)[0]
        if len(idx) > max_pf_display:
            order = idx[np.argsort(pfs_pt[idx])[::-1]]
            idx = order[:max_pf_display]
        de, dp = center_coords(pfs_eta[idx], pfs_phi[idx], j_eta, j_phi)
        pf_payload = {
            "deta": de.astype(np.float32),
            "dphi": dp.astype(np.float32),
            "pt": np.asarray(pfs_pt[idx], dtype=np.float32),
        }

    visible_dh = get_visible_dark_hadrons_in_reco_jet(
        pids_evt=pids_evt,
        status_evt=status_evt,
        pt_evt=pt_evt,
        eta_evt=eta_evt,
        phi_evt=phi_evt,
        d1_evt=d1_evt,
        d2_evt=d2_evt,
        jet_eta=j_eta,
        jet_phi=j_phi,
        fatjet_r=fatjet_r,
        require_status_8384=require_status_8384,
    )
    visible_dh_idx = {row["idx"] for row in visible_dh}

    dh_idx = int(dh_info["dh_idx"])
    desc_all = sorted(set(get_descendants(dh_idx, d1_evt, d2_evt, len(pids_evt))))
    if len(desc_all) > max_decay_nodes:
        desc_order = sorted(desc_all, key=lambda i: float(pt_evt[i]) if np.isfinite(pt_evt[i]) else -1.0, reverse=True)
        keep = set([dh_idx] + desc_order[: max_decay_nodes - 1])
        nodes = [dh_idx] + [i for i in desc_all if i in keep and i != dh_idx]
    else:
        nodes = [dh_idx] + desc_all

    node_rows = []
    for idx in nodes:
        if not (np.isfinite(pt_evt[idx]) and np.isfinite(eta_evt[idx]) and np.isfinite(phi_evt[idx])):
            continue
        de, dp = center_coords(np.array([eta_evt[idx]]), np.array([phi_evt[idx]]), j_eta, j_phi)
        is_vis = is_visible_final_state(pids_evt[idx], status_evt[idx])
        inside_reco = bool(float(calc_dr(eta_evt[idx], phi_evt[idx], j_eta, j_phi)) < fatjet_r)
        node_rows.append({
            "idx": int(idx),
            "pid": int(pids_evt[idx]),
            "status": int(status_evt[idx]),
            "pt": float(pt_evt[idx]),
            "eta": float(eta_evt[idx]),
            "phi": float(phi_evt[idx]),
            "deta": float(de[0]),
            "dphi": float(dp[0]),
            "is_dh": int(idx == dh_idx),
            "is_visible_dark": int(idx in visible_dh_idx),
            "is_visible_final": int(is_vis),
            "inside_reco": int(inside_reco),
        })

    meta = {
        "root_file": str(root_file),
        "file_idx": int(file_idx),
        "event_idx": int(event_idx),
        "jet_idx": int(jet_idx),
        "fraction_center": float(center),
        "target_mpi": float(target_mpi) if target_mpi is not None and np.isfinite(target_mpi) else np.nan,
        "matched_fraction": float(frac),
        "matched_fraction_reco": float(frac_reco),
        "matched_fraction_genfatjet": float(frac_genfatjet),
        "category": category,
        "sdmass": float(sdmass),
        "sdpt": float(sdpt),
        "n_dark": int(n_dark),
        "n_visible_dark": int(n_visible_dark),
        "n_inside_visible_desc": int(n_inside),
        "n_total_visible_desc": int(n_total),
        "matched_dh_idx": int(dh_idx),
        "matched_dh_pid": int(dh_info["dh_pid"]),
        "matched_dh_name": pid_name(int(dh_info["dh_pid"])),
        "matched_dh_pt": float(dh_info["dh_pt"]),
        "matched_genfatjet_idx": int(closest_gen),
        "reco_genfatjet_dr": float(closest_dr),
        "reco_pt": float(j_pt),
        "reco_eta": float(j_eta),
        "reco_phi": float(j_phi),
        "reco_mass": float(j_m),
    }
    return {"meta": meta, "pf": pf_payload, "nodes": node_rows, "visible_dh": visible_dh}


# =============================================================================
# Per-file processing
# =============================================================================
def empty_result(root_file, file_idx):
    return {
        "ok": False,
        "root_file": root_file,
        "file_idx": file_idx,
        "error": None,
        "traceback": None,
        "n_events": 0,
        "n_total_recojets": 0,
        "n_ptcut_recojets": 0,
        "n_matched_recojets": 0,
        "n_selected_jets": 0,
        "arrays": {
            "true_mass": [],
            "frac": [],
            "frac_reco": [],
            "frac_genfatjet": [],
            "sdmass": [],
            "sdpt": [],
            "ndark": [],
            "n_visible_dark": [],
            "n_visible_desc": [],
            "n_inside_visible_desc": [],
            "category_code": [],
            "reco_pt": [],
            "reco_mass": [],
            "reco_eta": [],
            "reco_phi": [],
            "file_idx": [],
            "event_idx": [],
            "jet_idx": [],
            "matched_dh_pid": [],
            "matched_dh_pt": [],
            "matched_genfatjet_idx": [],
            "reco_genfatjet_dr": [],
        },
        "examples": {},
    }


def process_one_file(job):
    (
        root_file,
        file_idx,
        keep_mode,
        max_events_per_file,
        fatjet_r,
        matching_r,
        pt_cut,
        require_status_8384,
        fraction_reference,
        mass_min,
        mass_max,
        sdmass_min,
        sdmass_max,
        example_centers,
        max_pf_display,
        max_decay_nodes,
    ) = job

    result = empty_result(root_file, file_idx)
    target_mpi = parse_mpi_from_path(root_file)
    if target_mpi is None:
        target_mpi = float("nan")

    try:
        root_handle, tree = get_delphes_tree(root_file)
        try:
            gen_pid = get_branch(tree, ["GenParticle/GenParticle.PID", "GenParticle.PID"])
            gen_pt = get_branch(tree, ["GenParticle/GenParticle.PT", "GenParticle.PT"])
            gen_eta = get_branch(tree, ["GenParticle/GenParticle.Eta", "GenParticle.Eta"])
            gen_phi = get_branch(tree, ["GenParticle/GenParticle.Phi", "GenParticle.Phi"])
            gen_mass = get_branch(tree, ["GenParticle/GenParticle.Mass", "GenParticle.Mass"])
            gen_status = get_branch(tree, ["GenParticle/GenParticle.Status", "GenParticle.Status"])
            gen_d1 = get_branch(tree, ["GenParticle/GenParticle.D1", "GenParticle.D1"])
            gen_d2 = get_branch(tree, ["GenParticle/GenParticle.D2", "GenParticle.D2"])

            gf_eta = get_branch(tree, ["GenFatJet/GenFatJet.Eta", "GenFatJet.Eta"])
            gf_phi = get_branch(tree, ["GenFatJet/GenFatJet.Phi", "GenFatJet.Phi"])

            rf_pt = get_branch(tree, ["FatJet/FatJet.PT", "FatJet.PT"])
            rf_eta = get_branch(tree, ["FatJet/FatJet.Eta", "FatJet.Eta"])
            rf_phi = get_branch(tree, ["FatJet/FatJet.Phi", "FatJet.Phi"])
            rf_m = get_branch(tree, ["FatJet/FatJet.Mass", "FatJet.Mass"])

            pf_pt = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.PT", "ParticleFlowCandidate.PT"], required=False)
            pf_eta = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Eta", "ParticleFlowCandidate.Eta"], required=False)
            pf_phi = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Phi", "ParticleFlowCandidate.Phi"], required=False)

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
            result["n_events"] = int(n_events)

            best_examples = {float(c): None for c in example_centers}

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

                if pf_pt is not None and pf_eta is not None and pf_phi is not None:
                    pfs_pt = ak.to_numpy(pf_pt[iev])
                    pfs_eta = ak.to_numpy(pf_eta[iev])
                    pfs_phi = ak.to_numpy(pf_phi[iev])
                else:
                    pfs_pt = np.array([], dtype=np.float32)
                    pfs_eta = np.array([], dtype=np.float32)
                    pfs_phi = np.array([], dtype=np.float32)

                genfatjet_dh = build_genfatjet_dh_map(
                    pids_evt=pids_evt,
                    status_evt=status_evt,
                    pt_evt=pt_evt,
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
                    result["n_total_recojets"] += 1
                    j_pt = float(rfs_pt[r_idx])
                    j_eta = float(rfs_eta[r_idx])
                    j_phi = float(rfs_phi[r_idx])
                    j_m = float(rfs_m[r_idx])
                    if not (np.isfinite(j_pt) and np.isfinite(j_eta) and np.isfinite(j_phi) and np.isfinite(j_m)):
                        continue
                    if j_pt < pt_cut:
                        continue
                    result["n_ptcut_recojets"] += 1
                    if len(gf_eta_evt) == 0:
                        continue

                    drs_rg = calc_dr(j_eta, j_phi, gf_eta_evt, gf_phi_evt)
                    closest_gen = int(np.argmin(drs_rg))
                    closest_dr = float(drs_rg[closest_gen])
                    if closest_dr >= matching_r:
                        continue
                    result["n_matched_recojets"] += 1
                    if closest_gen not in genfatjet_dh:
                        continue

                    dh_info = genfatjet_dh[closest_gen]
                    vis_desc = dh_info["vis_desc"]

                    frac_reco, n_inside_reco, n_total, _inside_reco = compute_visible_desc_fraction(
                        vis_desc, eta_evt, phi_evt, j_eta, j_phi, fatjet_r
                    )
                    frac_gf, n_inside_gf, _n_total_gf, _inside_gf = compute_visible_desc_fraction(
                        vis_desc, eta_evt, phi_evt, gf_eta_evt[closest_gen], gf_phi_evt[closest_gen], fatjet_r
                    )
                    if fraction_reference == "reco":
                        frac = frac_reco
                        n_inside = n_inside_reco
                    else:
                        frac = frac_gf
                        n_inside = n_inside_gf
                    cat = fraction_category(frac)
                    if not should_keep_category(cat, keep_mode):
                        continue

                    sdm = safe_softdrop_mass(rf_sdmass, iev, r_idx)
                    sdp = safe_softdrop_pt(rf_sdpt, iev, r_idx)
                    if not (np.isfinite(target_mpi) and target_mpi > 0):
                        continue
                    if not (np.isfinite(sdm) and sdm > 0):
                        continue
                    if not (mass_min <= target_mpi <= mass_max):
                        continue
                    if not (sdmass_min <= sdm <= sdmass_max):
                        continue

                    n_dark, n_visible_dark = count_dark_hadrons_in_reco_jet(
                        pids_evt=pids_evt,
                        status_evt=status_evt,
                        eta_evt=eta_evt,
                        phi_evt=phi_evt,
                        d1_evt=d1_evt,
                        d2_evt=d2_evt,
                        jet_eta=j_eta,
                        jet_phi=j_phi,
                        fatjet_r=fatjet_r,
                        require_status_8384=require_status_8384,
                    )

                    arr = result["arrays"]
                    arr["true_mass"].append(float(target_mpi))
                    arr["frac"].append(float(frac))
                    arr["frac_reco"].append(float(frac_reco))
                    arr["frac_genfatjet"].append(float(frac_gf))
                    arr["sdmass"].append(float(sdm))
                    arr["sdpt"].append(float(sdp))
                    arr["ndark"].append(int(n_dark))
                    arr["n_visible_dark"].append(int(n_visible_dark))
                    arr["n_visible_desc"].append(int(n_total))
                    arr["n_inside_visible_desc"].append(int(n_inside))
                    arr["category_code"].append(int(CATEGORY_CODE[cat]))
                    arr["reco_pt"].append(float(j_pt))
                    arr["reco_mass"].append(float(j_m))
                    arr["reco_eta"].append(float(j_eta))
                    arr["reco_phi"].append(float(j_phi))
                    arr["file_idx"].append(int(file_idx))
                    arr["event_idx"].append(int(iev))
                    arr["jet_idx"].append(int(r_idx))
                    arr["matched_dh_pid"].append(int(dh_info["dh_pid"]))
                    arr["matched_dh_pt"].append(float(dh_info["dh_pt"]))
                    arr["matched_genfatjet_idx"].append(int(closest_gen))
                    arr["reco_genfatjet_dr"].append(float(closest_dr))
                    result["n_selected_jets"] += 1

                    # Keep nearest example per fraction center for this file.
                    for center in example_centers:
                        score = abs(frac - center)
                        old = best_examples[float(center)]
                        if old is None or score < old["score"]:
                            payload = make_display_payload(
                                root_file=root_file,
                                file_idx=file_idx,
                                event_idx=iev,
                                jet_idx=r_idx,
                                center=center,
                                target_mpi=target_mpi,
                                frac=frac,
                                frac_reco=frac_reco,
                                frac_genfatjet=frac_gf,
                                category=cat,
                                sdmass=sdm,
                                sdpt=sdp,
                                n_dark=n_dark,
                                n_visible_dark=n_visible_dark,
                                n_inside=n_inside,
                                n_total=n_total,
                                closest_gen=closest_gen,
                                closest_dr=closest_dr,
                                dh_info=dh_info,
                                pids_evt=pids_evt,
                                status_evt=status_evt,
                                pt_evt=pt_evt,
                                eta_evt=eta_evt,
                                phi_evt=phi_evt,
                                d1_evt=d1_evt,
                                d2_evt=d2_evt,
                                gf_eta_evt=gf_eta_evt,
                                gf_phi_evt=gf_phi_evt,
                                rfs_pt=rfs_pt,
                                rfs_eta=rfs_eta,
                                rfs_phi=rfs_phi,
                                rfs_m=rfs_m,
                                pfs_pt=pfs_pt,
                                pfs_eta=pfs_eta,
                                pfs_phi=pfs_phi,
                                fatjet_r=fatjet_r,
                                max_pf_display=max_pf_display,
                                max_decay_nodes=max_decay_nodes,
                            )
                            best_examples[float(center)] = {"score": float(score), "payload": payload}

            result["examples"] = {k: v for k, v in best_examples.items() if v is not None}
            result["ok"] = True
            return result
        finally:
            root_handle.close()
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return result


# =============================================================================
# Merge and output helpers
# =============================================================================
def merge_results(results: Sequence[dict]) -> dict:
    merged = empty_result("merged", -1)
    merged["ok"] = True
    for r in results:
        for key in ["n_events", "n_total_recojets", "n_ptcut_recojets", "n_matched_recojets", "n_selected_jets"]:
            merged[key] += int(r.get(key, 0))
        for k, vals in r["arrays"].items():
            merged["arrays"][k].extend(vals)

    for k, vals in merged["arrays"].items():
        if k in ["ndark", "n_visible_dark", "n_visible_desc", "n_inside_visible_desc", "category_code", "file_idx", "event_idx", "jet_idx", "matched_dh_pid", "matched_genfatjet_idx"]:
            merged["arrays"][k] = np.asarray(vals, dtype=np.int64)
        else:
            merged["arrays"][k] = np.asarray(vals, dtype=np.float64)
    return merged


def choose_global_examples(results: Sequence[dict], centers: Sequence[float]) -> Dict[float, dict]:
    chosen: Dict[float, dict] = {}
    for center in centers:
        best = None
        for r in results:
            ex = r.get("examples", {}).get(float(center))
            if ex is None:
                continue
            if best is None or ex["score"] < best["score"]:
                best = ex
        if best is not None:
            chosen[float(center)] = best["payload"]
    return chosen


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_jets_summary(data: dict, outdir: Path, max_rows: int = 200000) -> None:
    arr = data["arrays"]
    n = len(arr["true_mass"])
    rows = []
    n_write = min(n, max_rows)
    for i in range(n_write):
        rows.append({
            "i": i,
            "true_mass": arr["true_mass"][i],
            "matched_fraction": arr["frac"][i],
            "matched_fraction_reco": arr["frac_reco"][i],
            "matched_fraction_genfatjet": arr["frac_genfatjet"][i],
            "sdmass": arr["sdmass"][i],
            "sdpt": arr["sdpt"][i],
            "ndark": int(arr["ndark"][i]),
            "n_visible_dark": int(arr["n_visible_dark"][i]),
            "n_visible_desc": int(arr["n_visible_desc"][i]),
            "n_inside_visible_desc": int(arr["n_inside_visible_desc"][i]),
            "reco_pt": arr["reco_pt"][i],
            "reco_mass": arr["reco_mass"][i],
            "file_idx": int(arr["file_idx"][i]),
            "event_idx": int(arr["event_idx"][i]),
            "jet_idx": int(arr["jet_idx"][i]),
            "matched_dh_pid": int(arr["matched_dh_pid"][i]),
            "matched_dh_pt": arr["matched_dh_pt"][i],
            "matched_genfatjet_idx": int(arr["matched_genfatjet_idx"][i]),
            "reco_genfatjet_dr": arr["reco_genfatjet_dr"][i],
        })
    write_csv(outdir / "selected_jets_summary.csv", rows)


# =============================================================================
# Plotting
# =============================================================================
def plot_fraction_hist(frac, outdir: Path, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    if len(frac):
        ax.hist(frac, bins=np.linspace(0.0, 1.0, 51), histtype="step", linewidth=2.0)
    style_ax(ax, xlabel="Matched visible-descendant fraction", ylabel="Jets", title="Matched fraction distribution")
    fig.tight_layout()
    fig.savefig(outdir / "matched_fraction_distribution.pdf")
    pdf.savefig(fig)
    plt.close(fig)

def plot_mdark_vs_fraction(true_m, frac, outdir: Path, pdf: PdfPages, mass_range, bins) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    if len(true_m) == 0:
        ax.text(0.5, 0.5, "No selected jets", ha="center", va="center", transform=ax.transAxes)
    else:
        h = ax.hist2d(true_m, frac, bins=[bins, 50], range=[mass_range, (0.0, 1.0)], norm=LogNorm())
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")
    style_ax(
        ax,
        xlabel=r"True dark-hadron mass $m_{dark}=m_\pi=m_\rho$ [GeV]",
        ylabel="Matched visible-descendant fraction",
        title="Matched fraction vs true dark mass",
    )
    fig.tight_layout()
    fig.savefig(outdir / "mdark_vs_matched_fraction.pdf")
    pdf.savefig(fig)
    plt.close(fig)

def plot_ndark_vs_fraction(ndark, frac, outdir: Path, pdf: PdfPages, ndark_max: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    if len(ndark) == 0:
        ax.text(0.5, 0.5, "No selected jets", ha="center", va="center", transform=ax.transAxes)
    else:
        xb = np.arange(-0.5, ndark_max + 1.5, 1.0)
        h = ax.hist2d(ndark, frac, bins=[xb, np.linspace(0.0, 1.0, 51)], norm=LogNorm())
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_ax(
        ax,
        xlabel=r"$N_{dark}$ inside reco FatJet",
        ylabel="Matched visible-descendant fraction",
        title=r"Matched fraction vs $N_{dark}$",
    )
    fig.tight_layout()
    fig.savefig(outdir / "ndark_vs_matched_fraction.pdf")
    pdf.savefig(fig)
    plt.close(fig)

def draw_sdmass_heatmap(fig, ax, tm, sd, title, mass_range, sdmass_range, bins):
    if len(tm) == 0:
        ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
        tau = float("nan")
        met = {"rel_median": np.nan, "rel_width68": np.nan}
    else:
        h = ax.hist2d(tm, sd, bins=bins, range=[mass_range, sdmass_range], norm=LogNorm())
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")
        ax.plot(mass_range, mass_range, linestyle="--", linewidth=1.2, label="y=x")
        ax.legend(frameon=True, loc="upper left")
        tau = compute_tau(tm, sd)
        met = rel_metrics(tm, sd)
        txt = (
            f"N = {len(tm):,}\n"
            f"Kendall tau = {tau:.3f}\n"
            f"median rel = {met['rel_median']:.3f}\n"
            f"68% half-width = {met['rel_width68']:.3f}"
        )
        ax.text(
            0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.5"), fontsize=10,
        )
    style_ax(
        ax,
        xlabel=r"True $m_{dark}=m_\pi=m_\rho$ [GeV]",
        ylabel="SoftDrop mass [GeV]",
        title=title,
    )
    return tau, met


def plot_sdmass_pages(true_m, sdmass, frac, category_code, outdir: Path, summary_pdf: PdfPages, mass_range, sdmass_range, bins) -> List[dict]:
    rows = []
    pages_path = outdir / "sdmass_vs_mdark_by_partial_fraction.pdf"
    with PdfPages(pages_path) as pages_pdf:
        # unmatched first
        page_specs = [("unmatched", None, None, "SoftDrop mass vs true mass, unmatched")]
        for center, lo, hi in fraction_slices():
            page_specs.append((center, lo, hi, f"SoftDrop mass vs true mass, matched fraction {lo:.2f}-{hi:.2f}"))
        page_specs.append(("full", None, None, "SoftDrop mass vs true mass, fully matched"))

        for label, lo, hi, title in page_specs:
            if label == "unmatched":
                mm = (category_code == CATEGORY_CODE['none']) & np.isfinite(true_m) & np.isfinite(sdmass) & (true_m > 0) & (sdmass > 0)
            elif label == "full":
                mm = (category_code == CATEGORY_CODE['full']) & np.isfinite(true_m) & np.isfinite(sdmass) & (true_m > 0) & (sdmass > 0)
            else:
                mm = (category_code == CATEGORY_CODE['partial']) & (frac >= lo) & (frac < hi) & np.isfinite(true_m) & np.isfinite(sdmass) & (true_m > 0) & (sdmass > 0)
            tm = true_m[mm]
            sd = sdmass[mm]
            fig, ax = plt.subplots(figsize=(8.5, 7.0))
            tau, met = draw_sdmass_heatmap(fig, ax, tm, sd, title, mass_range, sdmass_range, bins)
            fig.tight_layout()
            pages_pdf.savefig(fig)
            summary_pdf.savefig(fig)
            suffix = safe_name(label) if isinstance(label, float) else str(label)
            if label == 'unmatched':
                fig.savefig(outdir / 'sdmass_vs_mdark_unmatched.pdf')
            elif label == 'full':
                fig.savefig(outdir / 'sdmass_vs_mdark_fully_matched.pdf')
            else:
                fig.savefig(outdir / f"sdmass_vs_mdark_frac_{suffix}.pdf")
            plt.close(fig)
            rows.append({
                "slice": str(label),
                "fraction_lo": lo if lo is not None else "",
                "fraction_hi": hi if hi is not None else "",
                "entries": int(len(tm)),
                "kendall_tau_sdmass_vs_mdark": tau,
                "rel_median": met.get("rel_median", np.nan),
                "rel_mean": met.get("rel_mean", np.nan),
                "rel_width68": met.get("rel_width68", np.nan),
            })

    # Partial-only summary metrics curve
    part_rows = [r for r in rows if r['slice'] not in ('unmatched', 'full')]
    centers = np.array([float(r['slice']) for r in part_rows])
    entries = np.array([r['entries'] for r in part_rows], dtype=float)
    taus = np.array([r['kendall_tau_sdmass_vs_mdark'] for r in part_rows], dtype=float)
    medrel = np.array([r['rel_median'] for r in part_rows], dtype=float)
    width68 = np.array([r['rel_width68'] for r in part_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(centers, entries, marker="o", linewidth=2.0)
    ax.set_yscale("log")
    style_ax(ax, xlabel="Matched fraction bin center", ylabel="Jets", title="Entries per partial-fraction slice")
    fig.tight_layout()
    fig.savefig(outdir / "entries_vs_partial_fraction.pdf")
    summary_pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(centers, taus, marker="o", linewidth=2.0, label=r"Kendall $\tau(m_{SD}, m_{dark})$")
    ax.plot(centers, medrel, marker="s", linewidth=2.0, label="median rel residual")
    ax.plot(centers, width68, marker="^", linewidth=2.0, label="68% rel half-width")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.legend(frameon=True)
    style_ax(ax, xlabel="Matched fraction bin center", ylabel="Metric", title="SoftDrop performance vs partial matched fraction")
    fig.tight_layout()
    fig.savefig(outdir / "sdmass_metrics_vs_partial_fraction.pdf")
    summary_pdf.savefig(fig)
    plt.close(fig)

    return rows


def draw_jet_2d_panel(ax, payload: dict, title: str = "", show_legend: bool = False) -> None:
    meta = payload["meta"]
    pf = payload["pf"]
    nodes = payload["nodes"]
    visible_dh = payload.get("visible_dh", [])

    if len(pf.get("pt", [])) > 0:
        pt = np.asarray(pf["pt"])
        size = 6.0 + 55.0 * np.sqrt(pt / max(float(np.nanmax(pt)), 1e-6))
        ax.scatter(pf["deta"], pf["dphi"], s=size, c="0.78", alpha=0.5, edgecolors="none", label="PF candidates")

    vis_inside_x, vis_inside_y, vis_out_x, vis_out_y = [], [], [], []
    for node in nodes:
        if node["is_visible_final"] and node["inside_reco"]:
            vis_inside_x.append(node["deta"])
            vis_inside_y.append(node["dphi"])
        elif node["is_visible_final"]:
            vis_out_x.append(node["deta"])
            vis_out_y.append(node["dphi"])

    if vis_inside_x:
        ax.scatter(vis_inside_x, vis_inside_y, s=58, marker="o", c="tab:green", edgecolors="black", linewidths=0.35, label="visible DH descendants inside reco jet")
    if vis_out_x:
        ax.scatter(vis_out_x, vis_out_y, s=65, marker="x", c="tab:red", linewidths=1.8, label="visible DH descendants outside reco jet")

    if visible_dh:
        vd_x = [row["deta"] for row in visible_dh]
        vd_y = [row["dphi"] for row in visible_dh]
        ax.scatter(vd_x, vd_y, s=220, marker="*", c="gold", edgecolors="black", linewidths=0.8, label="visible dark hadrons")

    ax.add_patch(Circle((0.0, 0.0), 0.8, fill=False, linestyle="--", linewidth=1.5, color="black", label="AK8 reco cone"))
    ax.set_aspect("equal", adjustable="box")
    style_ax(ax, xlabel=r"$\Delta\eta$ from reco FatJet", ylabel=r"$\Delta\phi$ from reco FatJet", title=title)
    if show_legend:
        ax.legend(frameon=True, fontsize=8, loc="best")


def draw_example_page(payload: dict, pdf: PdfPages, outdir: Path) -> dict:
    meta = payload["meta"]
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    title = "2D jet/decay containment display"
    draw_jet_2d_panel(ax, payload, title=title, show_legend=True)
    txt = (
        f"matched fraction={meta['matched_fraction']:.3f}, category={meta['category']}\n"
        f"mDark={meta['target_mpi']:.2f} GeV, mSD={meta['sdmass']:.2f} GeV\n"
        f"Ndark={meta['n_dark']}, visible dark hadrons={meta['n_visible_dark']}\n"
        f"visible descendants inside/total={meta['n_inside_visible_desc']}/{meta['n_total_visible_desc']}\n"
        f"file_idx={meta['file_idx']}, event={meta['event_idx']}, jet={meta['jet_idx']}"
    )
    fig.suptitle(txt, fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    pdf.savefig(fig)
    label = meta['category'] if meta['category'] in ('none','full') else safe_name(meta['fraction_center'])
    out_pdf = outdir / f"example_root_2d_frac_{label}_event_{meta['event_idx']}_jet_{meta['jet_idx']}.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    return dict(meta)


def make_example_grid_pdf(examples: Dict[float, dict], outdir: Path, summary_pdf: PdfPages) -> None:
    ordered = sorted(examples.items(), key=lambda kv: kv[0])
    if not ordered:
        return
    n = len(ordered)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (center, payload) in zip(axes.ravel(), ordered):
        ax.axis('on')
        meta = payload['meta']
        if center <= 0.001:
            ttl = '0% matched'
        elif center >= 0.999:
            ttl = '100% matched'
        else:
            ttl = f'{int(round(center*100))}% matched'
        draw_jet_2d_panel(ax, payload, title=ttl, show_legend=False)
        ax.text(0.02, 0.98, f"mDark={meta['target_mpi']:.0f}, mSD={meta['sdmass']:.0f}\nfrac={meta['matched_fraction']:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, fontsize=9)
    fig.suptitle('Representative 2D jet displays from 0% to 100% matched', fontsize=14)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.96])
    outpath = outdir / 'example_partial_fraction_subplots_2d.pdf'
    fig.savefig(outpath)
    summary_pdf.savefig(fig)
    plt.close(fig)


def make_example_pdf(examples: Dict[float, dict], outdir: Path, summary_pdf: PdfPages) -> List[dict]:
    rows = []
    example_path = outdir / "example_partial_fraction_jets_root_2d.pdf"
    with PdfPages(example_path) as ex_pdf:
        for center in sorted(examples):
            rows.append(draw_example_page(examples[center], ex_pdf, outdir))

    make_example_grid_pdf(examples, outdir, summary_pdf)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.axis("off")
    ax.text(
        0.02, 0.95,
        "Representative ROOT-level example jet displays written to:\n"
        f"{example_path}\n\n"
        "These are 2D displays only. PF candidates shown here are only the PF candidates\n"
        "inside the reco AK8 jet cone. Gold stars mark the visible dark hadrons in the jet.\n",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
    )
    fig.tight_layout()
    summary_pdf.savefig(fig)
    fig.savefig(outdir / "example_display_note.pdf")
    plt.close(fig)
    return rows


def write_run_summary(outdir: Path, args, files: Sequence[str], good_results, bad_results, merged, example_rows) -> None:
    arr = merged["arrays"]
    with open(outdir / "run_summary.txt", "w") as f:
        f.write("ROOT-level matched jet study\n")
        f.write("====================================\n")
        f.write(f"n_input_files: {len(files)}\n")
        f.write(f"n_good_files: {len(good_results)}\n")
        f.write(f"n_failed_files: {len(bad_results)}\n")
        f.write(f"n_events_read: {merged['n_events']}\n")
        f.write(f"n_total_recojets: {merged['n_total_recojets']}\n")
        f.write(f"n_ptcut_recojets: {merged['n_ptcut_recojets']}\n")
        f.write(f"n_matched_recojets: {merged['n_matched_recojets']}\n")
        f.write(f"n_selected_jets: {len(arr['true_mass'])}\n")
        f.write(f"n_example_pages: {len(example_rows)}\n")
        f.write(f"pt_cut: {args.pt_cut}\n")
        f.write(f"fatjet_r: {args.fatjet_r}\n")
        f.write(f"matching_r: {args.matching_r}\n")
        f.write(f"keep_mode: {args.keep_mode}\n")
        f.write(f"fraction_reference: {args.fraction_reference}\n")
        f.write(f"require_status_8384: {args.require_status_8384}\n")
        f.write(f"mass_range: {args.mass_min}, {args.mass_max}\n")
        f.write(f"sdmass_range: {args.sdmass_min}, {args.sdmass_max}\n")
        f.write("\nOutputs:\n")
        for p in sorted(outdir.glob("*.pdf")):
            f.write(f"  {p.name}\n")
        for p in sorted(outdir.glob("*.csv")):
            f.write(f"  {p.name}\n")


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="ROOT-level partial matched jet study with real decay displays.")
    parser.add_argument("--filelist", default=None, help="Text file with ROOT paths, one per line.")
    parser.add_argument("--input-glob", nargs="*", default=None, help="Alternative: local ROOT glob(s).")
    parser.add_argument("--eos-host", default="", help="Prefix for /store paths, e.g. root://cmseos.fnal.gov")
    parser.add_argument("--outdir", default="partial_match_root_plots", help="Output directory.")

    parser.add_argument("--pt-cut", type=float, default=20.0, help="Reco FatJet pT cut.")
    parser.add_argument("--fatjet-r", type=float, default=0.8, help="FatJet cone radius used for containment.")
    parser.add_argument("--matching-r", type=float, default=0.4, help="Reco FatJet to GenFatJet matching radius.")
    parser.add_argument(
        "--fraction-reference",
        choices=["reco", "genfatjet"],
        default="reco",
        help="Axis used for matched fraction. reco = fraction inside reco FatJet cone. genfatjet = older builder-style fraction.",
    )
    parser.add_argument(
        "--keep-mode",
        default="all",
        choices=["full_only", "partial_only", "none_only", "full_plus_partial", "partial_plus_none", "not_full", "all"],
        help="Containment category selection after computing the chosen matched fraction.",
    )
    parser.add_argument(
        "--no-require-status-8384",
        dest="require_status_8384",
        action="store_false",
        help="Count all dark hadrons for Ndark. Default counts status 83/84 only.",
    )
    parser.set_defaults(require_status_8384=True)

    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-events-per-file", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--mass-min", type=float, default=1.0)
    parser.add_argument("--mass-max", type=float, default=200.0)
    parser.add_argument("--sdmass-min", type=float, default=0.0)
    parser.add_argument("--sdmass-max", type=float, default=250.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--ndark-max", type=int, default=20)
    parser.add_argument("--max-pf-display", type=int, default=250, help="Max PF candidates stored/displayed per example jet.")
    parser.add_argument("--max-decay-nodes", type=int, default=180, help="Max GenParticle descendant nodes displayed per example jet.")
    parser.add_argument("--selected-csv-max-rows", type=int, default=200000)
    args = parser.parse_args()

    if not args.filelist and not args.input_glob:
        parser.error("Provide --filelist or --input-glob")
    args.mass_range = (args.mass_min, args.mass_max)
    args.sdmass_range = (args.sdmass_min, args.sdmass_max)
    return args


def main() -> None:
    args = parse_args()
    setup_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(args)
    if not files:
        raise RuntimeError("No input ROOT files found.")

    print("==========================================================", flush=True)
    print("ROOT-level partial matched DRACON/SVJ jet study", flush=True)
    print(f"Input files        : {len(files)}", flush=True)
    print(f"Outdir             : {outdir}", flush=True)
    print(f"pT cut             : {args.pt_cut} GeV", flush=True)
    print(f"FatJet R           : {args.fatjet_r}", flush=True)
    print(f"Reco-Gen match R   : {args.matching_r}", flush=True)
    print(f"Fraction reference : {args.fraction_reference}", flush=True)
    print(f"Keep mode          : {args.keep_mode}", flush=True)
    print(f"Workers            : {args.workers}", flush=True)
    print("==========================================================", flush=True)

    with open(outdir / "input_files_used.txt", "w") as f:
        for i, path in enumerate(files):
            f.write(f"{i}\t{path}\n")

    centers = [0.0] + [c for c, _lo, _hi in fraction_slices()] + [1.0]
    jobs = [
        (
            root_file,
            file_idx,
            args.keep_mode,
            args.max_events_per_file,
            args.fatjet_r,
            args.matching_r,
            args.pt_cut,
            args.require_status_8384,
            args.fraction_reference,
            args.mass_min,
            args.mass_max,
            args.sdmass_min,
            args.sdmass_max,
            centers,
            args.max_pf_display,
            args.max_decay_nodes,
        )
        for file_idx, root_file in enumerate(files)
    ]

    results = []
    if args.workers <= 1:
        for i, job in enumerate(jobs):
            if i % 25 == 0:
                print(f"[INFO] Processing file {i+1}/{len(jobs)}", flush=True)
            results.append(process_one_file(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one_file, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                r = fut.result()
                results.append(r)
                if i % 25 == 0 or i == len(futs):
                    n_ok = sum(1 for rr in results if rr.get("ok"))
                    n_bad = len(results) - n_ok
                    print(f"[INFO] Finished {i}/{len(futs)} files | ok={n_ok}, failed={n_bad}", flush=True)

    good = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    if bad:
        write_csv(outdir / "failed_files.csv", [
            {"file_idx": r.get("file_idx"), "root_file": r.get("root_file"), "error": r.get("error"), "traceback": r.get("traceback")}
            for r in bad
        ])
    if not good:
        raise RuntimeError("All ROOT files failed; see failed_files.csv")

    merged = merge_results(good)
    arr = merged["arrays"]
    nsel = len(arr["true_mass"])
    print("==========================================================", flush=True)
    print(f"Good files          : {len(good)}", flush=True)
    print(f"Failed files        : {len(bad)}", flush=True)
    print(f"Reco jets total     : {merged['n_total_recojets']}", flush=True)
    print(f"Reco jets pT cut    : {merged['n_ptcut_recojets']}", flush=True)
    print(f"Reco jets genmatch  : {merged['n_matched_recojets']}", flush=True)
    print(f"Selected jets         : {nsel}", flush=True)
    print("==========================================================", flush=True)

    if nsel == 0:
        raise RuntimeError("No selected jets after cuts. Try --keep-mode all or loosen the mass/SoftDrop ranges.")

    examples = choose_global_examples(good, centers)

    combined_pdf = outdir / "partial_match_root_summary.pdf"
    with PdfPages(combined_pdf) as pdf:
        plot_fraction_hist(arr["frac"], outdir, pdf)
        plot_mdark_vs_fraction(arr["true_mass"], arr["frac"], outdir, pdf, args.mass_range, args.bins)
        plot_ndark_vs_fraction(arr["ndark"], arr["frac"], outdir, pdf, args.ndark_max)
        frac_rows = plot_sdmass_pages(arr["true_mass"], arr["sdmass"], arr["frac"], arr["category_code"], outdir, pdf, args.mass_range, args.sdmass_range, args.bins)
        example_rows = make_example_pdf(examples, outdir, pdf)

    write_csv(outdir / "partial_fraction_summary.csv", frac_rows)
    write_csv(outdir / "representative_jets.csv", example_rows)
    write_selected_jets_summary(merged, outdir, max_rows=args.selected_csv_max_rows)
    write_run_summary(outdir, args, files, good, bad, merged, example_rows)

    print("[DONE] Wrote:", flush=True)
    print(f"  {combined_pdf}", flush=True)
    print(f"  {outdir / 'sdmass_vs_mdark_by_partial_fraction.pdf'}", flush=True)
    print(f"  {outdir / 'example_partial_fraction_jets_root_2d.pdf'}", flush=True)
    print(f"  {outdir / 'example_partial_fraction_subplots_2d.pdf'}", flush=True)
    print(f"  {outdir / 'partial_fraction_summary.csv'}", flush=True)
    print(f"  {outdir / 'representative_jets.csv'}", flush=True)
    print(f"  {outdir / 'selected_jets_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
