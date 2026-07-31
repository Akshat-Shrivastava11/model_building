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
     - companion multipage PDF with inverted ratio cuts: pT(DH2)/pT(DH1) <= cut

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

    # LPC/XRootD wants root://cmseos.fnal.gov//store/...
    # not root://cmseos.fnal.gov/store/...
    if path.startswith("root://"):
        m = re.match(r"^(root://[^/]+)/(store/.*)$", path)
        if m:
            return m.group(1).rstrip("/") + "//" + m.group(2)
        return path

    if eos_host and path.startswith("/store/"):
        return eos_host.rstrip("/") + "//" + path.lstrip("/")

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



def visible_dark_hadrons_in_reco_jet(
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
) -> List[dict]:
    """
    Return visible dark hadrons inside the reco FatJet, sorted by gen-level pT.

    Each entry is:
      {
        "idx": particle index,
        "pt": gen pT,
        "eta": gen eta,
        "phi": gen phi,
      }

    This is the same selection as visible_dark_hadron_pts_in_reco_jet(),
    but keeps eta/phi so we can compute DR(DH1,DH2).
    """
    finite = np.isfinite(eta_evt) & np.isfinite(phi_evt) & np.isfinite(pt_evt)
    pid_mask = np.isin(pids_evt, list(DARK_HADRONS))

    if require_status_8384:
        status_mask = np.isin(status_evt, list(DARK_HADRON_COUNT_STATUSES))
    else:
        status_mask = np.ones_like(pid_mask, dtype=bool)

    dh_indices = np.where(finite & pid_mask & status_mask)[0]
    out = []

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
            out.append({
                "idx": int(dh_idx),
                "pt": float(pt_evt[dh_idx]),
                "eta": float(eta_evt[dh_idx]),
                "phi": float(phi_evt[dh_idx]),
            })

    out.sort(key=lambda d: d["pt"], reverse=True)
    return out


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
        "dh1_eta": [],
        "dh1_phi": [],
        "dh2_eta": [],
        "dh2_phi": [],
        "dh12_dr": [],
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

                dhs = visible_dark_hadrons_in_reco_jet(
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

                nvdh = len(dhs)

                dh1_pt  = dhs[0]["pt"]  if nvdh >= 1 else np.nan
                dh1_eta = dhs[0]["eta"] if nvdh >= 1 else np.nan
                dh1_phi = dhs[0]["phi"] if nvdh >= 1 else np.nan

                dh2_pt  = dhs[1]["pt"]  if nvdh >= 2 else np.nan
                dh2_eta = dhs[1]["eta"] if nvdh >= 2 else np.nan
                dh2_phi = dhs[1]["phi"] if nvdh >= 2 else np.nan

                ratio = dh2_pt / dh1_pt if nvdh >= 2 and dh1_pt > 0 else np.nan
                dh12_dr = (
                    float(calc_dr(dh1_eta, dh1_phi, dh2_eta, dh2_phi))
                    if nvdh >= 2
                    and np.isfinite(dh1_eta)
                    and np.isfinite(dh1_phi)
                    and np.isfinite(dh2_eta)
                    and np.isfinite(dh2_phi)
                    else np.nan
                )

                cols["true_mass"].append(float(target_mpi))
                cols["sdmass"].append(safe_softdrop_mass(rf_sdmass, iev, r_idx))
                cols["sdpt"].append(safe_softdrop_pt(rf_sdpt, iev, r_idx))
                cols["reco_pt"].append(j_pt)
                cols["reco_mass"].append(j_m)
                cols["n_visible_dh"].append(int(nvdh))
                cols["dh1_pt"].append(float(dh1_pt) if np.isfinite(dh1_pt) else np.nan)
                cols["dh2_pt"].append(float(dh2_pt) if np.isfinite(dh2_pt) else np.nan)
                cols["dh1_eta"].append(float(dh1_eta) if np.isfinite(dh1_eta) else np.nan)
                cols["dh1_phi"].append(float(dh1_phi) if np.isfinite(dh1_phi) else np.nan)
                cols["dh2_eta"].append(float(dh2_eta) if np.isfinite(dh2_eta) else np.nan)
                cols["dh2_phi"].append(float(dh2_phi) if np.isfinite(dh2_phi) else np.nan)
                cols["dh12_dr"].append(float(dh12_dr) if np.isfinite(dh12_dr) else np.nan)
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
                "central_rel_residual": float(np.nanpercentile(rel, 50)) if len(rel) else float("nan"),
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



def make_ratio_vs_mass(data, outdir: Path, args) -> None:
    true_m = np.asarray(data["true_mass"]).reshape(-1)
    ratio = np.asarray(data["dh2_over_dh1"]).reshape(-1)
    nvdh = np.asarray(data["n_visible_dh"]).reshape(-1)

    mask = (
        np.isfinite(true_m)
        & np.isfinite(ratio)
        & (nvdh >= 2)
        & (true_m >= args.mass_range[0])
        & (true_m <= args.mass_range[1])
        & (ratio >= 0.0)
        & (ratio <= 1.0)
    )

    x = true_m[mask]
    y = ratio[mask]
    tau = compute_tau(x, y)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    if len(x) == 0:
        ax.text(
            0.5, 0.5,
            "No valid N_visible_DH >= 2 entries",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        h = ax.hist2d(
            x,
            y,
            bins=[args.bins, args.ratio_bins],
            range=[args.mass_range, (0.0, 1.0)],
            norm=LogNorm(),
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")

    ax.text(
        0.03, 0.97,
        f"N = {len(x):,}\nKendall tau = {tau:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    style_ax(
        ax,
        xlabel=r"True dark-hadron mass $m_{\mathrm{DH}}$ [GeV]",
        ylabel=r"$p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$",
        title=r"Visible-DH pT balance vs true dark-hadron mass, $N_{\mathrm{DH}}^{vis}\geq2$",
    )

    ax.set_xlim(*args.mass_range)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    outpath = outdir / "dh2_over_dh1_vs_true_mass.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def make_ratio_cut_pages(data, outdir: Path, args, ratio_rows: List[dict]) -> None:
    pdf_path = outdir / "sdmass_vs_true_mass_by_DH2overDH1_cut_pages.pdf"
    with PdfPages(pdf_path) as pdf:
        for cut in args.ratio_cuts:
            m = (
                valid_base_mask(data, args.mass_range, args.sdmass_range)
                & (data["n_visible_dh"] >= 2)
                & np.isfinite(data["dh2_over_dh1"])
                & (data["dh2_over_dh1"] >= cut)
                & (data["dh2_over_dh1"] <= 1.0)
            )
            true_m = data["true_mass"][m]
            sd = data["sdmass"][m]
            rel = (sd - true_m) / true_m if len(true_m) else np.array([])
            extra = ""
            if len(rel):
                extra = f"central rel = {np.nanpercentile(rel, 50):.3f}\n68% half-width = {robust_rel_width(rel):.3f}"
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


def make_ratio_inverted_cut_pages(data, outdir: Path, args) -> List[dict]:
    """
    Make the inverted-cut companion PDF.

    Existing ratio-cut pages use:
        pT(DH2)/pT(DH1) >= cut

    This companion uses:
        pT(DH2)/pT(DH1) <= cut

    With the default --ratio-cuts, this gives pages for <= 0.00, 0.10, ..., 0.90.
    The <= 0.00 page will usually be empty unless exact-zero ratios exist.
    """
    rows = []
    pdf_path = outdir / "sdmass_vs_true_mass_by_DH2overDH1_inverted_cut_pages.pdf"

    with PdfPages(pdf_path) as pdf:
        for cut in args.ratio_cuts:
            m = (
                valid_base_mask(data, args.mass_range, args.sdmass_range)
                & (data["n_visible_dh"] >= 2)
                & np.isfinite(data["dh2_over_dh1"])
                & (data["dh2_over_dh1"] >= 0.0)
                & (data["dh2_over_dh1"] <= cut)
            )

            true_m = data["true_mass"][m]
            sd = data["sdmass"][m]
            rel = (sd - true_m) / true_m if len(true_m) else np.array([])

            tau = compute_tau(true_m, sd)
            central = float(np.nanpercentile(rel, 50)) if len(rel) else float("nan")
            width68 = robust_rel_width(rel)

            rows.append({
                "dh_pt_ratio_max_cut": float(cut),
                "entries": int(len(true_m)),
                "kendall_tau_sdmass_true_mass": float(tau),
                "central_rel_residual": central,
                "iqr68_rel_residual": width68,
            })

            extra = ""
            if len(rel):
                extra = f"central rel = {central:.3f}\n68% half-width = {width68:.3f}"

            draw_heatmap_page(
                pdf=pdf,
                true_m=true_m,
                sd_m=sd,
                title=fr"$m_{{SD}}$ vs $m_{{DH}}$, $N_{{DH}}^{{vis}}\geq2$, $p_T(DH_2)/p_T(DH_1)\leq{cut:.2f}$",
                mass_range=args.mass_range,
                sdmass_range=args.sdmass_range,
                bins=args.bins,
                extra_text=extra,
            )

    print(f"[SAVED] {pdf_path}")

    csv_path = outdir / "DH2overDH1_inverted_ratio_cut_softdrop_summary.csv"
    write_csv(csv_path, rows)
    print(f"[SAVED] {csv_path}")

    return rows




def make_dh12_dr_heatmaps(data, outdir: Path, args) -> None:
    """
    Make two requested heatmaps:

      1. DR(DH1,DH2) vs mdark
      2. DR(DH1,DH2) vs pT(DH2)/pT(DH1)

    Requires N_visible_DH >= 2.
    """
    true_m = np.asarray(data["true_mass"]).reshape(-1)
    ratio = np.asarray(data["dh2_over_dh1"]).reshape(-1)
    dr12 = np.asarray(data["dh12_dr"]).reshape(-1)
    nvdh = np.asarray(data["n_visible_dh"]).reshape(-1)

    base = (
        np.isfinite(true_m)
        & np.isfinite(ratio)
        & np.isfinite(dr12)
        & (nvdh >= 2)
        & (true_m >= args.mass_range[0])
        & (true_m <= args.mass_range[1])
        & (ratio >= 0.0)
        & (ratio <= 1.0)
        & (dr12 >= 0.0)
        & (dr12 <= args.fatjet_r * 2.0)
    )

    # ------------------------------------------------------------------
    # Plot 1: DR(DH1,DH2) vs mdark
    # ------------------------------------------------------------------
    x = true_m[base]
    y = dr12[base]
    tau = compute_tau(x, y)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    if len(x) == 0:
        ax.text(
            0.5, 0.5,
            "No valid jets with N_visible_DH >= 2",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        h = ax.hist2d(
            x,
            y,
            bins=[args.bins, getattr(args, "dr_bins", 80)],
            range=[args.mass_range, (0.0, args.fatjet_r * 2.0)],
            norm=LogNorm(),
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")

    ax.text(
        0.03, 0.97,
        f"N = {len(x):,}\nKendall tau = {tau:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    style_ax(
        ax,
        xlabel=r"True dark-hadron mass $m_{\mathrm{dark}}$ [GeV]",
        ylabel=r"$\Delta R(\mathrm{DH}_1,\mathrm{DH}_2)$",
        title=r"$\Delta R(\mathrm{DH}_1,\mathrm{DH}_2)$ vs $m_{\mathrm{dark}}$",
    )

    ax.set_xlim(*args.mass_range)
    ax.set_ylim(0.0, args.fatjet_r * 2.0)
    fig.tight_layout()

    outpath = outdir / "dh12_dr_vs_mdark.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[SAVED] {outpath}")

    # ------------------------------------------------------------------
    # Plot 2: DR(DH1,DH2) vs pT(DH2)/pT(DH1)
    # ------------------------------------------------------------------
    x = ratio[base]
    y = dr12[base]
    tau = compute_tau(x, y)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    if len(x) == 0:
        ax.text(
            0.5, 0.5,
            "No valid jets with N_visible_DH >= 2",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        h = ax.hist2d(
            x,
            y,
            bins=[getattr(args, "ratio_bins", 80), getattr(args, "dr_bins", 80)],
            range=[(0.0, 1.0), (0.0, args.fatjet_r * 2.0)],
            norm=LogNorm(),
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Jets")

    ax.text(
        0.03, 0.97,
        f"N = {len(x):,}\nKendall tau = {tau:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    style_ax(
        ax,
        xlabel=r"$p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$",
        ylabel=r"$\Delta R(\mathrm{DH}_1,\mathrm{DH}_2)$",
        title=r"$\Delta R(\mathrm{DH}_1,\mathrm{DH}_2)$ vs visible-DH pT balance",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, args.fatjet_r * 2.0)
    fig.tight_layout()

    outpath = outdir / "dh12_dr_vs_dh_pt_ratio.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def make_tau_vs_dh_pt_ratio_cut(data, outdir: Path, args) -> None:
    """
    Plot Kendall tau(m_SD, m_dark) as a function of the minimum
    pT(DH2)/pT(DH1) cut.

    For each cut c:
      require N_visible_DH >= 2
      require pT(DH2)/pT(DH1) >= c
      compute Kendall tau(sdmass, true_mass)
    """
    true_m = np.asarray(data["true_mass"]).reshape(-1)
    sd = np.asarray(data["sdmass"]).reshape(-1)
    ratio = np.asarray(data["dh2_over_dh1"]).reshape(-1)
    nvdh = np.asarray(data["n_visible_dh"]).reshape(-1)

    rows = []
    for cut in args.ratio_cuts:
        mask = (
            np.isfinite(true_m)
            & np.isfinite(sd)
            & np.isfinite(ratio)
            & (true_m > 0)
            & (sd > 0)
            & (nvdh >= 2)
            & (true_m >= args.mass_range[0])
            & (true_m <= args.mass_range[1])
            & (sd >= args.sdmass_range[0])
            & (sd <= args.sdmass_range[1])
            & (ratio >= cut)
            & (ratio <= 1.0)
        )

        tau = compute_tau(sd[mask], true_m[mask])
        rows.append({
            "dh_pt_ratio_min_cut": float(cut),
            "entries": int(np.sum(mask)),
            "kendall_tau_sdmass_mdark": float(tau),
        })

    cuts = np.array([r["dh_pt_ratio_min_cut"] for r in rows], dtype=float)
    entries = np.array([r["entries"] for r in rows], dtype=float)
    taus = np.array([r["kendall_tau_sdmass_mdark"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    good = np.isfinite(taus)
    if np.any(good):
        ax.plot(cuts[good], taus[good], marker="o", linewidth=2.2)

        for x, y, n in zip(cuts[good], taus[good], entries[good]):
            ax.text(
                x,
                y,
                f"{int(n):,}",
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=45,
            )
    else:
        ax.text(
            0.5, 0.5,
            "No valid Kendall tau points",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    style_ax(
        ax,
        xlabel=r"Minimum $p_T(\mathrm{DH}_2)/p_T(\mathrm{DH}_1)$",
        ylabel=r"Kendall $\tau(m_{\mathrm{SD}}, m_{\mathrm{dark}})$",
        title=r"SoftDrop--truth mass correlation vs visible-DH pT-balance cut",
    )

    ax.set_xlim(min(args.ratio_cuts) - 0.02, max(args.ratio_cuts) + 0.02)
    ax.set_ylim(0.10, 0.5)
    fig.tight_layout()

    out_pdf = outdir / "kendalltau_sdmass_mdark_vs_dh_pt_ratio_cut.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVED] {out_pdf}")

    out_csv = outdir / "kendalltau_sdmass_mdark_vs_dh_pt_ratio_cut.csv"
    write_csv(out_csv, rows)
    print(f"[SAVED] {out_csv}")


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
    parser.add_argument("--ratio-bins", type=int, default=80)
    parser.add_argument("--dr-bins", type=int, default=80)
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

    # Remove stale outputs from the old version.
    stale_outputs = [
        "visibleDH_ratio_softdrop_summary.pdf",
        "dh2_over_dh1_ratio_distribution.pdf",
        "sdmass_relative_residual_ratio_cut_overlay.pdf",
        "ratio_cut_yield_curve.pdf",
        "ratio_cut_kendalltau_curve.pdf",
        "ratio_cut_residual_metrics_curve.pdf",
        "DH2overDH1_ratio_cut_softdrop_summary.csv",
    ]
    for name in stale_outputs:
        stale = outdir / name
        if stale.exists():
            stale.unlink()

    # Main requested outputs only.
    make_ndark_pages(data, outdir, args)
    make_dh12_dr_heatmaps(data, outdir, args)
    make_tau_vs_dh_pt_ratio_cut(data, outdir, args)
    make_ratio_vs_mass(data, outdir, args)
    make_ratio_cut_pages(data, outdir, args, [])
    make_ratio_inverted_cut_pages(data, outdir, args)

    print("==========================================================")
    print("Done. Key outputs:")
    print(f"  {outdir / 'sdmass_vs_true_mass_by_nvisibleDH_pages.pdf'}")
    print(f"  {outdir / 'dh12_dr_vs_mdark.pdf'}")
    print(f"  {outdir / 'dh12_dr_vs_dh_pt_ratio.pdf'}")
    print(f"  {outdir / 'dh2_over_dh1_vs_true_mass.pdf'}")
    print(f"  {outdir / 'sdmass_vs_true_mass_by_DH2overDH1_cut_pages.pdf'}")
    print(f"  {outdir / 'sdmass_vs_true_mass_by_DH2overDH1_inverted_cut_pages.pdf'}")
    print("==========================================================")


if __name__ == "__main__":
    main()
