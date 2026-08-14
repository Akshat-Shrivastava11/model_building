"""Shared configuration for the soft drop (beta, z_cut) parameter study.

Everything downstream (data_pull.py, softdrop_recluster.py, scan_softdrop.py,
plotting.py, run_strategy.py) imports its constants from here, so the target
masses / grid / paths only need to be edited in one place.
"""
from pathlib import Path
import sys

SDMASS_PARAMETER_DIR = Path(__file__).resolve().parent
MODEL_BUILDING_DIR = SDMASS_PARAMETER_DIR.parent

# Make StrategiesForTrainingData.py (and, transitively, the dhbuilder module
# it imports) importable as a library, exactly the way run_strategies_job.py
# already does.
for p in (str(MODEL_BUILDING_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# Strategies, in the order requested: strat1 -> strat1.5 -> strat3 -> strat2
# =============================================================================
STRATEGY_ORDER = [
    "strat1_leading_full",
    "strat1_5_any_full",
    "strat3_leading_frac60",
    "strat2_all_full",
]

# =============================================================================
# Target dark hadron masses (GeV) -- all land exactly on the 0.025 GeV EOS
# mass-scan grid (verified against mass_points_refined_1to250.txt).
# =============================================================================
TARGET_MASSES = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 150.0]

# Tolerance (GeV) used when selecting jets "at" a target mass for the 1D
# histogram overlays -- same pattern as StrategiesForTrainingData.py's
# _sdmass_vals_at_point (np.abs(masses[:,0] - m) <= tol), just wider here
# since our pooled samples span a small window around each nominal mass
# rather than being exactly on top of it.
MASS_SELECT_TOL_FRAC = 0.03   # +/- 3% of the target mass ...
MASS_SELECT_TOL_MIN = 0.2     # ... with an absolute floor (GeV) for low masses

# =============================================================================
# Adaptive EOS-window data pull (see data_pull.py)
# =============================================================================
# Windows are tried in order until a strategy reaches MIN_JETS_TARGET jets
# (or the widest window is reached, in which case we take what we get and
# say so loudly).
WINDOW_FRACTIONS = [0.02, 0.05, 0.10, 0.20, 0.50]
WINDOW_ABS_FLOOR_GEV = 0.15
MIN_JETS_TARGET = 5000
MAX_FILES_PER_MASS = 3000    # hard ceiling on files pulled for one target mass
MAX_EVENTS_PER_FILE = None   # None = process every event in every pulled file

# =============================================================================
# Soft drop parameter grid -- grounded in arXiv:1402.2657 (Soft Drop paper).
# The paper's own MC comparisons sweep beta in {2,1,0,-0.5} at zcut=0.1
# (Figs 3,4,6,8) and extend to beta=-3/2 in the W-tagging study (Figs 11,12);
# zcut=0.1 is the paper's (and this repo's Delphes card's) default operating
# point. R0=0.8 matches cards/delphes_card_CMS.tcl's FatJetFinder.
# =============================================================================
R0 = 0.8

BETA_DEFAULT = 0.0     # fixed beta used for the z_cut scan (mMDT / Delphes default)
ZCUT_DEFAULT = 0.1     # fixed z_cut used for the beta scan (paper + Delphes default)

BETA_GRID = [-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ZCUT_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# Representative grid points to draw full soft-drop-dropped-constituent event
# displays for (drawing all of BETA_GRID x ZCUT_GRID x events would be far
# too many images).
EVENT_DISPLAY_BETAS = [-1.0, 0.0, 1.0, 2.0]      # at ZCUT_DEFAULT
EVENT_DISPLAY_ZCUTS = [0.05, 0.1, 0.2]           # at BETA_DEFAULT
N_EVENT_DISPLAY_JETS_PER_MASS = 3

# =============================================================================
# Paths
# =============================================================================
DATA_CACHE_DIR = SDMASS_PARAMETER_DIR / "data_cache"
RAW_CACHE_DIR = DATA_CACHE_DIR / "raw"
SCAN_CACHE_DIR = DATA_CACHE_DIR / "sdmass_scan"
PLOTS_DIR = SDMASS_PARAMETER_DIR / "plots"


def mass_label(m):
    """Filesystem-safe label for a target mass, e.g. 1.0 -> '1p0', 150.0 -> '150p0'."""
    return f"{m:g}".replace(".", "p").replace("-", "m")


def raw_cache_path(strategy, m):
    return RAW_CACHE_DIR / strategy / f"{mass_label(m)}.npz"


def scan_cache_path(strategy, m):
    return SCAN_CACHE_DIR / strategy / f"{mass_label(m)}.npz"


def plots_dir_for(strategy):
    d = PLOTS_DIR / strategy
    d.mkdir(parents=True, exist_ok=True)
    return d
