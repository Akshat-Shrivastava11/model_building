# Jet radius optimization

Which jet radius best reconstructs the dark shower, at different stages of the
shower? The study looks at jets of **dark hadrons** at radius R_DH (the truth
jets) and jets of **visible particles** at radius R. For each R_DH it asks which
R best reconstructs the dark-hadron jets. It is built on the dark-hadron
ancestry in `darkancestry.py`.

## Definitions: two stages of the shower

The dark shower is probed at two stages, each clustered into jets with anti-kT
at its own radius (pT > 15 GeV for both, as in the Delphes card):

| Radius | Stage | Clustered from | Standard Delphes jets on this axis |
|---|---|---|---|
| **R_DH** | dark hadrons, after the dark shower has hadronized and before they decay | `DarkHadronCandidate` (the dark hadrons' four-momenta) | R_DH = 0.8: `DarkHadronJet` |
| **R** | visible particles, after the dark hadrons decay to SM particles and hadronize | `GenCandidate` (stable visible particles; neutrinos and stable dark hadrons removed) | R = 0.4: `GenJet`; R = 0.8: `GenFatJet` |

- **R_DH** is the radius of the jets **built from the dark hadrons**; it sets
  how the dark-hadron stage of the shower is grouped into objects (small R_DH:
  several jets per dark quark; large R_DH: about one jet per dark quark). These
  dark-hadron jets are the truth jets.
- **R** is the radius of the **generator-level visible jets**, i.e. a
  `GenJet`/`GenFatJet` with a variable radius. This is the stage an experiment
  reconstructs.
- The question at each R_DH: which visible-jet radius R best reconstructs the
  dark-hadron jets of radius R_DH? Scanning both gives the mapping between the
  two stages.

## What it does

For each (R_DH, R):

1. Cluster the dark hadrons at R_DH (truth jets) and the visible particles at R
   (visible jets), as defined above. At R_DH = R = 0.8 these are exactly the
   Delphes `DarkHadronJet` and `GenFatJet`.
2. Match truth jets one-to-one to visible jets by shared visible pT.
3. Split the matched jet's pT by where it came from, using `darkancestry`.
   A dark hadron is **complete** in the jet if the jet holds at least a
   fraction `--min-containment` (default 0.95) of its visible pT, and
   **partial** if the jet holds some but less:

| | pT fraction from |
|---|---|
| `f_full` | **complete** dark hadrons |
| `f_partial` | **partial** dark hadrons |
| `f_nodh` | particles with **no** dark-hadron ancestor (ISR, UE, ...) |

These add up to 1.

## Scores

All are fractions of pT. For a truth jet T matched to a visible jet J, with
pT(S) the scalar pT sum over a set of particles S, V_d the visible descendants
of dark hadron d, and V_T the union of V_d over the dark hadrons of T:

- d is **complete** in J if pT(V_d ∩ J) ≥ 0.95 · pT(V_d) (`--min-containment`,
  default 0.95; 1 requires every visible descendant), the same definition as
  for `f_full`

| Score | Definition | Meaning |
|---|---|---|
| **acceptance** | Σ_{d ∈ T, complete} pT(V_d ∩ J) / pT(V_T) | fraction of the truth jet's visible pT that J holds through complete dark hadrons (penalizes R too small) |
| **purity** | Σ_{d complete in J} pT(V_d ∩ J) / pT(J) = `f_full` | fraction of J's pT from complete dark hadrons, of any truth jet (penalizes UE/ISR and partial dark hadrons) |
| **quality** | acceptance × purity | |
| **IoU** | pT(V_T ∩ J) / pT(V_T ∪ J) | overlap of J with the truth jet's visible particles; does not depend on the containment threshold |

An unmatched truth jet scores 0. Truth jets whose dark hadrons have no visible
descendants are skipped. The output also keeps the counts `n_dh` (dark hadrons
of T with visible descendants) and `n_full` (complete ones). Results are shown
both inclusively and in bins of the truth jet's visible pT, pT(V_T).

## Jet shapes

The jet-shape variables defined in `Histogram.py` are computed for the truth
and the matched visible jets, with ΔR in (η, φ) as in `Histogram.py`:

| Variable | Definition | Computed for |
|---|---|---|
| `truth_r90`, `truth_r95`, `truth_r99` | radius around the dark-hadron jet axis containing 90/95/99% of the truth jet's visible pT, pT(V_T) | truth jet T |
| `psi_truth` | Ψ_T(R) = pT(V_T within R of the dark-hadron jet axis) / pT(V_T), at each scanned R | truth jet T |
| `jet_r90`, `jet_r95`, `jet_r99` | radius around the jet axis containing 90/95/99% of the jet's scalar constituent pT | matched jet J |
| `jet_girth` | Σ pT,i ΔR_i / pT(J) | matched jet J |
| `jet_ptd` | √(Σ pT,i²) / Σ pT,i | matched jet J |
| `jet_major`, `jet_minor` | √eigenvalues of the pT²-weighted (Δη, Δφ) covariance | matched jet J |
| `jet_nconst` | number of constituents | matched jet J |

Ψ(r) is the integrated jet shape of Seymour [1]: the fraction of the jet's pT
within r of its axis. The radii are its inverse (Ψ = 0.90, 0.95, 0.99). Following
[1], they are computed over the particles assigned to the jet (or, for the
truth jet, its visible descendants), not over everything in the cone. The truth
radii give a geometric containment radius for the dark-hadron jets that can be
compared directly with the best R from quality and IoU.

## Dijet and transverse mass

The scores above measure containment jet by jet. The event-level quantities
most sensitive to containment are the masses built from the two leading jets
(pT > 15 GeV), computed for every R and R_DH:

| Variable | Definition |
|---|---|
| `mjj` (visible) | invariant mass of the two leading visible jets at R |
| `mt` | transverse mass of the dijet system with the generator-level missing pT, mT² = mjj² + 2 (√(mjj² + pT,jj²) MET − **pT,jj** · **MET**), as in the CMS semi-visible jet search |
| `mjj` (truth) | invariant mass of the two leading dark-hadron jets at R_DH (all dark hadrons, including stable ones); approaches m_med when R_DH holds the whole dark shower |
| `msd1`, `msd2` | soft-drop mass of the leading and subleading visible jet at R (constituents reclustered with Cambridge/Aachen; β = 0, z_cut = 0.1, R0 = R, as in CMS), to compare with the dark-hadron mass |

## Relation to the literature

| Here | Closest standard quantity | Reference |
|---|---|---|
| Ψ_T(R), `truth_r95` | integrated jet shape Ψ(r) and its inverse | Seymour [1] |
| acceptance, purity | energy fraction of correctly clustered particles relative to the true and to the reconstructed jet, with 95% cuts on both | ILD [2] |
| IoU | energy-weighted intersection over union (EIOU) of true and reconstructed showers | Qasim et al. [3] |
| complete dark hadron | "fully contained" boosted W/Z/top jets: all decay partons within ΔR < 0.75 R (ATLAS) or ΔR < 0.6 of an AK8 jet (CMS); binary and parton-level, where ours is pT-weighted and hadron-level | [4], [5] |
| quality = acceptance × purity | no standard equivalent found; the "quality" of [6] is a different, mass-peak-based measure | [6] |
| the trade-off in R | out-of-cone losses (hadronization, ∝ 1/R) against UE contamination (∝ R²) | [7] |

1. M. H. Seymour, *Jet shapes in hadron collisions: higher orders, resummation and hadronization*, Nucl. Phys. B513 (1998) 269, [arXiv:hep-ph/9707338](https://arxiv.org/abs/hep-ph/9707338)
2. B. Bliewert et al., *Towards an update of the ILD ZHH analysis*, [arXiv:2410.15323](https://arxiv.org/abs/2410.15323)
3. S. R. Qasim, K. Long, J. Kieseler, M. Pierini, R. Nawaz, *Multi-particle reconstruction in the High Granularity Calorimeter using object condensation and graph neural networks*, [arXiv:2106.01832](https://arxiv.org/abs/2106.01832)
4. ATLAS Collaboration, *Performance of top-quark and W-boson tagging with ATLAS in Run 2 of the LHC*, Eur. Phys. J. C 79 (2019) 375, [arXiv:1808.07858](https://arxiv.org/abs/1808.07858)
5. CMS Collaboration, *Identification of heavy, energetic, hadronically decaying particles using machine-learning techniques*, JINST 15 (2020) P06005, [arXiv:2004.08262](https://arxiv.org/abs/2004.08262)
6. M. Cacciari, J. Rojo, G. P. Salam, *Quantifying the performance of jet definitions for kinematic reconstruction at the LHC*, JHEP 0812 (2008) 032, [arXiv:0810.1304](https://arxiv.org/abs/0810.1304)
7. M. Dasgupta, L. Magnea, G. P. Salam, *Non-perturbative QCD effects in jets at hadron colliders*, JHEP 0802 (2008) 055, [arXiv:0712.3014](https://arxiv.org/abs/0712.3014)

## Files and running

- `radius.py` (library): clustering, matching, metrics, dijet masses.
- `scan.py`: runs the scan for one `run_model` directory (needs `events.root`
  and `Meta.pkl`; default R = 0.2–2.0
  and R_DH = 0.8, 1.0; 0.8 and 1.0 are always included in both grids; set
  `--truth-radii` for the full 2D grid) and writes `JetRadius.pkl`.
- `plot.py`: scores vs R at one R_DH (inclusive and in bins of truth visible
  pT) and vs the model parameter, 2D maps of quality and IoU over (R_DH, R)
  with the best R per R_DH marked, the best-R points of all models overlaid
  on the same axes and as R / R_DH; distributions of the scores and jet shapes
  at several R, and of the truth containment radii at several R_DH; the dijet
  and transverse masses vs R and R_DH. R_DH is on the horizontal axis, since
  it is chosen first and R is then chosen given R_DH. With `--displays`, also event displays
  and all truth jets stacked around their axes (needs `events.root` in the
  model directory).

```bash
# full 2D scan for one model directory made by run_model
python3 JetRadiusOptimization/scan.py -d models/<model> --truth-radii $(seq 0.2 0.2 2.0)

# optional: first check our clustering against the stored GenFatJet and DarkHadronJet
python3 JetRadiusOptimization/scan.py -d models/<model> --check-clustering

# plots for all scanned models, one curve per value of a model parameter
python3 JetRadiusOptimization/plot.py -m "models/*" -x rinv

# also event displays and stacked truth jets (reads events.root again)
python3 JetRadiusOptimization/plot.py -m "models/*" -x rinv --displays
```

## Output

- `JetRadius.pkl` (from `scan.py`): the model metadata, the radius grids, and
  one row per (event, truth jet, R_DH, R) with the scores, `f_full`,
  `f_partial`, `f_nodh`, `n_dh`, `n_full`, the truth and matched-jet pT, and
  the jet-shape variables above; and one row per (event, R or R_DH) with the
  number of jets, `mjj`, `mt`, `msd1` and `msd2`.
- `JetRadius_plots/` (from `plot.py`, `--dir`):

| File | Shows |
|---|---|
| `<score>.pdf` | each score vs R at one R_DH (`--truth-radius`, default 0.8), one curve per model |
| `summary_<score>.pdf`, `summary_<score>_quantiles.pdf` | the score vs the model parameter (`-x`): mean at several R and at the best R; median, 16–84% and 5–95% ranges at R = R_DH and at the best R |
| `<score>_ptbins_<model>.pdf` | each score vs R in bins of the truth jet's visible pT (`--pt-bins`) |
| `best_radius_vs_pt_<score>.pdf` | best R per truth-pT bin; bins with fewer than `--min-jets` truth jets, or a zero score at every R, have no best R |
| `truth_pt_dist.pdf` | truth jet visible pT, all models overlaid |
| `quality_2d_<model>.pdf`, `iou_2d_<model>.pdf` | the mean score over (R_DH, R), with the best R per R_DH marked and the mean truth ΔR₉₀/₉₅/₉₉ drawn |
| `best_radius_<score>.pdf` | the best-R points of all models overlaid, with the truth ΔR₉₅ (all three for one model) |
| `best_ratio_<score>.pdf`, `best_ratio_<model>.pdf` | best R / R_DH vs R_DH: per score for all models, and per model for all scores with truth ΔR₉₅,₉₉ / R_DH |
| `truth_jets_vs_rdh.pdf`, `mjj_truth_vs_rdh.pdf` | inputs for choosing R_DH: truth jets per event, the pT fraction in the two leading ones, and their dijet mass |
| `mjj_vs_r.pdf`, `mt_vs_r.pdf`, `msd_vs_r.pdf`, `<mjj\|mt\|msd>_dist_<model>.pdf` | visible dijet, transverse and soft-drop mass vs R (median, 16–84% range and relative width; m_jj and m_T divided by m_med, m_SD by the dark-hadron mass m_π = m_ρ, when known) and their distributions at several R |
| `<quantity>_dist_<model>.pdf` | distributions of the scores, Ψ_T and the jet shapes at several R (`--dist-radii`) |
| `truth_radius_dist_<model>.pdf`, `truth_r<pct>_by_rdh_<model>.pdf` | distributions of the truth ΔR₉₀/₉₅/₉₉, and of each at several R_DH (`--dist-truth-radii`) |
| `truth_radius_vs_rdh.pdf` | mean truth ΔR₉₀/₉₅/₉₉ vs R_DH |
| `event_<i>_<model>.pdf` (`--displays`) | one event per truth-pT bin: particles around the leading truth jet by origin, its truth ΔR₉₀/₉₅/₉₉, and the matched jet at each R (`--display-radii`) with its own ΔR₉₅ |
| `stacked_<model>[_pt<bin>].pdf` (`--displays`) | all truth jets stacked around their axes, one image per particle origin (this truth jet's dark hadrons, other dark hadrons, ISR/UE), and pT vs ΔR stacked by origin; inclusive and per truth-pT bin |

## Sanity test

`scan.py --check-clustering` reclusters R = 0.8 and R_DH = 0.8 and compares
the jets with the stored `GenFatJet` and `DarkHadronJet` constituent by
constituent. A few jets can differ
through numerical near-ties in anti-kT: Delphes clustered the unrounded
momenta, while we cluster the float32 values stored in the file. A mismatched
event whose clustering changes when the inputs are nudged by a relative 1e-6 is
counted in `unstable_jets`; anything else is counted in `mismatched_jets`,
which should be 0.

## Known limitations / follow-ups

- **Dark-parton stage:** the same scan one step earlier (dark partons → dark
  hadrons), using `darkancestry` with partons as targets, would complete the
  parton → hadron → visible chain.
- **Generator level only:** visible jets are clustered from `GenCandidate`.
  A reco-level study is needed, with its own metrics where PF →
  generator-particle links are not available.
- **Choice of score:** quality and IoU can point to different radii; which one
  to use is open.
