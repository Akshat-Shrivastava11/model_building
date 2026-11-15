"""Run the jet-radius scan for one generated model.

Reads events.root and Meta.pkl from a model directory made by run_model and
writes JetRadius.pkl next to them: the model metadata and one row of metrics
per (event, truth jet, truth radius, radius), as defined in radius.py.

    python3 JetRadiusOptimization/scan.py -d models/<model>
    python3 JetRadiusOptimization/scan.py -d models/<model> --truth-radii $(seq 0.2 0.2 2.0)
"""

import os
import pickle

import numpy as np
from magiconfig import ArgumentParser, ArgumentDefaultsRawHelpFormatter

from radius import PT_MIN, check_clustering, event_scan, load, radius_scan

ALWAYS_RADII = (0.8, 1.0)

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("-d", "--dir", required=True, help="model directory with events.root and Meta.pkl")
    parser.add_argument("--radii", type=float, nargs="+", default=np.round(np.arange(0.2, 2.05, 0.1), 1).tolist(), help="visible jet radii; 0.8 and 1.0 are always added")
    parser.add_argument("--truth-radii", type=float, nargs="+", default=[0.8], help="radii of the dark-hadron (truth) jets; 0.8 and 1.0 are always added")
    parser.add_argument("--pt-min", type=float, default=PT_MIN, help="minimum clustered jet pT [GeV]")
    parser.add_argument("--min-containment", type=float, default=0.95, help="fraction of a dark hadron's visible pT a jet must hold for the dark hadron to be complete in the jet (1 = every visible descendant)")
    parser.add_argument("-n", "--max-events", type=int, default=None, help="number of events")
    parser.add_argument("--check-clustering", action="store_true", help="first compare our R = 0.8 clustering with the stored GenFatJet and DarkHadronJet")
    args = parser.parse_args()
    # the standard radii 0.8 and 1.0 are always part of both grids
    args.radii = sorted({round(r, 3) for r in args.radii} | set(ALWAYS_RADII))
    args.truth_radii = sorted({round(r, 3) for r in args.truth_radii} | set(ALWAYS_RADII))

    with open(os.path.join(args.dir, "Meta.pkl"), "rb") as f:
        model = pickle.load(f)
    events = load(os.path.join(args.dir, "events.root"))
    if args.max_events is not None:
        events = events[: args.max_events]
    if args.check_clustering:
        for candidates, jets in (("GenCandidate", "GenFatJet"), ("DarkHadronCandidate", "DarkHadronJet")):
            print(jets, check_clustering(events, candidates, jets))

    output = {
        "model": model,
        "radii": args.radii,
        "truth_radii": args.truth_radii,
        "pt_min": args.pt_min,
        "min_containment": args.min_containment,
        "metrics": radius_scan(events, args.radii, truth_radii=args.truth_radii,
                               pt_min=args.pt_min, min_containment=args.min_containment),
        "events": event_scan(events, args.radii, args.truth_radii, pt_min=args.pt_min),
    }
    with open(os.path.join(args.dir, "JetRadius.pkl"), "wb") as out:
        pickle.dump(output, out)
