#!/usr/bin/env python
"""Can the H5 relabels be a protected audit set?  Screen them against every GST
training set in the workspace.

Same test as the retired ``screen_al_ladder_306.py``: composition first (a necessary
condition for a duplicate), then the total-RDF fingerprint against every
same-composition training frame, duplicate threshold 0.02.  The ladder from
gen6 onward and the aging set *do* contain Ge68Sb68Te170, so composition alone
cannot settle it.

Writes ``data/reference/phase1_h5_relabel.leakage.json`` bound to the sha256 of
the harvested file -- :func:`mlipeval.datasets.labelled_h5` grants
``protected_audit`` only against that record -- and ``results/h5_leakage.csv``.

    python harvest_phase1.py && python screen_h5_leakage.py

No model is loaded; safe on the login node.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from ase.io import read

from mlipeval import datasets, paths
from mlipeval.neighbors import total_rdf_fingerprint
from mlipeval.schema import DataRole

DUP = 0.02
SPECIALISED = paths.WORKSPACE / "MLIP_run/specialized"
AGING = Path("/scratch/t0937929/myproject/1-GST225-aging/dataset")
LADDER = SPECIALISED / "al_ladder/train_sets"
TRAINING_SETS = {
    "GST-ACE-24": [paths.EXTERNAL_ACE],
    "processed_NE": [paths.EXTERNAL_NE],
    "spec splits (gst_*, gst_ne_*)": sorted(SPECIALISED.glob("data/gst*.xyz")),
    "GST225-aging": [AGING / "GST225-train-updated.xyz", AGING / "GST225-val-updated.xyz"],
    **{f"AL {p.stem}": [p] for p in sorted(LADDER.glob("[!.]*.xyz"))},
}

labelled = datasets.labelled_h5(role=DataRole.DIAGNOSTIC)
formula = labelled.frames[0].get_chemical_formula(mode="hill")
assert all(a.get_chemical_formula(mode="hill") == formula for a in labelled.frames)
h5_fp = np.array([total_rdf_fingerprint(a, r_max=6.0, bins=60) for a in labelled.frames])
print(f"H5: {len(labelled)} frames of {formula}", flush=True)

rows = []
for name, files in TRAINING_SETS.items():
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        rows.append({"training_set": name, "n_frames": 0, "composition_matches": 0,
                     "V_per_atom_median": np.nan, "nearest_rdf": np.nan,
                     "verdict": f"MISSING {missing}"})
        continue
    frames = [a for f in files for a in read(str(f), index=":")]
    cands = [a for a in frames if a.get_chemical_formula(mode="hill") == formula]
    if not cands:
        rows.append({"training_set": name, "n_frames": len(frames), "composition_matches": 0,
                     "V_per_atom_median": np.nan, "nearest_rdf": np.nan,
                     "verdict": "clean (no composition match)"})
    else:
        fps = np.array([total_rdf_fingerprint(a, r_max=6.0, bins=60) for a in cands])
        near = float(np.linalg.norm(h5_fp[:, None, :] - fps[None, :, :], axis=2).min())
        rows.append({"training_set": name, "n_frames": len(frames),
                     "composition_matches": len(cands),
                     "V_per_atom_median": round(float(np.median([a.get_volume() / len(a) for a in cands])), 3),
                     "nearest_rdf": round(near, 4),
                     "verdict": "CONTAMINATED" if near < DUP else "clean"})
    print(f"{name}: {rows[-1]['verdict']}", flush=True)

out = pd.DataFrame(rows)
verdict = "clean" if out["verdict"].str.startswith("clean").all() else "not clean"
record = {
    "file": str(paths.H5_LABELS),
    "sha256": datasets._sha256(paths.H5_LABELS),
    "verdict": verdict,
    "threshold_rdf": DUP,
    "h5_volume_per_atom_A3": round(float(np.median(labelled.volumes / labelled.n_atoms)), 3),
    "screened_at": datetime.now().isoformat(timespec="seconds"),
    "rows": json.loads(out.to_json(orient="records")),
}
paths.H5_LEAKAGE.write_text(json.dumps(record, indent=2))
dest = HERE.parent / "results" / "h5_leakage.csv"
out.to_csv(dest, index=False)
print(f"\nverdict: {verdict}\nwrote {paths.H5_LEAKAGE}\nwrote {dest}")
print(out.to_string(index=False))
