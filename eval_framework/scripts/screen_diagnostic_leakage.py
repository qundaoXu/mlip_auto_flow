#!/usr/bin/env python
"""Which frames of the L0 diagnostic set has each model already been trained on?

Same test as ``screen_h5_leakage.py`` -- composition first (a necessary
condition for a duplicate), then the total-RDF fingerprint against every
same-composition training frame, duplicate threshold 0.02 -- with two
differences that the diagnostic set needs:

* the verdict is **per frame**, not per set.  The diagnostic set is not a
  protected audit set that is either clean or not; it is scored per model, and
  a model is scored only on the frames it has not seen
  (:func:`mlipeval.datasets.diagnostic_training_overlap`);
* composition is compared as the **reduced** formula.  The set holds 9-atom MP
  cells and 2-3 atom element cells, which a training set may well contain as
  supercells; the RDF fingerprint is per-density, so a supercell of the same
  structure matches, and an exact-formula test would have let it through.

Writes ``data/reference/phase1_static_diagnostic.leakage.json`` bound to the
sha256 of the harvested file, and ``results/diagnostic_leakage.csv``.

    python harvest_phase1.py && python screen_diagnostic_leakage.py

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
from mlipeval.leakage import TRAINING_SET_FILES
from mlipeval.neighbors import total_rdf_fingerprint

DUP = 0.02


def reduced(atoms) -> str:
    return atoms.get_chemical_formula(mode="hill", empirical=True)


labelled = datasets.labelled_diagnostic()
configs = labelled.config_types()
by_comp: dict[str, list[int]] = {}
for i, a in enumerate(labelled.frames):
    by_comp.setdefault(reduced(a), []).append(i)
fp = np.array([total_rdf_fingerprint(a, r_max=6.0, bins=60) for a in labelled.frames])
print(f"diagnostic set: {len(labelled)} frames, compositions {sorted(by_comp)}", flush=True)

rows, record_sets = [], {}
for name, files in TRAINING_SET_FILES.items():
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"training set '{name}': missing {missing}")
    train = [a for f in files for a in read(str(f), index=":")]
    overlapping: list[str] = []
    for comp, members in sorted(by_comp.items()):
        cands = [a for a in train if reduced(a) == comp]
        nearest = np.full(len(members), np.inf)
        if cands:
            tfp = np.array([total_rdf_fingerprint(a, r_max=6.0, bins=60) for a in cands])
            nearest = np.linalg.norm(fp[members][:, None, :] - tfp[None, :, :], axis=2).min(axis=1)
        hit = [configs[m] for m, d in zip(members, nearest) if d < DUP]
        overlapping += hit
        subsets = sorted({str(labelled.tags["subset"][m]) for m in members})
        rows.append({
            "training_set": name, "composition": comp, "subsets": ",".join(subsets),
            "n_diagnostic_frames": len(members), "composition_matches": len(cands),
            "nearest_rdf": round(float(nearest.min()), 4) if cands else np.nan,
            "n_overlapping": len(hit),
            "overlapping_tasks": ",".join(sorted({h.split(":")[0] for h in hit})),
        })
    record_sets[name] = {"n_frames": len(train), "overlapping_frames": overlapping}
    print(f"{name}: {len(overlapping)} overlapping frame(s)", flush=True)

out = pd.DataFrame(rows)
record = {
    "file": str(paths.DIAGNOSTIC_LABELS),
    "sha256": datasets._sha256(paths.DIAGNOSTIC_LABELS),
    "threshold_rdf": DUP,
    "composition_match": "reduced formula",
    "screened_at": datetime.now().isoformat(timespec="seconds"),
    "training_sets": record_sets,
    "rows": json.loads(out.to_json(orient="records")),
}
paths.DIAGNOSTIC_LEAKAGE.write_text(json.dumps(record, indent=2))
dest = HERE.parent / "results" / "diagnostic_leakage.csv"
out.to_csv(dest, index=False)
print(f"\nwrote {paths.DIAGNOSTIC_LEAKAGE}\nwrote {dest}")
pd.set_option("display.width", 220)
print(out[out["composition_matches"] > 0].to_string(index=False))
