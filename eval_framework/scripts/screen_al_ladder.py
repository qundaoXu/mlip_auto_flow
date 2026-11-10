#!/usr/bin/env python
"""Leakage screen for the AL-replay ladder, one generation at a time.

The framework's rule is that a GST-specialised potential may not be scored on a
reference its training set contains, and that the finding has to be measured
rather than inherited. `mace_gst_aging` was screened against the 4577-frame
deployed set; the ladder points are NOT subsets of it (prod_train drops 151 of
gen9's frames), so each generation is screened on its own.

    python screen_al_ladder.py            # -> results/al_ladder_leakage.csv

No model is loaded, so this is safe on the login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd

from mlipeval import paths
from mlipeval.leakage import _screen_external

TRAIN = paths.AL_LADDER_TRAIN_SETS
LADDER = [("al_gen0", "gen0"), ("al_gen3", "gen3"), ("al_gen4", "gen4"),
          ("al_gen6", "gen6"), ("al_gen7", "gen7"), ("al_gen8", "gen8"),
          ("al_gen9", "gen9"), ("al_prod", "prod_train")]

frames = []
for key, stem in LADDER:
    path = TRAIN / f"{stem}.xyz"
    print(f"screening {key} ({path.name}) ...", flush=True)
    df = _screen_external(path, key)
    frames.append(df)
    bad = df[df["verdict"].astype(str).str.startswith("CONTAMINATED")]
    print(f"  {len(bad)} contaminated reference(s): "
          f"{', '.join(bad['reference']) if len(bad) else 'none'}", flush=True)

out = pd.concat(frames, ignore_index=True)
dest = HERE.parent / "results" / "al_ladder_leakage.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False)
print(f"\nwrote {dest}")
print(out.to_string(index=False))
