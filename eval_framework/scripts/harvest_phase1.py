#!/usr/bin/env python
"""Bring the finished phase-1 reference data into the framework.

Three things, all read-only on the VASP tree:

1. harvest the 120 H5 static relabels into ``data/reference/phase1_h5_relabel.extxyz``
   (+ ``.manifest.json`` with per-task OUTCAR sha256 and SCF step counts) --
   the L0 ``protected_audit`` set;
2. harvest every ionic step of the crystalline L1/L2/H1/H2 tasks into
   ``data/reference/phase1_static_diagnostic.extxyz`` -- the L0 ``diagnostic`` set;
3. print what the framework can now see: the H3/H4 canonical AIMD segment table
   and the reference inventory.

    python harvest_phase1.py [--skip-h5]
    # then: python screen_h5_leakage.py && python screen_diagnostic_leakage.py

``--skip-h5`` leaves the H5 file untouched: re-harvesting it changes its sha256,
and ``protected_audit`` is then withheld until the H5 screen is rerun.

No model is loaded; safe on the login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd

from mlipeval import datasets

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)

harvests = [] if "--skip-h5" in sys.argv else [("H5", datasets.harvest_h5_labels)]
harvests.append(("diagnostic", datasets.harvest_static_diagnostic))
for label, harvest in harvests:
    manifest = harvest()
    print(f"{label}: {manifest['n_frames']} frames -> {manifest['file']}")
    print(f"    sha256 {manifest['sha256']}")
    if manifest["skipped"]:
        print("    skipped:")
        for task, why in manifest["skipped"].items():
            print(f"      {task}: {why}")

status = datasets.canonical_aimd_status()
print("\ncanonical AIMD segments (H3 + H4):")
print(status.to_string(index=False))
incomplete = status[~status["complete"]]
if len(incomplete):
    print(f"\n{len(incomplete)} segment(s) below their planned length:")
    print(incomplete[["task", "frames", "planned_steps"]].to_string(index=False))

print("\nreference inventory:")
print(datasets.inventory().to_string(index=False))
