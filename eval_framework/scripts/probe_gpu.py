#!/usr/bin/env python
"""GPU smoke probe: confirm MACE runs on CUDA here and measure its speed.

    qsub pbs/mlipeval_gpu_probe.pbs

Writes nothing into the result set -- it only reports what a GPU node can do, so
the walltimes in ``submit_pbs.py`` can be based on a measurement instead of a
guess.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlipeval import datasets
from mlipeval.calculators import build_model
from mlipeval.md_runner import MDSettings, run_md
from mlipeval.pbs import describe_environment


def main() -> int:
    print(json.dumps(describe_environment(), indent=2))

    t0 = time.time()
    model = build_model()
    print(f"\nmodel loaded in {time.time() - t0:.1f}s on device={model.spec.device}, "
          f"dtype={model.spec.dtype}")

    # 194-atom cubic frames of the L0 diagnostic set: the L0/L2 workload.
    # Ten *different* frames, because ASE's Calculator.check_state returns the
    # cached result when handed an identical configuration -- timing the same
    # frame repeatedly measures the cache, not the model.
    labelled = datasets.labelled_diagnostic()
    cubic = [f for f, s in zip(labelled.frames, labelled.tags["subset"]) if s == "eos_cubic"]
    frames = cubic[:11]
    model.energy_forces_stress(frames[0])                  # warm-up / JIT
    t0 = time.time()
    for frame in frames[1:]:
        model.energy_forces_stress(frame)
    sp = (time.time() - t0) / (len(frames) - 1)
    print(f"single point, {len(frames[0])} atoms : {sp*1000:.1f} ms  "
          f"-> {len(labelled)} diagnostic frames ~ {sp*len(labelled)/60:.1f} min")

    # 306-atom MD: the L3/L4/L5 workload
    start = datasets.preflight("nvt_1200K", stride=1).atoms(0)
    settings = MDSettings(ensemble="nvt", temperature_k=300.0, timestep_fs=3.0,
                          n_steps=30, sample_every=10, protocol_id="gpu_probe")
    t0 = time.time()
    result = run_md(start, model, settings)
    rate = settings.n_steps / (time.time() - t0)
    print(f"MD, {len(start)} atoms          : {rate:.2f} step/s  "
          f"-> 10000 steps ~ {10000/rate/60:.1f} min, 20000 steps ~ {20000/rate/60:.1f} min")
    print(f"MD completed={result.completed}  mean T={np.nanmean(result.temperature):.0f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
