"""Phase-1 MLIP L0-L5 diagnostic framework for the GST225 benchmark.

The framework this implements is ``validation_methods_L0-L5.md``; the protocol
and data-role rules come from ``database_link.md``.  Two of those rules are
enforced in code rather than left to discipline:

* labels from different ``protocol_id`` are never pooled into one statistic;
* a metric with unknown reference spread, or fewer than two independent
  reference trajectories, can only ever be ``provisional`` -- never graded.

Module map
----------
``paths``          read-only reference locations and the writable eval roots
``schema``         Protocol / ModelSpec / StructureRef / MetricResult + grading
``protocols``      registry of the GST225 protocols and their measured distances
``io_vasp``        fast XDATCAR / OUTCAR / INCAR readers and ``Trajectory``
``io_labeled``     labelled extxyz, virial convention audit, leakage screen
``thermo``         temperature and timestep recovered from a bare trajectory
``datasets``       loaders binding each reference dataset to its protocol
``state_points``   AIMD state points paired with their matched MLIP runs
``calculators``    pinned MACE adapter and batch prediction
``cache``          npz cache so notebooks never repay an expensive evaluation
``neighbors``      frozen ``StructureParams`` and the RDF/ADF/CN primitives
``md_runner``      MLIP MD under the same protocol as the AIMD reference
``l0_reference``   L0 numerical consistency
``l1_pes``         L1 PES sanity and stability (the gate for L2-L5)
``l2_static``      L2 relaxation, phase ordering, EOS
``l3a_short_range``  L3a short-range order and chemical order
``l3b_medium_range`` L3b rings, ALTBC, S(q)
``l4_dynamics``    L4 MSD and self-diffusion
``l5_pathway``     L5 temperature-programme pathway comparison
``failure_export`` de-duplicated candidate structures for the next DFT round
``plotting``       shared style and reusable panels
"""

from . import paths, schema  # noqa: F401

__version__ = "0.1.0"
__all__ = ["paths", "schema"]
