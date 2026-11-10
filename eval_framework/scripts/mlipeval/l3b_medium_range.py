"""L3b -- medium-range order and topology.

Scope (``validation_methods_L0-L5.md`` L3b):

1. ring-size distribution;
2. four-membered-ring fraction and planarity;
3. ALTBC (angular-limited three-body correlation);
4. ``S(q)`` and the first sharp diffraction peak.

``S(q)`` is computed by summing over the reciprocal-lattice vectors the periodic
cell allows, not by Fourier transforming a truncated ``g(r)`` -- for a 22 A box
the transform is meaningless below ~0.63 1/A and produces negative ``S(q)``
there.  The FSDP of a-GST225 sits near 1 1/A, close enough to the box limit
``2 pi / L = 0.285 1/A`` that whether it is resolved at all must be reported.

Ring definition
---------------
Exactly one ring algorithm is implemented, as the framework requires: the
**King shortest-path criterion**.  For every atom ``i`` and every pair of its
bonded neighbours ``(j, k)``, the shortest ``j -> k`` path that avoids ``i`` is
found; together with ``i`` it closes a ring.  Rings are deduplicated by their
node set and truncated at ``params.max_ring_size``.  Changing this choice
changes every ring number, so it is recorded in ``StructureParams`` and echoed
into every metric.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from .io_vasp import Trajectory
from .neighbors import StructureParams, all_pairs, bond_list, pair_key
from .schema import EvidenceTier, MetricResult, ResultSet, Status

#: Coherent neutron scattering lengths (fm), for the neutron-weighted S(q).
NEUTRON_B = {"Ge": 8.185, "Sb": 5.57, "Te": 5.80}


def _result(metric_id: str, value, unit: str, **kw) -> MetricResult:
    kw.setdefault("evidence_tier", EvidenceTier.A)
    kw.setdefault("status", Status.PROVISIONAL)
    return MetricResult(metric_id=metric_id, level="L3b", value=value, unit=unit, **kw)


# --------------------------------------------------------------------------
# 1-2. rings
# --------------------------------------------------------------------------
def adjacency(atoms: Atoms, params: StructureParams) -> list[list[int]]:
    """Bonded adjacency list under the frozen per-pair cutoffs."""
    i, j, _, _ = bond_list(atoms, params)
    adj: list[list[int]] = [[] for _ in range(len(atoms))]
    for a, b in zip(i, j):
        adj[a].append(int(b))
    return adj


def _shortest_path(adj: Sequence[Sequence[int]], start: int, goal: int, blocked: int, max_len: int) -> list[int] | None:
    """BFS shortest ``start -> goal`` path avoiding ``blocked``, capped at ``max_len``."""
    if start == goal:
        return [start]
    prev = {start: -1}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_len:
            continue
        for nxt in adj[node]:
            if nxt == blocked or nxt in prev:
                continue
            prev[nxt] = node
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            queue.append((nxt, depth + 1))
    return None


def king_rings(atoms: Atoms, params: StructureParams) -> dict[int, list[tuple[int, ...]]]:
    """All King shortest-path rings up to ``params.max_ring_size``.

    Returns ``{ring_size: [ring_as_node_tuple, ...]}`` with rings deduplicated
    by their node set.
    """
    adj = adjacency(atoms, params)
    max_size = params.max_ring_size
    seen: set[frozenset[int]] = set()
    rings: dict[int, list[tuple[int, ...]]] = {}
    for centre in range(len(atoms)):
        neighbours = adj[centre]
        for a_idx in range(len(neighbours)):
            for b_idx in range(a_idx + 1, len(neighbours)):
                j, k = neighbours[a_idx], neighbours[b_idx]
                path = _shortest_path(adj, j, k, centre, max_size - 1)
                if path is None:
                    continue
                ring = (centre, *path)
                size = len(ring)
                if size < 3 or size > max_size:
                    continue
                key = frozenset(ring)
                if len(key) != size or key in seen:
                    continue
                seen.add(key)
                rings.setdefault(size, []).append(tuple(ring))
    return rings


def ring_statistics(
    traj: Trajectory,
    params: StructureParams,
    *,
    stride: int = 1,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Rings per atom by size, averaged over frames, plus 4-ring planarity."""
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "ring_statistics",
            {"traj": trajectory_identity(traj), "params": params.to_dict(), "stride": stride},
            lambda: ring_statistics(traj, params, stride=stride, cache=False),
        )

    sizes = np.arange(3, params.max_ring_size + 1)
    counts = np.zeros(sizes.size, dtype=float)
    planarity: list[float] = []
    n_frames = 0
    for k in range(0, len(traj), stride):
        atoms = traj.atoms(k)
        rings = king_rings(atoms, params)
        for size, members in rings.items():
            if 3 <= size <= params.max_ring_size:
                counts[size - 3] += len(members)
        for ring in rings.get(4, []):
            planarity.append(ring_planarity(atoms, ring))
        n_frames += 1
    counts /= max(n_frames, 1) * traj.n_atoms
    return {
        "sizes": sizes,
        "rings_per_atom": counts,
        "fraction": counts / counts.sum() if counts.sum() > 0 else counts,
        "planarity_4ring": np.asarray(planarity),
        "n_frames": np.array([n_frames]),
    }


def ring_planarity(atoms: Atoms, ring: Sequence[int]) -> float:
    """RMS out-of-plane deviation of a ring, in A (0 = perfectly planar).

    Positions are unwrapped relative to the first ring member so that a ring
    crossing the periodic boundary is not reported as buckled.
    """
    indices = list(ring)
    cell = atoms.cell.array
    inverse = np.linalg.inv(cell)
    origin = atoms.positions[indices[0]]
    points = []
    for index in indices:
        delta = atoms.positions[index] - origin
        fractional = delta @ inverse
        fractional -= np.round(fractional)
        points.append(fractional @ cell)
    points = np.asarray(points)
    centred = points - points.mean(axis=0)
    # smallest singular direction is the plane normal
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    return float(np.sqrt(np.mean((centred @ normal) ** 2)))


# --------------------------------------------------------------------------
# 3. ALTBC
# --------------------------------------------------------------------------
def altbc(
    traj: Trajectory,
    params: StructureParams,
    *,
    stride: int = 1,
    species: str | None = None,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Angular-limited three-body correlation.

    For every central atom, all bond pairs whose angle lies within
    ``params.altbc_angle_window_deg`` of 180 degrees contribute their two bond
    lengths to a 2D histogram.  In GST the Peierls distortion shows up as
    off-diagonal weight (one short, one long bond); a model that over-symmetrises
    the local octahedra piles the density onto the diagonal.
    """
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "altbc",
            {"traj": trajectory_identity(traj), "params": params.to_dict(),
             "stride": stride, "species": species},
            lambda: altbc(traj, params, stride=stride, species=species, cache=False),
        )

    from .neighbors import neighbour_list

    symbols = np.asarray(traj.symbols)
    edges = np.linspace(2.0, params.altbc_r_max, params.altbc_bins + 1)
    hist = np.zeros((params.altbc_bins, params.altbc_bins), dtype=float)
    cos_limit = np.cos(np.radians(180.0 - params.altbc_angle_window_deg))
    n_frames = 0

    for k in range(0, len(traj), stride):
        atoms = traj.atoms(k)
        i, j, d, D = neighbour_list(atoms, params.altbc_r_max, "ijdD")
        order = np.argsort(i, kind="stable")
        i, j, d, D = i[order], j[order], d[order], D[order]
        starts = np.searchsorted(i, np.arange(len(atoms)), side="left")
        ends = np.searchsorted(i, np.arange(len(atoms)), side="right")
        for centre in range(len(atoms)):
            if species is not None and symbols[centre] != species:
                continue
            s0, s1 = starts[centre], ends[centre]
            if s1 - s0 < 2:
                continue
            dd, DD = d[s0:s1], D[s0:s1]
            unit = DD / dd[:, None]
            cos = unit @ unit.T
            a_idx, b_idx = np.triu_indices(dd.size, k=1)
            sel = cos[a_idx, b_idx] <= cos_limit
            if not sel.any():
                continue
            d1, d2 = dd[a_idx[sel]], dd[b_idx[sel]]
            h, _, _ = np.histogram2d(
                np.concatenate([d1, d2]), np.concatenate([d2, d1]), bins=[edges, edges]
            )
            hist += h
        n_frames += 1

    total = hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {
        "r": centres,
        "hist": hist / total if total > 0 else hist,
        "n_frames": np.array([n_frames]),
        "asymmetry": np.array([_altbc_asymmetry(centres, hist)]),
    }


def _altbc_asymmetry(centres: np.ndarray, hist: np.ndarray) -> float:
    """Mean ``|d1 - d2|`` over the ALTBC map -- the Peierls bond-length alternation."""
    total = hist.sum()
    if total <= 0:
        return float("nan")
    d1 = centres[:, None]
    d2 = centres[None, :]
    return float((hist * np.abs(d1 - d2)).sum() / total)


# --------------------------------------------------------------------------
# 4. structure factor
# --------------------------------------------------------------------------
def structure_factor(
    rdf: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    number_density: float,
    params: StructureParams,
    *,
    weighting: str = "neutron",
    lorch: bool = True,
) -> dict[str, np.ndarray]:
    """Faber-Ziman ``S(q)`` obtained by Fourier transforming the partial RDFs.

    Kept as a cross-check of :func:`structure_factor_direct` at intermediate
    ``q`` only.  Below ``2 pi / r_max`` (0.63 1/A for ``r_max = 10 A``) the
    truncation ripples dominate and this estimator returns unphysical values --
    for the 306-atom GST cell it puts ``S(0.4) = 6.7`` and ``S(0.7) = -0.6``.
    Do not use it for the FSDP.

    ``lorch=True`` applies the Lorch window ``sin(pi r / r_max) / (pi r / r_max)``
    which suppresses the truncation ripples caused by the finite ``r_max``; it
    slightly broadens genuine peaks, so it is recorded in the metric ``extra``.
    """
    symbols = np.asarray(symbols)
    species = sorted(set(symbols.tolist()))
    r = np.asarray(rdf["r"])
    dr = r[1] - r[0]
    q = np.linspace(params.sq_q_min, params.sq_q_max, params.sq_n_q)

    if weighting == "neutron":
        weights_of = {s: NEUTRON_B.get(s, 1.0) for s in species}
    elif weighting == "number":
        weights_of = {s: 1.0 for s in species}
    else:
        raise ValueError(f"unknown weighting '{weighting}'")

    concentration = {s: float((symbols == s).sum()) / len(symbols) for s in species}
    mean_weight = sum(concentration[s] * weights_of[s] for s in species)

    window = np.ones_like(r)
    if lorch:
        x = np.pi * r / params.r_max
        window = np.where(x > 0, np.sin(x) / np.where(x == 0, 1.0, x), 1.0)

    partials: dict[str, np.ndarray] = {}
    total = np.zeros_like(q)
    for a, b in all_pairs(species):
        key = pair_key(a, b)
        integrand = r[None, :] * np.sin(q[:, None] * r[None, :]) * (rdf[key][None, :] - 1.0) * window[None, :]
        s_ab = 1.0 + (4.0 * np.pi * number_density / q) * np.sum(integrand, axis=1) * dr
        partials[key] = s_ab
        weight = (
            concentration[a] * concentration[b] * weights_of[a] * weights_of[b] * (1 if a == b else 2)
        ) / mean_weight**2
        total += weight * (s_ab - 1.0)
    return {"q": q, "S": total + 1.0, "partials": partials, "weighting": weighting, "lorch": lorch}


def structure_factor_direct(
    traj,
    params: StructureParams,
    *,
    stride: int = 1,
    weighting: str = "neutron",
    q_max: float = 6.0,
    n_shells: int = 140,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """``S(q)`` summed directly over the q-vectors the periodic box allows.

    Fourier transforming a truncated ``g(r)`` is unusable at low ``q`` for a box
    this small: with ``r_max = 10 A`` the transform is only meaningful above
    ``2 pi / r_max ~ 0.63 1/A``, and below that it produces large ripples and
    even negative ``S(q)``.  Summing over the reciprocal lattice instead is
    exact for the simulation cell:

        S(q) = 1 + [ <|sum_j b_j exp(i q.r_j)|^2> / N - <b^2> ] / <b>^2

    The smallest accessible wavevector is ``2 pi / L`` (0.285 1/A for the
    22.05 A GST cell), which is the true resolution limit of the box -- and it
    is what decides whether the FSDP is measurable at all.
    """
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "structure_factor_direct",
            {"traj": trajectory_identity(traj), "params": params.to_dict(), "stride": stride,
             "weighting": weighting, "q_max": q_max, "n_shells": n_shells},
            lambda: structure_factor_direct(traj, params, stride=stride, weighting=weighting,
                                            q_max=q_max, n_shells=n_shells, cache=False),
        )

    symbols = np.asarray(traj.symbols)
    species = sorted(set(symbols.tolist()))
    if weighting == "neutron":
        weights = np.array([NEUTRON_B.get(s, 1.0) for s in symbols])
    elif weighting == "number":
        weights = np.ones(len(symbols))
    else:
        raise ValueError(f"unknown weighting '{weighting}'")
    mean_b = float(weights.mean())
    mean_b2 = float((weights**2).mean())

    cell = traj.cells[0]
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    b_len = np.linalg.norm(reciprocal, axis=1)
    n_max = np.ceil(q_max / b_len).astype(int)

    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in n_max], indexing="ij")
    hkl = np.stack([g.reshape(-1) for g in grids], axis=1)
    q_vectors = hkl @ reciprocal
    q_norm = np.linalg.norm(q_vectors, axis=1)
    # |S(-q)| = |S(q)| for real positions, so only the half space is evaluated.
    first_nonzero = np.where(hkl[:, 0] != 0, hkl[:, 0], np.where(hkl[:, 1] != 0, hkl[:, 1], hkl[:, 2]))
    keep = (q_norm > 1e-8) & (q_norm <= q_max) & (first_nonzero > 0)
    q_vectors, q_norm = q_vectors[keep], q_norm[keep]

    accumulated = np.zeros(q_norm.size)
    n_frames = 0
    chunk = max(1, int(4e7 // max(len(symbols), 1)))  # cap the (n_atoms, n_q) temporary
    for k in range(0, len(traj), stride):
        positions = traj.positions[k]
        for lo in range(0, q_norm.size, chunk):
            block = q_vectors[lo : lo + chunk]
            phase = positions @ block.T                    # (n_atoms, n_q_block)
            amplitude = np.einsum("a,aq->q", weights, np.exp(1j * phase))
            accumulated[lo : lo + chunk] += np.abs(amplitude) ** 2
        n_frames += 1
    accumulated /= max(n_frames, 1) * len(symbols)
    s_of_q = 1.0 + (accumulated - mean_b2) / mean_b**2

    q_min = float(q_norm.min())
    edges = np.linspace(q_min - 1e-9, q_max, n_shells + 1)
    index = np.digitize(q_norm, edges) - 1
    q_shell = np.full(n_shells, np.nan)
    s_shell = np.full(n_shells, np.nan)
    counts = np.zeros(n_shells, dtype=int)
    for shell in range(n_shells):
        sel = index == shell
        counts[shell] = int(sel.sum())
        if counts[shell]:
            q_shell[shell] = float(q_norm[sel].mean())
            s_shell[shell] = float(s_of_q[sel].mean())
    ok = counts > 0
    return {
        "q": q_shell[ok],
        "S": s_shell[ok],
        "n_vectors": counts[ok],
        "q_min_accessible": q_min,
        "weighting": weighting,
        "n_frames": n_frames,
        "method": "direct sum over reciprocal-lattice vectors of the simulation cell",
    }


def fsdp(
    q: np.ndarray,
    s: np.ndarray,
    *,
    q_lo: float = 0.8,
    q_hi: float = 1.7,
    q_min_accessible: float | None = None,
) -> dict[str, float]:
    """First sharp diffraction peak: position, height and FWHM.

    The window defaults to 0.8-1.7 1/A, *below* the principal peak of a-GST225
    (~2.2 1/A).  Searching a single window that spans both would simply return
    the principal peak and label it FSDP.

    ``resolved`` is False when the window reaches below the smallest wavevector
    the box supports, or when the peak sits on the window edge -- for a 306-atom
    GST cell the FSDP is at the very limit of what the box can express, and that
    has to be reported rather than papered over.
    """
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    win = (q >= q_lo) & (q <= q_hi)
    out = {"q_fsdp": np.nan, "S_fsdp": np.nan, "fwhm": np.nan, "resolved": False,
           "window": (q_lo, q_hi), "n_points_in_window": int(win.sum())}
    if win.sum() < 3:
        return out
    qw, sw = q[win], s[win]
    k = int(np.argmax(sw))
    height = float(sw[k])
    half = 0.5 * (height + float(np.min(sw)))
    left = qw[:k][sw[:k] <= half]
    right = qw[k:][sw[k:] <= half]
    on_edge = k == 0 or k == qw.size - 1
    below_box = q_min_accessible is not None and q_lo < q_min_accessible
    out.update({
        "q_fsdp": float(qw[k]),
        "S_fsdp": height,
        "fwhm": float(right[0] - left[-1]) if left.size and right.size else np.nan,
        "resolved": bool(not on_edge and not below_box),
        "on_window_edge": bool(on_edge),
        "below_box_resolution": bool(below_box),
    })
    return out


def principal_peak(q: np.ndarray, s: np.ndarray, *, q_lo: float = 1.7, q_hi: float = 3.2) -> dict[str, float]:
    """Position and height of the principal ``S(q)`` peak (~2.2 1/A in a-GST225)."""
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    win = (q >= q_lo) & (q <= q_hi)
    if win.sum() < 3:
        return {"q_pp": np.nan, "S_pp": np.nan}
    qw, sw = q[win], s[win]
    k = int(np.argmax(sw))
    return {"q_pp": float(qw[k]), "S_pp": float(sw[k])}


# --------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------
@dataclass
class L3bReport:
    params: StructureParams
    rings_ref: dict[str, np.ndarray]
    rings_model: dict[str, np.ndarray] | None
    altbc_ref: dict[str, np.ndarray]
    altbc_model: dict[str, np.ndarray] | None
    sq_ref: dict[str, np.ndarray]
    sq_model: dict[str, np.ndarray] | None
    fsdp_ref: dict[str, float]
    fsdp_model: dict[str, float] | None
    results: ResultSet


def analyse_medium_range(
    reference: Trajectory,
    params: StructureParams,
    *,
    model_traj: Trajectory | None = None,
    stride: int = 1,
    model_stride: int | None = None,
    state_label: str = "state",
    weighting: str = "neutron",
    sq_stride: int = 25,
    sq_q_max: float = 6.0,
    n_independent_references: int = 1,
) -> L3bReport:
    """Ring statistics, ALTBC and S(q) for a reference (+ optional model).

    ``S(q)`` is summed directly over the cell's reciprocal lattice rather than
    Fourier transforming a truncated ``g(r)``; see
    :func:`structure_factor_direct` for why that matters at low ``q``.
    """
    model_stride = stride if model_stride is None else model_stride
    tier = EvidenceTier.A if model_traj is not None else EvidenceTier.S
    ref_pid = reference.protocol.protocol_id if reference.protocol else None
    mod_pid = model_traj.protocol.protocol_id if (model_traj and model_traj.protocol) else None

    results = ResultSet(
        "L3b",
        state=state_label,
        frozen_params=params.to_dict(),
        ring_algorithm=params.ring_algorithm,
        max_ring_size=params.max_ring_size,
        reference=reference.describe(),
        model_trajectory=model_traj.describe() if model_traj else None,
        n_independent_references=n_independent_references,
    )

    def add(metric_id, value, unit, **kw):
        kw.setdefault("evidence_tier", tier)
        kw.setdefault("reference_protocol", ref_pid)
        kw.setdefault("model_protocol", mod_pid)
        kw.setdefault("status", Status.PROVISIONAL)
        return results.add(
            MetricResult(metric_id=f"{metric_id}@{state_label}", level="L3b", value=value, unit=unit, **kw)
        )

    # -- rings -------------------------------------------------------------
    rings_ref = ring_statistics(reference, params, stride=stride)
    rings_model = ring_statistics(model_traj, params, stride=model_stride) if model_traj else None
    for idx, size in enumerate(rings_ref["sizes"]):
        ref_value = float(rings_ref["rings_per_atom"][idx])
        if rings_model is not None:
            model_value = float(rings_model["rings_per_atom"][idx])
            add(
                f"L3b.rings.size{size}",
                model_value,
                "rings/atom",
                extra={"reference": ref_value, "model": model_value, "difference": model_value - ref_value},
            )
        else:
            add(f"L3b.rings_reference.size{size}", ref_value, "rings/atom", status=Status.OK)

    for label, data in (("reference", rings_ref), ("model", rings_model)):
        if data is None:
            continue
        planarity = data["planarity_4ring"]
        if planarity.size:
            add(
                f"L3b.four_ring_planarity.{label}",
                float(planarity.mean()),
                "A rms out-of-plane",
                status=Status.OK if label == "reference" else Status.PROVISIONAL,
                extra={"n_rings": int(planarity.size), "p90": float(np.percentile(planarity, 90))},
            )

    # -- ALTBC -------------------------------------------------------------
    altbc_ref = altbc(reference, params, stride=stride)
    altbc_model = altbc(model_traj, params, stride=model_stride) if model_traj else None
    if altbc_model is not None:
        add(
            "L3b.altbc.asymmetry",
            float(altbc_model["asymmetry"][0]),
            "A",
            extra={
                "reference": float(altbc_ref["asymmetry"][0]),
                "model": float(altbc_model["asymmetry"][0]),
                "note": "mean |d1-d2| over near-linear bond pairs; the Peierls distortion measure",
                "angle_window_deg": params.altbc_angle_window_deg,
            },
        )
    else:
        add("L3b.altbc_reference.asymmetry", float(altbc_ref["asymmetry"][0]), "A", status=Status.OK)

    # -- S(q) --------------------------------------------------------------
    sq_ref = structure_factor_direct(reference, params, stride=sq_stride, weighting=weighting, q_max=sq_q_max)
    fsdp_ref = fsdp(sq_ref["q"], sq_ref["S"], q_min_accessible=sq_ref["q_min_accessible"])
    pp_ref = principal_peak(sq_ref["q"], sq_ref["S"])
    sq_model = fsdp_model = None
    if model_traj is not None:
        sq_model = structure_factor_direct(
            model_traj, params, stride=max(1, sq_stride // 2), weighting=weighting, q_max=sq_q_max
        )
        fsdp_model = fsdp(sq_model["q"], sq_model["S"], q_min_accessible=sq_model["q_min_accessible"])
        pp_model = principal_peak(sq_model["q"], sq_model["S"])
        for field_name, unit in (("q_fsdp", "1/A"), ("S_fsdp", "-"), ("fwhm", "1/A")):
            add(
                f"L3b.fsdp.{field_name}",
                fsdp_model[field_name],
                unit,
                status=Status.PROVISIONAL if fsdp_model["resolved"] else Status.SKIPPED,
                failure_reason=(
                    None
                    if fsdp_model["resolved"]
                    else "FSDP not resolved by this cell: peak on the search-window edge or below 2 pi / L"
                ),
                extra={
                    "reference": fsdp_ref[field_name],
                    "model": fsdp_model[field_name],
                    "difference": fsdp_model[field_name] - fsdp_ref[field_name],
                    "weighting": weighting,
                    "q_min_accessible_1_per_A": sq_ref["q_min_accessible"],
                    "search_window": list(fsdp_ref["window"]),
                    "method": sq_ref["method"],
                },
            )
        for field_name, unit in (("q_pp", "1/A"), ("S_pp", "-")):
            add(
                f"L3b.principal_peak.{field_name}",
                pp_model[field_name],
                unit,
                extra={
                    "reference": pp_ref[field_name],
                    "model": pp_model[field_name],
                    "difference": pp_model[field_name] - pp_ref[field_name],
                    "note": "principal S(q) peak (~2.2 1/A in a-GST225); far better resolved than the FSDP",
                },
            )
    else:
        for field_name, unit in (("q_fsdp", "1/A"), ("S_fsdp", "-"), ("fwhm", "1/A")):
            add(f"L3b.fsdp_reference.{field_name}", fsdp_ref[field_name], unit, status=Status.OK)
        for field_name, unit in (("q_pp", "1/A"), ("S_pp", "-")):
            add(f"L3b.principal_peak_reference.{field_name}", pp_ref[field_name], unit, status=Status.OK)

    if n_independent_references < 2:
        for item in results:
            if item.status is Status.PROVISIONAL:
                item.extra.setdefault(
                    "grading_note",
                    "provisional: one reference trajectory only, sigma_ref unknown -- no formal grade",
                )

    return L3bReport(
        params=params,
        rings_ref=rings_ref,
        rings_model=rings_model,
        altbc_ref=altbc_ref,
        altbc_model=altbc_model,
        sq_ref=sq_ref,
        sq_model=sq_model,
        fsdp_ref=fsdp_ref,
        fsdp_model=fsdp_model,
        results=results,
    )
