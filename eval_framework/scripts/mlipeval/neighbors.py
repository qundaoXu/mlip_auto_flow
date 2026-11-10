"""Neighbour lists and the structural primitives shared by L1, L3 and L5.

All L3 numbers depend on post-processing knobs (bin width, bond cutoff, ring
algorithm).  Those knobs live in :class:`StructureParams` so a single frozen
object can be handed to both the AIMD and the MLIP side, which is what
``validation_methods_L0-L5.md`` L3a requires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations_with_replacement
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

try:  # matscipy is ~10x faster and is installed in the mace env
    from matscipy.neighbours import neighbour_list as _ms_neighbour_list

    _HAVE_MATSCIPY = True
except Exception:  # pragma: no cover - fallback
    from ase.neighborlist import neighbor_list as _ase_neighbour_list

    _HAVE_MATSCIPY = False


def neighbour_list(atoms: Atoms, cutoff: float, quantities: str = "ijdD"):
    """``matscipy``-backed full neighbour list (both i->j and j->i present)."""
    if _HAVE_MATSCIPY:
        return _ms_neighbour_list(quantities, atoms, cutoff)
    return _ase_neighbour_list(quantities, atoms, cutoff)  # pragma: no cover


# --------------------------------------------------------------------------
# frozen post-processing parameters
# --------------------------------------------------------------------------
@dataclass
class StructureParams:
    """Post-processing knobs, frozen once and reused for every trajectory.

    ``bond_cutoffs`` maps an unordered pair key ``"Ge-Te"`` to a distance in A.
    They are meant to be *derived from the AIMD partial RDF first minimum* and
    then applied unchanged to the MLIP trajectory.
    """

    r_max: float = 10.0
    n_bins: int = 400
    bond_cutoffs: dict[str, float] = field(default_factory=dict)
    angle_bins: int = 180
    max_ring_size: int = 12
    ring_algorithm: str = "king-shortest-path"
    altbc_r_max: float = 4.2
    altbc_angle_window_deg: float = 25.0
    altbc_bins: int = 120
    sq_q_min: float = 0.5
    sq_q_max: float = 8.0
    sq_n_q: int = 300
    provenance: str = "unset"
    cutoff_diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def dr(self) -> float:
        return self.r_max / self.n_bins

    @property
    def r_grid(self) -> np.ndarray:
        edges = np.linspace(0.0, self.r_max, self.n_bins + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def cutoff(self, a: str, b: str) -> float:
        key = pair_key(a, b)
        if key not in self.bond_cutoffs:
            raise KeyError(f"no frozen bond cutoff for {key}; derive it from the AIMD RDF first")
        return self.bond_cutoffs[key]

    @property
    def max_cutoff(self) -> float:
        if not self.bond_cutoffs:
            raise ValueError("bond_cutoffs is empty")
        return max(self.bond_cutoffs.values())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def pair_key(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


def all_pairs(species: Sequence[str]) -> list[tuple[str, str]]:
    return [tuple(sorted(p)) for p in combinations_with_replacement(sorted(species), 2)]


# --------------------------------------------------------------------------
# radial distribution functions
# --------------------------------------------------------------------------
def partial_rdf_frame(
    atoms: Atoms,
    params: StructureParams,
    species: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Unnormalised pair-distance histograms for one frame.

    Returns raw counts keyed by ``"Ge-Te"``; normalisation happens once over the
    whole trajectory so that a short trajectory does not bias the average.
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    species = list(species) if species is not None else sorted(set(symbols.tolist()))
    i, j, d = neighbour_list(atoms, params.r_max, "ijd")
    edges = np.linspace(0.0, params.r_max, params.n_bins + 1)
    counts: dict[str, np.ndarray] = {}
    si, sj = symbols[i], symbols[j]
    for a, b in all_pairs(species):
        if a == b:
            mask = (si == a) & (sj == b)
        else:
            mask = ((si == a) & (sj == b)) | ((si == b) & (sj == a))
        counts[pair_key(a, b)], _ = np.histogram(d[mask], bins=edges)
    return counts


def partial_rdf(
    traj,
    params: StructureParams,
    *,
    stride: int = 1,
    species: Sequence[str] | None = None,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Trajectory-averaged partial RDFs ``g_ab(r)``.

    Normalisation: ``g_ab(r) = n_ab(r) / (4 pi r^2 dr * rho_b * N_a)`` with
    ``n_ab`` counting neighbours of type ``b`` around atoms of type ``a``.  For
    ``a == b`` the ordered-pair count is used consistently, so ``g_aa -> 1`` at
    large ``r`` like the unlike-species partials.
    """
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "partial_rdf",
            {"traj": trajectory_identity(traj), "params": params.to_dict(),
             "stride": stride, "species": list(species) if species else None},
            lambda: partial_rdf(traj, params, stride=stride, species=species, cache=False),
        )

    symbols = np.asarray(traj.symbols)
    species = list(species) if species is not None else traj.species
    counts_sum: dict[str, np.ndarray] = {}
    n_frames = 0
    volume_sum = 0.0
    for k in range(0, len(traj), stride):
        atoms = traj.atoms(k)
        frame_counts = partial_rdf_frame(atoms, params, species)
        for key, c in frame_counts.items():
            counts_sum[key] = counts_sum.get(key, np.zeros_like(c, dtype=float)) + c
        volume_sum += atoms.get_volume()
        n_frames += 1

    volume = volume_sum / n_frames
    r = params.r_grid
    shell = 4.0 * np.pi * r**2 * params.dr
    n_of = {s: int((symbols == s).sum()) for s in species}

    out: dict[str, np.ndarray] = {"r": r}
    for a, b in all_pairs(species):
        key = pair_key(a, b)
        n_a, n_b = n_of[a], n_of[b]
        rho_b = n_b / volume
        # counts_sum holds unordered pair counts doubled for a != b as well,
        # because the full neighbour list contains i->j and j->i.
        mean_counts = counts_sum[key] / n_frames
        norm = shell * rho_b * n_a
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(norm > 0, mean_counts / norm, 0.0)
        out[key] = g
    out["_meta"] = {"n_frames": n_frames, "volume": volume, "counts_of": n_of}
    return out


def total_rdf(rdf: Mapping[str, np.ndarray], symbols: Sequence[str]) -> np.ndarray:
    """Number-weighted total RDF ``sum_ab c_a c_b g_ab``."""
    symbols = np.asarray(symbols)
    species = sorted(set(symbols.tolist()))
    n = len(symbols)
    total = np.zeros_like(rdf["r"])
    for a, b in all_pairs(species):
        c_a = (symbols == a).sum() / n
        c_b = (symbols == b).sum() / n
        weight = c_a * c_b * (1 if a == b else 2)
        total += weight * rdf[pair_key(a, b)]
    return total


def total_rdf_fingerprint(atoms: Atoms, r_max: float = 6.0, bins: int = 60) -> np.ndarray:
    """Coarse normalised RDF vector used only for duplicate screening."""
    _, _, d = neighbour_list(atoms, r_max, "ijd")
    hist, _ = np.histogram(d, bins=bins, range=(0.0, r_max), density=False)
    hist = hist.astype(float) / max(len(atoms), 1)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


# --------------------------------------------------------------------------
# peak / minimum extraction
# --------------------------------------------------------------------------
def first_peak(r: np.ndarray, g: np.ndarray, r_lo: float = 1.5, r_hi: float = 4.5) -> dict[str, float]:
    """Position, height and FWHM of the first RDF peak inside ``[r_lo, r_hi]``."""
    win = (r >= r_lo) & (r <= r_hi)
    if not win.any() or not np.isfinite(g[win]).any():
        return {"r_peak": np.nan, "g_peak": np.nan, "fwhm": np.nan}
    rw, gw = r[win], g[win]
    k = int(np.argmax(gw))
    g_peak, r_peak = float(gw[k]), float(rw[k])
    half = g_peak / 2.0
    left = rw[:k][gw[:k] <= half]
    right = rw[k:][gw[k:] <= half]
    fwhm = float(right[0] - left[-1]) if left.size and right.size else np.nan
    return {"r_peak": r_peak, "g_peak": g_peak, "fwhm": fwhm}


def _smooth(y: np.ndarray, window_bins: int) -> np.ndarray:
    """Savitzky-Golay smoothing, degraded to a moving average for short arrays."""
    window = max(5, window_bins | 1)
    if y.size <= window:
        return y
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(y, window_length=window, polyorder=3, mode="interp")
    except Exception:  # pragma: no cover
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode="same")


def first_minimum(
    r: np.ndarray,
    g: np.ndarray,
    *,
    r_lo: float = 1.8,
    r_hi: float = 4.5,
    r_peak_max: float = 3.6,
    smooth_A: float = 0.25,
    min_depth: float = 0.25,
    peak_rule: str = "max",
) -> dict[str, float]:
    """First minimum of ``g(r)`` after its first peak -- the bond cutoff.

    The raw histogram is too noisy for a naive "first index that stops
    decreasing" rule, especially for the dilute like-species partials, so the
    curve is Savitzky-Golay smoothed over ``smooth_A`` and the minimum is taken
    as the lowest smoothed value between the peak and ``r_hi``.

    ``depth = (g_peak - g_min) / g_peak`` measures how well defined the minimum
    is.  At high temperature the partials wash out and ``depth`` falls; when it
    drops below ``min_depth`` the result is flagged ``well_defined=False`` so the
    caller can fall back rather than freeze a meaningless cutoff.

    ``peak_rule`` says which maximum is "the first peak".  ``"max"`` (the
    original rule, kept for the legacy state points) takes the highest point in
    ``[r_lo, r_peak_max]``.  ``"first"`` takes the first prominent local maximum
    (prominence >= 10% of that window's maximum), falling back to ``"max"`` when
    there is none.  The difference matters for a weak like-species partial: in
    the canonical 300 K amorphous state the Ge-Ge first peak (g = 0.88 at 2.70 A)
    is lower than the start of the second shell at the window edge (g = 1.20 at
    3.59 A), so ``"max"`` lands on the second shell and misses the real minimum
    near 3.15 A.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    dr = float(r[1] - r[0])
    window = int(round(smooth_A / dr))
    gs = _smooth(g, window)

    peak_win = (r >= r_lo) & (r <= r_peak_max)
    if peak_win.sum() < 3:
        return {"r_min": float("nan"), "g_min": float("nan"), "r_peak": float("nan"), "depth": float("nan"), "well_defined": False}
    k_peak = int(np.flatnonzero(peak_win)[np.argmax(gs[peak_win])])
    if peak_rule == "first":
        try:
            from scipy.signal import find_peaks

            window_idx = np.flatnonzero(peak_win)
            maxima, _ = find_peaks(gs[window_idx], prominence=0.1 * float(gs[window_idx].max()))
            if maxima.size:
                k_peak = int(window_idx[maxima[0]])
        except Exception:  # pragma: no cover
            pass
    elif peak_rule != "max":
        raise ValueError(f"unknown peak_rule '{peak_rule}'")
    g_peak = float(gs[k_peak])

    search = (r > r[k_peak] + smooth_A) & (r <= r_hi)
    if search.sum() < 3 or g_peak <= 0:
        return {"r_min": float("nan"), "g_min": float("nan"), "r_peak": float(r[k_peak]), "depth": float("nan"), "well_defined": False}
    idx = np.flatnonzero(search)
    # Prefer the FIRST prominent local minimum after the peak.  Taking the global
    # argmin instead lets a hot, washed-out partial pick up a shallow wiggle far
    # out in the flat tail, which would freeze a much too generous bond cutoff.
    k_min = int(idx[np.argmin(gs[idx])])
    try:
        from scipy.signal import find_peaks

        prominence = 0.05 * g_peak
        troughs, _ = find_peaks(-gs[idx], prominence=prominence)
        if troughs.size:
            k_min = int(idx[troughs[0]])
    except Exception:  # pragma: no cover
        pass
    g_min = float(gs[k_min])
    depth = (g_peak - g_min) / g_peak
    # a minimum sitting on the last sampled point is a monotone tail, not a minimum
    on_edge = k_min >= idx[-1] - 1
    return {
        "r_min": float(r[k_min]),
        "g_min": g_min,
        "r_peak": float(r[k_peak]),
        "g_peak": g_peak,
        "depth": float(depth),
        "well_defined": bool(depth >= min_depth and not on_edge),
    }


def derive_bond_cutoffs(
    rdf: Mapping[str, np.ndarray],
    species: Sequence[str],
    symbols: Sequence[str] | None = None,
    *,
    r_lo: float = 1.8,
    r_hi: float = 4.5,
    r_peak_max: float = 3.6,
    min_depth: float = 0.25,
    peak_rule: str = "max",
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Per-pair bond cutoffs from the RDF first minima, plus their provenance.

    A partial whose minimum is not well defined (typically a dilute like-species
    pair, or any pair in a hot liquid) falls back to the first minimum of the
    **total** RDF, and the fallback is recorded in the returned diagnostics so
    the notebook and the result JSON both show which cutoffs were measured and
    which were inherited.
    """
    r = rdf["r"]
    diagnostics: dict[str, dict[str, float]] = {}
    fallback = float("nan")
    if symbols is not None:
        total = first_minimum(r, total_rdf(rdf, symbols), r_lo=r_lo, r_hi=r_hi, r_peak_max=r_peak_max,
                              min_depth=min_depth, peak_rule=peak_rule)
        diagnostics["_total"] = total
        fallback = total["r_min"]

    cutoffs: dict[str, float] = {}
    for a, b in all_pairs(species):
        key = pair_key(a, b)
        info = first_minimum(r, rdf[key], r_lo=r_lo, r_hi=r_hi, r_peak_max=r_peak_max,
                             min_depth=min_depth, peak_rule=peak_rule)
        if info["well_defined"]:
            cutoffs[key] = info["r_min"]
            info["source"] = "partial RDF first minimum"
        elif np.isfinite(fallback):
            cutoffs[key] = fallback
            info["source"] = "fallback: total RDF first minimum (partial minimum ill-defined)"
        else:
            cutoffs[key] = info["r_min"]
            info["source"] = "partial RDF first minimum (ill-defined, no total-RDF fallback available)"
        diagnostics[key] = info
    return cutoffs, diagnostics


# --------------------------------------------------------------------------
# bonds, coordination, angles
# --------------------------------------------------------------------------
def cutoff_matrix(params: StructureParams, species: Sequence[str]) -> tuple[dict[str, int], np.ndarray]:
    """``(species -> row index, cutoff matrix)`` for vectorised bond selection."""
    order = {s: k for k, s in enumerate(species)}
    matrix = np.zeros((len(order), len(order)), dtype=float)
    for a, ia in order.items():
        for b, ib in order.items():
            matrix[ia, ib] = params.cutoff(a, b)
    return order, matrix


def bond_list(atoms: Atoms, params: StructureParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bonded ``(i, j, d, D)`` under the frozen per-pair cutoffs.

    The per-pair cutoff is applied through a species x species lookup rather
    than a Python loop over neighbour pairs: this function is called once per
    frame by CN, ADF, ring and ALTBC analysis, so the loop version dominated the
    whole of L3.
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    species = sorted(set(symbols.tolist()))
    order, matrix = cutoff_matrix(params, species)
    codes = np.array([order[s] for s in symbols], dtype=np.intp)
    i, j, d, D = neighbour_list(atoms, params.max_cutoff, "ijdD")
    mask = d <= matrix[codes[i], codes[j]]
    return i[mask], j[mask], d[mask], D[mask]


def coordination_numbers(atoms: Atoms, params: StructureParams) -> np.ndarray:
    """Per-atom coordination number under the frozen cutoffs."""
    i, _, _, _ = bond_list(atoms, params)
    return np.bincount(i, minlength=len(atoms))


def coordination_distribution(
    traj,
    params: StructureParams,
    *,
    stride: int = 1,
    max_cn: int = 12,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Full CN distribution per species (not just the mean)."""
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "coordination_distribution",
            {"traj": trajectory_identity(traj), "params": params.to_dict(),
             "stride": stride, "max_cn": max_cn},
            lambda: coordination_distribution(traj, params, stride=stride,
                                              max_cn=max_cn, cache=False),
        )

    symbols = np.asarray(traj.symbols)
    species = traj.species
    hist = {s: np.zeros(max_cn + 1, dtype=float) for s in species}
    hist["all"] = np.zeros(max_cn + 1, dtype=float)
    n_frames = 0
    for k in range(0, len(traj), stride):
        cn = coordination_numbers(traj.atoms(k), params)
        cn = np.clip(cn, 0, max_cn)
        for s in species:
            hist[s] += np.bincount(cn[symbols == s], minlength=max_cn + 1)[: max_cn + 1]
        hist["all"] += np.bincount(cn, minlength=max_cn + 1)[: max_cn + 1]
        n_frames += 1
    out = {"cn_values": np.arange(max_cn + 1)}
    for key, h in hist.items():
        total = h.sum()
        out[key] = h / total if total > 0 else h
    out["_meta"] = {"n_frames": n_frames}
    return out


def homopolar_fraction(traj, params: StructureParams, *, stride: int = 1,
                       cache: bool = True) -> dict[str, float]:
    """Fraction of bonds that are homopolar (``Ge-Ge``, ``Sb-Sb``, ``Te-Te``)."""
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "homopolar_fraction",
            {"traj": trajectory_identity(traj), "params": params.to_dict(), "stride": stride},
            lambda: homopolar_fraction(traj, params, stride=stride, cache=False),
        )

    symbols = np.asarray(traj.symbols)
    total = 0.0
    homo: dict[str, float] = {}
    for k in range(0, len(traj), stride):
        i, j, _, _ = bond_list(traj.atoms(k), params)
        si, sj = symbols[i], symbols[j]
        keys = np.char.add(np.char.add(si, "-"), sj)
        same = si == sj
        total += i.size
        for s in traj.species:
            homo[s] = homo.get(s, 0.0) + float(((si == s) & same).sum())
    out = {s: (v / total if total else np.nan) for s, v in homo.items()}
    out["total"] = float(sum(homo.values()) / total) if total else np.nan
    return out


def bond_angle_distribution(
    traj,
    params: StructureParams,
    *,
    centre: str | None = None,
    stride: int = 1,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Species-resolved bond-angle distribution, normalised to unit area."""
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "bond_angle_distribution",
            {"traj": trajectory_identity(traj), "params": params.to_dict(),
             "centre": centre, "stride": stride},
            lambda: bond_angle_distribution(traj, params, centre=centre,
                                            stride=stride, cache=False),
        )

    symbols = np.asarray(traj.symbols)
    edges = np.linspace(0.0, 180.0, params.angle_bins + 1)
    centres_wanted = traj.species if centre is None else [centre]
    hist = {s: np.zeros(params.angle_bins, dtype=float) for s in centres_wanted}
    hist["all"] = np.zeros(params.angle_bins, dtype=float)

    for k in range(0, len(traj), stride):
        atoms = traj.atoms(k)
        i, j, d, D = bond_list(atoms, params)
        order = np.argsort(i, kind="stable")
        i, j, d, D = i[order], j[order], d[order], D[order]
        starts = np.searchsorted(i, np.arange(len(atoms)), side="left")
        ends = np.searchsorted(i, np.arange(len(atoms)), side="right")
        for a in range(len(atoms)):
            s0, s1 = starts[a], ends[a]
            if s1 - s0 < 2:
                continue
            vec = D[s0:s1] / d[s0:s1, None]
            cos = np.clip(vec @ vec.T, -1.0, 1.0)
            iu = np.triu_indices(vec.shape[0], k=1)
            angles = np.degrees(np.arccos(cos[iu]))
            h, _ = np.histogram(angles, bins=edges)
            sym = symbols[a]
            if sym in hist:
                hist[sym] += h
            hist["all"] += h

    out = {"angle": 0.5 * (edges[:-1] + edges[1:])}
    width = edges[1] - edges[0]
    for key, h in hist.items():
        total = h.sum() * width
        out[key] = h / total if total > 0 else h
    return out


def tetrahedral_order(
    traj,
    params: StructureParams,
    *,
    species: str = "Ge",
    stride: int = 1,
    n_neighbours: int = 4,
    cache: bool = True,
) -> np.ndarray:
    """Chau-Hardwick ``q`` for the four nearest bonded neighbours.

    ``q = 1 - 3/8 * sum_{i<j} (cos(theta_ij) + 1/3)^2``; ``q = 1`` is ideal
    tetrahedral, ``q = 0`` is the ideal-gas value.  Atoms with fewer than
    ``n_neighbours`` bonded neighbours are skipped and reported via the count.
    """
    if cache:
        from .memo import cached_arrays, trajectory_identity

        return cached_arrays(
            "tetrahedral_order",
            {"traj": trajectory_identity(traj), "params": params.to_dict(),
             "species": species, "stride": stride, "n_neighbours": n_neighbours},
            lambda: tetrahedral_order(traj, params, species=species, stride=stride,
                                      n_neighbours=n_neighbours, cache=False),
        )

    symbols = np.asarray(traj.symbols)
    targets = np.flatnonzero(symbols == species)
    values: list[float] = []
    for k in range(0, len(traj), stride):
        atoms = traj.atoms(k)
        i, j, d, D = bond_list(atoms, params)
        order = np.argsort(i, kind="stable")
        i, j, d, D = i[order], j[order], d[order], D[order]
        starts = np.searchsorted(i, targets, side="left")
        ends = np.searchsorted(i, targets, side="right")
        for s0, s1 in zip(starts, ends):
            if s1 - s0 < n_neighbours:
                continue
            local_d, local_D = d[s0:s1], D[s0:s1]
            sel = np.argsort(local_d)[:n_neighbours]
            vec = local_D[sel] / local_d[sel, None]
            cos = np.clip(vec @ vec.T, -1.0, 1.0)
            iu = np.triu_indices(n_neighbours, k=1)
            values.append(1.0 - 0.375 * np.sum((cos[iu] + 1.0 / 3.0) ** 2))
    return np.asarray(values)
