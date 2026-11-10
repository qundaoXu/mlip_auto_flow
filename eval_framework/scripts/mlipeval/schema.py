"""Minimal schema for structures, protocols, models and metric results.

The rules encoded here come from ``validation_methods_L0-L5.md`` section 3-4 and
``database_link.md`` section 3:

* every label set carries its own ``protocol_id``; labels from different
  protocols are never pooled into one error statistic;
* every frame carries exactly one :class:`DataRole`;
* a metric result is only gradeable when the reference spread is known, and a
  single reference trajectory can only ever produce a ``provisional`` result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA_VERSION = "mlipeval-0.1"


# --------------------------------------------------------------------------
# enumerations
# --------------------------------------------------------------------------
class DataRole(str, Enum):
    """Isolation classes from ``database_link.md`` 3.5."""

    TRAINING = "training"
    DIAGNOSTIC = "diagnostic"
    PROTECTED_AUDIT = "protected_audit"


class EvidenceTier(str, Enum):
    """Evidence tiers from ``validation_methods_L0-L5.md`` section 4."""

    A = "A"  # same protocol, same size AIMD, direct comparison
    B = "B"  # static DFT/DFPT under the same project protocol
    C = "C"  # external protocol / validated reference potential / indirect
    S = "S"  # reference-free self-consistency and stability check


class Status(str, Enum):
    OK = "ok"
    PROVISIONAL = "provisional"  # computed, but reference spread unknown
    FAILED = "failed"  # the model or the run failed the check
    SKIPPED = "skipped"  # prerequisites unmet (e.g. L1 gate closed)
    ERROR = "error"  # the analysis itself could not run


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    F = "F"
    UNGRADED = "ungraded"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, memoised by (size, mtime) in a sidecar.

    The MACE checkpoint is 80 MB and hashing it costs ~6 s.  A notebook that only
    reads cached results should not pay that on every kernel restart, and the
    memo is invalidated automatically if the file changes.
    """
    stat = path.stat()
    # The memo lives under eval/data, never next to the model: everything outside
    # eval/ is read-only reference material for this project.
    from .paths import data_path

    memo = data_path("cache", f"sha256_{path.name}.json")
    if memo.exists():
        try:
            cached = json.loads(memo.read_text())
            if cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
                return str(cached["sha256"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    digest = h.hexdigest()
    try:
        memo.write_text(json.dumps(
            {"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}))
    except OSError:
        pass  # if the cache directory is unwritable we simply recompute next time
    return digest


@dataclass(frozen=True)
class Protocol:
    """Reference or simulation protocol.

    ``protocol_id`` is the isolation key: two label sets with different
    ``protocol_id`` must not be merged into a single RMSE.
    """

    protocol_id: str
    kind: str  # "dft_static" | "aimd" | "mlip_md" | "mlip_static" | "relaxation"
    description: str = ""
    # electronic structure
    code: str | None = None
    code_version: str | None = None
    functional: str | None = None
    hubbard_u: str | None = None
    dispersion: str | None = None
    pseudopotentials: str | None = None
    encut_ev: float | None = None
    kpoints: str | None = None
    smearing: str | None = None
    ediff_ev: float | None = None
    # dynamics
    ensemble: str | None = None
    thermostat: str | None = None
    timestep_fs: float | None = None
    n_steps: int | None = None
    temperature_k: float | tuple[float, float] | None = None
    seed: int | None = None
    # post-processing knobs that change the number
    postprocessing: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ModelSpec:
    """Fully pinned description of the model under test."""

    name: str
    checkpoint: str
    checkpoint_sha256: str
    dtype: str
    device: str
    seed: int | None
    software: dict[str, str]
    native_protocol: str | None = None  # e.g. "PBE+U (MPtrj/sAlex)"
    dispersion: bool = False

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        name: str,
        *,
        dtype: str = "float64",
        device: str = "cpu",
        seed: int | None = 0,
        native_protocol: str | None = None,
        dispersion: bool = False,
    ) -> "ModelSpec":
        path = Path(path)
        return cls(
            name=name,
            checkpoint=str(path),
            checkpoint_sha256=_sha256(path),
            dtype=dtype,
            device=device,
            seed=seed,
            software=software_versions(),
            native_protocol=native_protocol,
            dispersion=dispersion,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class StructureRef:
    """Identity of an input configuration."""

    structure_id: str
    source: str
    n_atoms: int
    composition: str
    role: DataRole = DataRole.DIAGNOSTIC
    protocol_id: str | None = None
    frame_index: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class MetricResult:
    """One reportable number, traceable back to model / reference / inputs.

    Fields follow ``validation_methods_L0-L5.md`` section 3 ("结果至少包含").
    """

    metric_id: str
    level: str  # "L0".."L5"
    value: Any
    unit: str
    status: Status = Status.OK
    uncertainty: float | None = None
    evidence_tier: EvidenceTier | None = None
    grade: Grade = Grade.UNGRADED
    z_eff: float | None = None
    reference_protocol: str | None = None
    model_protocol: str | None = None
    structure_ids: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    outputs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def __str__(self) -> str:  # pragma: no cover - display helper
        val = self.value
        if isinstance(val, float):
            val = f"{val:.6g}"
        tier = self.evidence_tier.value if self.evidence_tier else "-"
        return (
            f"[{self.level}] {self.metric_id} = {val} {self.unit} "
            f"(status={self.status.value}, tier={tier}, grade={self.grade.value})"
        )


def as_spec(model) -> ModelSpec:
    """Accept a loaded ``Model`` or a bare :class:`ModelSpec`.

    Analysis that only needs the model's pinned identity -- protocol ids,
    provenance strings, cache keys -- should work from a ``ModelSpec`` alone, so
    a notebook can read precomputed results without loading the network (and
    without evaluating anything on the login node).
    """
    return getattr(model, "spec", model)


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------
def sigma_effective(sigma_ref: float | None, d_protocol: float | None) -> float | None:
    """``sigma_eff = max(sigma_ref, d_protocol / 2)``.

    Returns ``None`` when neither term is known, which forces a provisional
    result instead of a fabricated grade.
    """
    candidates = [s for s in (sigma_ref, None if d_protocol is None else d_protocol / 2.0) if s is not None]
    candidates = [c for c in candidates if np.isfinite(c) and c > 0]
    if not candidates:
        return None
    return float(max(candidates))


def grade_from_z(z_eff: float | None) -> Grade:
    if z_eff is None or not np.isfinite(z_eff):
        return Grade.UNGRADED
    if z_eff <= 1.0:
        return Grade.A
    if z_eff <= 2.0:
        return Grade.B
    if z_eff <= 4.0:
        return Grade.C
    return Grade.F


def grade_metric(
    result: MetricResult,
    distance: float,
    *,
    sigma_ref: float | None = None,
    d_protocol: float | None = None,
    n_independent_references: int = 1,
) -> MetricResult:
    """Attach ``z_eff``/grade, or mark the result ``provisional``.

    A single reference trajectory never yields a formal grade
    (``validation_methods_L0-L5.md`` section 4).
    """
    sigma = sigma_effective(sigma_ref, d_protocol)
    if sigma is None or n_independent_references < 2:
        result.status = Status.PROVISIONAL
        result.grade = Grade.UNGRADED
        result.z_eff = None
        reason = (
            "only one independent reference"
            if n_independent_references < 2
            else "neither sigma_ref nor d_protocol is known"
        )
        result.extra.setdefault("grading_note", f"provisional: {reason}")
        return result
    result.z_eff = float(abs(distance) / sigma)
    result.grade = grade_from_z(result.z_eff)
    result.uncertainty = sigma
    result.status = Status.OK
    return result


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "schema": SCHEMA_VERSION}
    for mod in ("numpy", "scipy", "ase", "torch", "mace", "matscipy", "sklearn"):
        try:
            module = __import__(mod)
            versions[mod] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependency
            versions[mod] = "absent"
    try:  # pragma: no cover - informational only
        versions["git"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        versions["git"] = "not-a-repo"
    return versions


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


class ResultSet:
    """Collect :class:`MetricResult` objects and dump a traceable report."""

    def __init__(self, level: str, model: ModelSpec | None = None, **context: Any) -> None:
        self.level = level
        self.model = model
        self.context = context
        self.results: list[MetricResult] = []

    def add(self, result: MetricResult) -> MetricResult:
        self.results.append(result)
        return result

    def extend(self, results: Iterable[MetricResult]) -> None:
        self.results.extend(results)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def get(self, metric_id: str) -> MetricResult | None:
        for result in self.results:
            if result.metric_id == metric_id:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "schema": SCHEMA_VERSION,
                "level": self.level,
                "model": self.model.to_dict() if self.model else None,
                "context": self.context,
                "software": software_versions(),
                "results": [r.to_dict() for r in self.results],
            }
        )

    def save(self, path: str | Path) -> Path:
        """Write the result file, stamping any known training leakage first.

        The leakage pass lives here rather than in each driver because this is
        the one call every driver makes: a model that trained on part of this
        workspace's reference data can then never produce a result file whose
        affected metrics are unmarked, including from drivers written later.
        For a model with nothing recorded (every universal potential) this is a
        no-op.
        """
        from . import leakage, models

        model_key = models.current_key()
        counts = leakage.apply_to_results(self.results, model_key)
        context = leakage.context_for(model_key)
        if context is not None:
            self.context["training_leakage_screen"] = context
        if counts["annotated"]:
            print(
                f"training-leakage screen: {counts['annotated']} metric(s) marked, "
                f"{counts['downgraded']} downgraded ok -> provisional"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for r in self.results:
            value = r.value
            if isinstance(value, (list, dict, np.ndarray)):
                value = "<array>"
            rows.append(
                {
                    "metric_id": r.metric_id,
                    "value": value,
                    "unit": r.unit,
                    "status": r.status.value,
                    "tier": r.evidence_tier.value if r.evidence_tier else "-",
                    "grade": r.grade.value,
                    "z_eff": r.z_eff,
                }
            )
        return rows

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.summary_rows())
