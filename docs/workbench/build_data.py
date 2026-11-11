#!/usr/bin/env python
"""Build the data embedded in the Workbench page (`const D = {...}` in index.html).

    MLIPEVAL_OUTPUT=/path/to/eval_framework python docs/workbench/build_data.py --inject docs/workbench/index.html
    MLIPEVAL_OUTPUT=/path/to/eval_framework python docs/workbench/build_data.py --out data.json

Reads the evaluation outputs (results/, data/models/, data/cache/, data/reference/)
under ``MLIPEVAL_OUTPUT``, the same tree the CLI writes, and the ranking code in
this repository.  Rerun it whenever result files change, so the published page
shows the current numbers.  ``--inject`` replaces only the data literal; the
application code in the page is left as it is.
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_framework" / "scripts"))
if not os.environ.get("MLIPEVAL_OUTPUT"):
    sys.exit("set MLIPEVAL_OUTPUT to the eval_framework output tree (results/, data/)")
EF = Path(os.environ["MLIPEVAL_OUTPUT"]).expanduser()
from mlipeval import models as registry  # noqa: E402
from mlipeval.ranking import build_ranking, LEVELS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=None, help="write the data as JSON")
ap.add_argument("--inject", default=None, help="replace the data literal in this page")
ARGS = ap.parse_args()
if not (ARGS.out or ARGS.inject):
    ap.error("give --out and/or --inject")
SHOW = ("mace_mpa0,nequip_oam_l,dpa4_pro,mattersim_5m,chgnet_030,mace_spec_gst,"
        "mace_spec_gst_ne,nequip_spec_gst,mace_gst_aging").split(",")
SPECIALISED = {"mace_spec_gst", "mace_spec_gst_ne", "nequip_spec_gst", "mace_gst_aging"}
STATES = ["canon_amorphous_300K", "canon_liquid_950K", "canon_liquid_1200K", "canon_liquid_3000K"]
PAIRS = ["Ge-Ge", "Ge-Sb", "Ge-Te", "Sb-Sb", "Sb-Te", "Te-Te"]
L5OBS = ["mean_cn_all", "mean_cn_Ge", "four_rings_per_atom", "tetrahedral_fraction", "rdf_first_peak_A"]


def r(x, sig=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0
    return float(f"{x:.{sig}g}")


def arr(a, sig=4):
    return [r(v, sig) for v in np.asarray(a).ravel()]


def load(p):
    return np.load(p, allow_pickle=True)


def jload(p):
    return json.loads(Path(p).read_text())


def artifact(key):
    return jload(EF / f"results/models/{key}/L0_audit.json")["model"]["software"]["artifact"]


def cache(prefix, key):
    hits = glob.glob(str(EF / f"data/cache/{prefix}__*{glob.escape(artifact(key))}__*.npz"))
    return load(hits[0]) if hits else None


# ---------------------------------------------------------------- reference labels (H5 audit, 120 frames)
from ase.io import read  # noqa: E402

h5 = read(EF / "data/reference/phase1_h5_relabel.extxyz", ":")
print("h5 info keys:", list(h5[0].info)[:12], "arrays:", list(h5[0].arrays))
ref_F = np.concatenate([a.arrays["REF_forces"] for a in h5])
frame_of_atom = np.concatenate([np.full(len(a), i) for i, a in enumerate(h5)])
pools = [str(a.info.get("subset") or a.info.get("pool") or a.info.get("config_type", "?")).split(":")[0] for a in h5]
print("pools:", sorted(set(pools)))
rng = np.random.default_rng(0)
pick = rng.choice(ref_F.size, 700, replace=False)

# ---------------------------------------------------------------- ranking
rk = build_ranking(SHOW)
D = {
    "order": SHOW,
    "levels": list(LEVELS),
    "weights": dict(rk.level_weights),
    "overall": {k: r(v) for k, v in rk.overall.items()},
    "level_scores": {lv: {k: r(v) for k, v in rk.level_scores[lv].items()} for lv in LEVELS},
    "gate_open": {k: bool(v) for k, v in rk.gate_open.items()},
    "indicators": [
        {"level": row.indicator.level, "key": row.indicator.key, "unit": row.indicator.unit,
         "w": row.indicator.weight, "floor": r(row.indicator.floor), "dir": row.indicator.direction,
         "note": row.indicator.note,
         "raw": {k: r(v) for k, v in row.raw.items()}, "score": {k: r(v, 3) for k, v in row.score.items()}}
        for row in rk.rows
    ],
    "models": {}, "ref": {"l3": {}, "l4": {}, "l5": {}},
}

al = jload(EF / "results/ranking_al_ladder_clean.json")
D["al"] = {
    "models": al["models"],
    "labels": {k: registry.get(k).label for k in al["models"]},
    "overall": {k: r(v) for k, v in al["overall"].items()},
    "level_scores": {lv: {k: r(v) for k, v in al["level_scores"][lv].items()} for lv in al["level_scores"]},
}


def metric_rows(key):
    rows = []
    for f in sorted(glob.glob(str(EF / f"results/models/{key}/L*.json"))):
        name = Path(f).stem
        if name.startswith("L3_uncertainty"):
            continue
        for m in jload(f)["results"]:
            v = m["value"]
            if isinstance(v, (dict, list)):
                continue
            rows.append([m["metric_id"], r(v) if not isinstance(v, str) else v, m["unit"], m["status"],
                         m["grade"], r(m.get("z_eff"), 3), r(m.get("uncertainty"), 3), m["evidence_tier"],
                         m.get("failure_reason")])
    return rows


for key in SHOW:
    e = registry.get(key)
    md = EF / f"data/models/{key}"
    M = {"label": e.label, "backend": e.backend, "protocol": e.native_protocol,
         "group": "specialised" if key in SPECIALISED else "universal"}
    l1 = jload(glob.glob(str(EF / f"results/models/{key}/L1_*.json"))[0])
    g = l1["context"].get("gate", {})
    M["gate"] = {"pass": bool(g.get("pass")), "n": g.get("n_checks"), "failed": g.get("failed_metrics", []),
                 "decision": g.get("decision")}
    M["metrics"] = metric_rows(key)

    # L0: force parity on the audit set
    c = cache("L0_h5_120", key)
    if c is not None:
        F = c["forces"]
        M["l0"] = {"parity": [arr(ref_F.ravel()[pick], 3), arr(F.ravel()[pick], 3)]}
        err = np.linalg.norm(F - ref_F, axis=1)
        M["l0"]["frame_err"] = [r(np.sqrt(np.mean(err[frame_of_atom == i] ** 2)), 3) for i in range(len(h5))]
        M["l0"]["frame_max"] = [r(err[frame_of_atom == i].max(), 3) for i in range(len(h5))]

    # L1: pair curves, NVE, lattice scan
    p = cache("pec2", key)
    M["l1"] = {}
    if p is not None:
        sl = slice(0, None, 2)
        en = np.array(p["energy"], dtype=float)[:, sl]
        en = np.where(np.isfinite(en), np.clip(en, -1e3, 1e3), np.nan)
        M["l1"]["pec"] = {"r": arr(p["r"][sl], 3), "pairs": [str(x) for x in p["pairs"]],
                          "e": [arr(row, 4) for row in en],
                          "mono": [bool(x) for x in p["monotonic_repulsive"]]}
    f = md / "l1_nve_dt3.npz"
    if f.exists():
        z = load(f)
        n = len(z["symbols"])
        etot = (z["potential_energy"] + z["kinetic_energy"]) / n * 1000
        M["l1"]["nve"] = {"t": arr(z["time_fs"] / 1000, 4), "e": arr(etot - etot[0], 4)}
    f = md / "l1_lattice_scan.npz"
    if f.exists():
        z = load(f)
        M["l1"]["scan"] = {"s": arr(z["scales"], 4), "e": arr(z["energy_per_atom"], 5),
                           "d": arr(z["min_distance"], 4)}

    # L2: equations of state
    M["l2"] = {}
    for ph in ("hex", "cubic"):
        fd, fm = md / f"l2_eos_{ph}_dft.npz", md / f"l2_eos_{ph}_mlip.npz"
        if fd.exists() and fm.exists():
            out = {}
            for tag, ff in (("dft", fd), ("mlip", fm)):
                z = load(ff)
                n = int(z["n_atoms"])
                en = np.array(z["energies"], float)
                if np.abs(en).max() > 30:
                    en = en / n
                v = np.array(z["volumes"], float)
                if v.max() > 200:
                    v = v / n
                out[tag] = {"v": arr(v, 5), "e": arr((en - en.min()) * 1000, 4)}
            M["l2"][ph] = out

    # L3: structure curves per state point
    M["l3"] = {}
    for st in STATES:
        f = md / f"l3_{st}.npz"
        if not f.exists():
            continue
        z = load(f)
        s2 = slice(0, None, 2)

        def side(tag):
            return {"rdf": {pq: arr(z[f"rdf_{tag}_{pq}"][s2], 3) for pq in PAIRS},
                    "adf": {sp: arr(z[f"adf_{tag}_{sp}"][s2], 3) for sp in ("all", "Ge", "Sb", "Te")},
                    "rings": arr(z[f"rings_{tag}"], 3), "sq": arr(z[f"sq_{tag}"], 3),
                    "cn": {sp: arr(z[f"cn_{tag}_{sp}"], 3) for sp in ("all", "Ge", "Sb", "Te")}}
        M["l3"][st] = side("model")
        if st not in D["ref"]["l3"]:
            ref = side("ref")
            ref.update({"r": arr(z["r"][s2], 4), "angle": arr(z["angle"][s2], 4), "q": arr(z["q"], 4),
                        "ring_sizes": [int(x) for x in z["ring_sizes"]], "cn_values": [int(x) for x in z["cn_values"]],
                        "cutoffs": json.loads(str(z["bond_cutoffs"]))})
            D["ref"]["l3"][st] = ref

    # L4: MSD
    M["l4"] = {}
    for st in STATES[1:]:
        f = md / f"l4_{st}.npz"
        if not f.exists():
            continue
        z = load(f)
        idx = np.unique(np.round(np.geomspace(1, len(z["time_ref"]) - 1, 70)).astype(int))
        M["l4"][st] = {"t": arr(z["time_model"][idx] / 1000, 4),
                       "msd": {sp: arr(z[f"msd_model_{sp}"][idx], 4) for sp in ("Ge", "Sb", "Te", "all")}}
        if st not in D["ref"]["l4"]:
            D["ref"]["l4"][st] = {"t": arr(z["time_ref"][idx] / 1000, 4),
                                  "msd": {sp: arr(z[f"msd_ref_{sp}"][idx], 4) for sp in ("Ge", "Sb", "Te", "all")}}

    # L5: pathway
    M["l5"] = {}
    for prog in ("canon_quench", "canon_heating"):
        f = md / f"l5_{prog}.npz"
        if not f.exists():
            continue
        z = load(f)
        M["l5"][prog] = {"T": arr(z["model_nominal_temperature_K"], 4),
                         "obs": {o: arr(z[f"model_{o}"], 4) for o in L5OBS},
                         "div": {k: (r(v.get("progress")) if isinstance(v, dict) else None)
                                 for k, v in json.loads(str(z["divergences"])).items() if k in L5OBS}}
        if prog not in D["ref"]["l5"]:
            D["ref"]["l5"][prog] = {"T": arr(z["ref_nominal_temperature_K"], 4), "p": arr(z["ref_progress"], 4),
                                    "obs": {o: arr(z[f"ref_{o}"], 4) for o in L5OBS},
                                    "band": {o: arr(z[f"band_{o}"], 3) for o in L5OBS}}

    # candidates for the next labelling round
    M["cands"] = []
    for mf in sorted(glob.glob(str(EF / f"results/models/{key}/candidates/*manifest.json"))):
        m = jload(mf)
        for cnd in m.get("candidates", []):
            M["cands"].append({"id": cnd["candidate_id"], "metric": cnd["source_metric"], "reason": cnd["reason"],
                               "protocol": cnd["suggested_protocol"], "novelty": r(cnd.get("novelty"), 3),
                               "cluster": cnd.get("cluster_size"), "from": m.get("n_before_dedup")})
    D["models"][key] = M

D["pools"] = pools

# ---------------------------------------------------------------- trajectory lab + sampling map (MACE-MPA-0 MLIP-MD)
TR = {"canon_quench": "quench 1200→300 K", "canon_liquid_950K": "liquid 950 K", "canon_amorphous_300K": "amorphous 300 K",
      "canon_liquid_1200K": "liquid 1200 K", "canon_liquid_3000K": "liquid 3000 K"}
D["traj"] = {}
fps_rows, fps_fp = [], []


def fingerprint(pos, cell, sym):
    frac = pos @ np.linalg.inv(cell)
    d = frac[:, None, :] - frac[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ cell, axis=-1)
    out = []
    masks = {s: sym == s for s in ("Ge", "Sb", "Te")}
    for a, b in (("Ge", "Te"), ("Sb", "Te"), ("Te", "Te"), ("Ge", "Ge")):
        dd = dist[np.ix_(masks[a], masks[b])].ravel()
        h, _ = np.histogram(dd[dd > 0.1], bins=30, range=(2.0, 6.5))
        out.append(h / max(h.sum(), 1))
    return np.concatenate(out)


for st, lab in TR.items():
    z = load(EF / f"data/models/mace_mpa0/md_{st}__lammps.npz")
    n = len(z["symbols"])
    k = max(1, len(z["time_fs"]) // 400)
    sl = slice(0, None, k)
    D["traj"][st] = {"label": lab, "t": arr(z["time_fs"][sl] / 1000, 4), "pe": arr(z["potential_energy"][sl] / n, 5),
                     "T": arr(z["temperature"][sl], 4), "dmin": arr(z["min_distance"][sl], 4),
                     "n_frames": int(len(z["time_fs"])), "n_atoms": n}
    sym = np.array([str(s) for s in z["symbols"]])
    for i in np.linspace(0, len(z["time_fs"]) - 1, 90).astype(int):
        fps_fp.append(fingerprint(z["positions"][i], z["cells"][i], sym))
        fps_rows.append([st, int(i), r(z["time_fs"][i] / 1000, 4), r(z["potential_energy"][i] / n, 5), r(z["temperature"][i], 4)])
X = np.array(fps_fp)
X = X - X.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
pcs = U[:, :8] * S[:8]
D["sample"] = {"rows": fps_rows, "pc": [arr(row, 3) for row in pcs],
               "var": arr(S[:8] ** 2 / (S ** 2).sum(), 3), "states": TR}

# ---------------------------------------------------------------- committee disagreement (AL gen6–9) on the 120 labelled frames
COMM = ["al_gen6", "al_gen7", "al_gen8", "al_gen9"]
Fs = np.array([cache("L0_h5_120", k)["forces"] for k in COMM])
Es = np.array([cache("L0_h5_120", k)["energies"] / np.array([len(a) for a in h5]) for k in COMM])
sF = np.linalg.norm(Fs.std(0), axis=1)
D["unc"] = {"committee": COMM,
            "sE": arr(Es.std(0) * 1000, 3),
            "sFrms": [r(np.sqrt(np.mean(sF[frame_of_atom == i] ** 2)), 3) for i in range(len(h5))],
            "sFmax": [r(sF[frame_of_atom == i].max(), 3) for i in range(len(h5))]}

payload = json.dumps(D, separators=(",", ":"), ensure_ascii=False)
if ARGS.out:
    Path(ARGS.out).write_text(payload)
    print("wrote", ARGS.out, len(payload) // 1024, "KB")
if ARGS.inject:
    page = Path(ARGS.inject)
    html = page.read_text(encoding="utf-8")
    marker = "const D = "
    start = html.index(marker) + len(marker)
    _old, length = json.JSONDecoder().raw_decode(html[start:])
    page.write_text(html[:start] + payload + html[start + length:], encoding="utf-8")
    print("updated", page, len(payload) // 1024, "KB of data")
for k, v in D["models"].items():
    print(k, {kk: (len(vv) if hasattr(vv, "__len__") else vv) for kk, vv in v.items()
              if kk in ("metrics", "l0", "l1", "l2", "l3", "l4", "l5", "cands")})
