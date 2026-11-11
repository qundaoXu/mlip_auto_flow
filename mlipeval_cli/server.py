"""Local Workbench backend: evaluation packs in, rescored L0–L2 results out.

    mlipeval serve [--host 127.0.0.1] [--port 8765] [--store DIR] [--reference R]

A small FastAPI app around :mod:`mlipeval.pack` and :mod:`mlipeval.rescore`.
It never loads a model: the HPC job produces the pack (``mlipeval pack model``),
this server verifies it against the frozen reference pack and rescores it on
the CPU.  The Workbench page (``docs/workbench``) is served at ``/workbench/``.

Endpoints (JSON unless noted)::

    GET    /api/health
    GET    /api/reference                     reference pack summary
    GET    /api/packs                         uploaded packs
    POST   /api/packs                         multipart upload of a .mlippack
    GET    /api/packs/{id}                    manifest summary + verification
    POST   /api/packs/{id}/rescore?levels=…   rescore, store, return summary + comparison
    GET    /api/packs/{id}/results            rescored result files
    GET    /api/packs/{id}/results/{stem}     one result file (L0_audit, L0_diagnostic, L1, L2)
    GET    /api/packs/{id}/workbench          the page's per-model object for L0–L2
    DELETE /api/packs/{id}

Bind it to localhost (the default) or reach it through an SSH tunnel: there is
no authentication, and anyone who can reach the port can upload packs.
"""

# No ``from __future__ import annotations`` here: FastAPI resolves the endpoint
# annotations (UploadFile, Query) at runtime, and they are imported inside
# create_app so the CLI works without the GUI extras installed.
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import _paths

API_VERSION = "1"
MAX_UPLOAD_BYTES = int(float(os.environ.get("MLIPEVAL_MAX_UPLOAD_GB", "4")) * 2**30)


def default_store() -> Path:
    return Path(os.environ.get("MLIPEVAL_STORE", "~/.mlipeval/workbench")).expanduser()


class Store:
    """Uploaded packs and their rescored results, one directory per pack id."""

    def __init__(self, root: Path, reference_path: Path) -> None:
        _paths.import_mlipeval()
        from mlipeval import pack

        self.pack = pack
        self.root = Path(root)
        (self.root / "packs").mkdir(parents=True, exist_ok=True)
        self.reference_path = Path(reference_path)
        self.reference = pack.open_reference(self.reference_path)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock(self, pack_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(pack_id, threading.Lock())

    def dir(self, pack_id: str) -> Path:
        if not pack_id.isalnum():
            raise KeyError(pack_id)
        return self.root / "packs" / pack_id

    def container(self, pack_id: str) -> Path:
        path = self.dir(pack_id) / "pack.mlippack"
        if not path.exists():
            raise KeyError(pack_id)
        return path

    def open(self, pack_id: str):
        return self.pack.open_model_pack(self.container(pack_id))

    def results_dir(self, pack_id: str) -> Path:
        return self.dir(pack_id) / "rescored"

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for d in sorted((self.root / "packs").iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                row = json.loads(meta.read_text())
                row["rescored"] = (d / "rescored" / "rescore_summary.json").exists()
                rows.append(row)
        return rows


def pack_summary(p) -> dict[str, Any]:
    m = p.manifest
    return {
        "pack_id": p.pack_id,
        "model_key": p.model_key,
        "label": m.get("label"),
        "model": {k: m.get("model", {}).get(k) for k in ("name", "checkpoint_sha256", "dtype", "device",
                                                        "native_protocol")},
        "reference": m.get("reference"),
        "levels": {lv: sorted(v) for lv, v in m.get("levels", {}).items()},
        "n_views": len(m.get("views", [])),
        "provenance": {k: m.get("provenance", {}).get(k) for k in ("built_at", "host", "backfilled", "absent")},
    }


def create_app(store: Store):
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    from mlipeval import rescore

    app = FastAPI(title="MLIP Workbench backend", version=API_VERSION)

    def _get(pack_id: str):
        try:
            return store.open(pack_id)
        except KeyError:
            raise HTTPException(404, f"no pack {pack_id}") from None

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        ref = store.reference
        return {"ok": True, "api": API_VERSION, "engine": rescore.RESCORE_VERSION,
                "reference": {"id": ref.reference_id, "complete": ref.manifest.get("complete", True)},
                "n_packs": len(store.list())}

    @app.get("/api/reference")
    def reference() -> dict[str, Any]:
        m = store.reference.manifest
        return {"reference_id": m["reference_id"], "system": m.get("system"), "complete": m.get("complete", True),
                "levels": m["levels"], "protocols": m.get("protocols"), "missing": m.get("missing", []),
                "fallbacks": m.get("provenance", {}).get("fallbacks", []),
                "content_sha256": m.get("content_sha256"), "path": str(store.reference_path)}

    @app.get("/api/packs")
    def list_packs() -> list[dict[str, Any]]:
        return store.list()

    @app.post("/api/packs")
    def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="upload_", dir=store.root))
        try:
            tmp = tmp_dir / "pack.mlippack"
            size = 0
            with open(tmp, "wb") as handle:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES / 2**30:.1f} GiB")
                    handle.write(chunk)
            try:
                p = store.pack.open_model_pack(tmp)
            except (store.pack.PackError, ValueError, OSError) as exc:
                raise HTTPException(422, f"not a valid model pack: {exc}") from None
            except Exception as exc:  # zipfile.BadZipFile and friends
                raise HTTPException(422, f"not a valid model pack: {type(exc).__name__}: {exc}") from None
            verification = store.pack.verify_model_pack(p, store.reference)
            target = store.dir(p.pack_id)
            with store.lock(p.pack_id):
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp), target / "pack.mlippack")
                meta = pack_summary(p) | {"filename": file.filename, "size": size,
                                          "verification_ok": verification["ok"]}
                (target / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            return {"pack": meta, "verification": verification}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @app.get("/api/packs/{pack_id}")
    def get_pack(pack_id: str) -> dict[str, Any]:
        p = _get(pack_id)
        return {"pack": pack_summary(p), "verification": store.pack.verify_model_pack(p, store.reference)}

    @app.post("/api/packs/{pack_id}/rescore")
    def do_rescore(pack_id: str, levels: str = Query("L0,L1,L2"), compare: bool = Query(True)) -> dict[str, Any]:
        p = _get(pack_id)
        wanted = [lv.strip().upper() for lv in levels.split(",") if lv.strip()]
        bad = [lv for lv in wanted if lv not in rescore.LEVELS]
        if bad:
            raise HTTPException(400, f"cannot rescore {bad}; packs carry {list(rescore.LEVELS)}")
        with store.lock(pack_id):
            out = rescore.rescore(p, store.reference, levels=wanted)
            out_dir = store.results_dir(pack_id)
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out.save(out_dir)
            summary = out.summary()
            if compare:
                summary["comparison"] = rescore.compare_with_pack(out)
            (out_dir / "rescore_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=float))
            (out_dir / "workbench.json").write_text(
                json.dumps(rescore.workbench_model(out), ensure_ascii=False, default=float))
        return summary

    def _results_dir(pack_id: str) -> Path:
        _get(pack_id)
        d = store.results_dir(pack_id)
        if not (d / "rescore_summary.json").exists():
            raise HTTPException(409, "not rescored yet: POST /api/packs/{id}/rescore first")
        return d

    @app.get("/api/packs/{pack_id}/results")
    def results(pack_id: str) -> dict[str, Any]:
        d = _results_dir(pack_id)
        summary = json.loads((d / "rescore_summary.json").read_text())
        return {"summary": summary, "files": sorted(p.name for p in d.glob("L*.json"))}

    @app.get("/api/packs/{pack_id}/results/{stem}")
    def result_file(pack_id: str, stem: str):
        d = _results_dir(pack_id)
        model_key = _get(pack_id).model_key
        name = f"{stem}_{model_key}.json" if stem in ("L1", "L2") else f"{stem}.json"
        path = d / name
        if stem not in ("L0_audit", "L0_diagnostic", "L1", "L2") or not path.exists():
            raise HTTPException(404, f"no result {stem} for pack {pack_id}")
        return JSONResponse(json.loads(path.read_text()))

    @app.get("/api/packs/{pack_id}/workbench")
    def workbench(pack_id: str):
        d = _results_dir(pack_id)
        return JSONResponse(json.loads((d / "workbench.json").read_text()))

    @app.delete("/api/packs/{pack_id}")
    def delete(pack_id: str) -> dict[str, Any]:
        _get(pack_id)
        with store.lock(pack_id):
            shutil.rmtree(store.dir(pack_id))
        return {"deleted": pack_id}

    docs = _paths.RELEASE_ROOT / "docs"
    if docs.is_dir():
        @app.get("/")
        def root():
            return RedirectResponse("/workbench/")

        app.mount("/", StaticFiles(directory=str(docs), html=True), name="docs")
    return app


def run(*, host: str = "127.0.0.1", port: int = 8765, store: str | None = None,
        reference: str | None = None) -> int:
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        raise SystemExit("mlipeval serve needs the GUI extras: pip install 'mlipeval[gui]' "
                         "(fastapi, uvicorn, python-multipart)") from None
    _paths.import_mlipeval()
    from mlipeval import pack

    ref_path = Path(reference) if reference else pack.default_reference_path()
    if not ref_path.exists():
        raise SystemExit(f"mlipeval serve: reference pack not found: {ref_path}\n"
                         "build it with `mlipeval pack reference` or pass --reference")
    s = Store(Path(store).expanduser() if store else default_store(), ref_path)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: serving on {host}; the API has no authentication")
    print(f"reference {s.reference.reference_id} (complete={s.reference.manifest.get('complete', True)}); "
          f"store {s.root}")
    print(f"Workbench: http://{host}:{port}/workbench/   API: http://{host}:{port}/docs")
    import uvicorn

    uvicorn.run(create_app(s), host=host, port=port, log_level="info")
    return 0
