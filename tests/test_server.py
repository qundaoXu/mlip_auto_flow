"""Workbench backend over HTTP (FastAPI TestClient).

Needs the GUI extras and the same data as the round-trip test in
``test_packs.py`` (``MLIPEVAL_OUTPUT``, ``MLIPEVAL_REFERENCE``); skipped otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_framework" / "scripts"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

REFERENCE = os.environ.get("MLIPEVAL_REFERENCE")
OUTPUT = os.environ.get("MLIPEVAL_OUTPUT")
pytestmark = pytest.mark.skipif(not (REFERENCE and OUTPUT and Path(REFERENCE).exists()),
                                reason="needs MLIPEVAL_OUTPUT and MLIPEVAL_REFERENCE")


def test_upload_rescore_and_read_back(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MLIPEVAL_PACK_CACHE", str(tmp_path / "unpacked"))
    from mlipeval import pack
    from mlipeval_cli import server

    ref = pack.open_reference(REFERENCE)
    path, _ = pack.build_model_pack("mace_mpa0", tmp_path / "m.mlippack", reference=ref)
    client = TestClient(server.create_app(server.Store(tmp_path / "store", Path(REFERENCE))))

    assert client.get("/api/health").json()["ok"]
    with open(path, "rb") as fh:
        up = client.post("/api/packs", files={"file": ("m.mlippack", fh)})
    assert up.status_code == 200 and up.json()["verification"]["ok"]
    pid = up.json()["pack"]["pack_id"]

    assert client.get(f"/api/packs/{pid}/results").status_code == 409  # not rescored yet
    summary = client.post(f"/api/packs/{pid}/rescore").json()
    assert all(c["identical"] for c in summary["comparison"].values())

    wb = client.get(f"/api/packs/{pid}/workbench").json()
    assert {"metrics", "gate", "l0", "l1", "l2"} <= set(wb)
    assert client.get(f"/api/packs/{pid}/results/L1").json()["level"] == "L1"

    assert client.post("/api/packs", files={"file": ("x.mlippack", b"not a zip")}).status_code == 422
    assert client.post(f"/api/packs/{pid}/rescore?levels=L4").status_code == 400
    assert client.get("/api/packs/../etc").status_code == 404
    assert client.delete(f"/api/packs/{pid}").status_code == 200
    assert client.get(f"/api/packs/{pid}").status_code == 404
