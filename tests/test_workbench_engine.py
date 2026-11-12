"""The Workbench's in-browser engine reproduces the HPC L0-L2 results.

For each of the nine example models in ``docs/workbench/examples/gst225`` the
page's engine (``docs/workbench/engine.js``) parses the upload files, computes
every L0 (audit), L1 and L2 metric and the relative level scores, and the test
checks them against ``expected_metrics.json`` (the HPC values) and the scores
embedded in ``index.html``.  Needs Playwright with Chrome; skipped otherwise.
"""

from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
playwright = pytest.importorskip("playwright.sync_api")

RUNNER = r"""
async () => {
  const E = window.MLIPEngine, base = '/workbench/examples/gst225/';
  const idx = await (await fetch(base + 'index.json')).json();
  const txt = async p => { const b = await (await fetch(base + p)).blob(); return E.readText(new File([b], p.split('/').pop())); };
  const expected = (await (await fetch(base + 'expected_metrics.json')).json()).models;
  const refL0 = E.parseExtxyz(await txt(idx.reference.files.l0_labels));
  const refL2 = JSON.parse(await txt(idx.reference.files.l2_reference)), refEos = E.parseCSV(await txt(idx.reference.files.l2_eos));
  const aimd = JSON.parse(await txt(idx.reference.files.l1_reference)).aimd_nve_drift_meV_per_atom_per_ps;
  const out = {};
  for (const [key, m] of Object.entries(idx.models)) {
    const f = m.files;
    const ml = E.parseExtxyz(await txt(f['l0_predictions.extxyz.gz']));
    const l0 = E.evalL0(refL0, ml);
    const l1 = E.evalL1({ pec: E.parseCSV(await txt(f['l1_pec.csv'])), lattice: E.parseCSV(await txt(f['l1_lattice_scan.csv'])),
      nve3: E.parseCSV(await txt(f['l1_nve_dt3.csv'])), nve15: E.parseCSV(await txt(f['l1_nve_dt1.5.csv'])),
      stability: E.parseCSV(await txt(f['l1_stability_2000K.csv'])), probes: JSON.parse(await txt(f['l1_probes.json'])) }, { aimdDrift: aimd });
    const l2 = E.evalL2(refL2, refEos, E.parseCSV(await txt(f['l2_relax.csv'])), E.parseCSV(await txt(f['l2_eos.csv'])));
    const all = l0.metrics.concat(l1.metrics, l2.metrics);
    if (m.training_overlap) E.applyOverlap(all, JSON.parse(await txt(m.training_overlap)).metrics);
    const cmp = E.compare(all, expected[key]);
    const sc = E.scoreAgainst(window.PAGE_D, key + '@test', E.indicators(all, l1.gate), ['L0', 'L1', 'L2']);
    out[key] = { check: E.checkL0(refL0, ml).ok, n: cmp.n, agree: cmp.agree, diffs: cmp.diffs.slice(0, 5),
                 gate: l1.gate.pass, gate_hpc: expected[key]['L1:gate_pass'][0], level: sc.level };
  }
  return out;
}
"""


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_engine_reproduces_hpc_l0_l2(server):
    html = (DOCS / "workbench" / "index.html").read_text(encoding="utf-8")
    start = html.index("const D = ") + len("const D = ")
    page_d, _ = json.JSONDecoder().raw_decode(html[start:])
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:  # no Chrome on this machine
            pytest.skip(f"Chrome not available: {exc}")
        page = browser.new_page()
        page.goto(server + "/")
        page.add_script_tag(url=server + "/workbench/engine.js")
        page.evaluate("d => { window.PAGE_D = d; }", {"indicators": page_d["indicators"]})
        results = page.evaluate(RUNNER)
        browser.close()
    assert set(results) == set(page_d["order"])
    for key, r in results.items():
        assert r["check"], key
        assert r["n"] > 100 and r["agree"] == r["n"], (key, r["diffs"])
        assert r["gate"] == r["gate_hpc"], key
        for level in ("L0", "L1", "L2"):
            # the page stores raw indicators to 4 significant figures, hence 0.05
            assert abs(r["level"][level] - page_d["level_scores"][level][key]) < 0.05, (key, level, r["level"][level])
