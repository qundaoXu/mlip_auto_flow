/* MLIP Workbench analysis engine: L0–L2 from uploaded files, in the browser.
 *
 * A line-by-line port of the mlipeval functions that turn model outputs into
 * MetricResults: l0_reference.evaluate_l0, the l1_pes judges (diagnose_pec,
 * invariance/force/stress, lattice scan, stability, timestep, NVE) and the
 * l2_static metrics (relaxation, phase order, phase ladder, formation energy,
 * Birch–Murnaghan EOS and its seed ensemble), plus the L0–L2 ranking indicators.
 * Metric ids, units and statuses match the Python output, so the page can show
 * a browser result next to the HPC file it should reproduce.
 *
 * Exposes window.MLIPEngine; no DOM access.
 */
(function (root) {
'use strict';

/* ------------------------------------------------------------------ helpers */
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const sum = a => a.reduce((s, v) => s + v, 0);
const mean = a => a.length ? sum(a) / a.length : NaN;
const nanmean = a => mean(a.filter(isNum));
const nanmin = a => { const f = a.filter(isNum); return f.length ? Math.min(...f) : NaN; };
const maxAbs = a => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
function percentile(arr, p) {               // numpy default ('linear')
  const a = Float64Array.from(arr).sort(); if (!a.length) return NaN;
  const r = p / 100 * (a.length - 1), lo = Math.floor(r), hi = Math.ceil(r);
  return a[lo] + (a[hi] - a[lo]) * (r - lo);
}
function sd1(a) { if (a.length < 2) return NaN; const m = mean(a); return Math.sqrt(sum(a.map(v => (v - m) ** 2)) / (a.length - 1)); }
function errorStats(e) {
  const v = Array.from(e).filter(isNum);
  if (!v.length) return { mae: NaN, rmse: NaN, p99: NaN, max: NaN, bias: NaN, n: NaN };
  const a = v.map(Math.abs);
  return { mae: mean(a), rmse: Math.sqrt(mean(v.map(x => x * x))), p99: percentile(a, 99), max: Math.max(...a), bias: mean(v), n: v.length };
}
function polyfit(x, y, deg) {               // least squares, normal equations (deg <= 2)
  const n = deg + 1, A = Array.from({ length: n }, () => new Array(n).fill(0)), b = new Array(n).fill(0);
  for (let i = 0; i < x.length; i++) for (let r = 0; r < n; r++) {
    b[r] += y[i] * x[i] ** (deg - r);
    for (let c = 0; c < n; c++) A[r][c] += x[i] ** (deg - r) * x[i] ** (deg - c);
  }
  return solve(A, b);
}
function solve(A, b) {                       // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (M[c][c] === 0) return new Array(n).fill(NaN);
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}
function slope(x, y) {                       // least-squares slope on centred data (numpy.polyfit deg 1)
  const ok = x.map((_, i) => isNum(x[i]) && isNum(y[i])), xx = x.filter((_, i) => ok[i]), yy = y.filter((_, i) => ok[i]);
  if (xx.length < 3) return NaN;
  const mx = mean(xx), my = mean(yy); let sxy = 0, sxx = 0;
  for (let i = 0; i < xx.length; i++) { sxy += (xx[i] - mx) * (yy[i] - my); sxx += (xx[i] - mx) ** 2; }
  return sxy / sxx;
}
const sign = v => v > 0 ? 1 : v < 0 ? -1 : 0;
const fmtG = v => String(+v.toPrecision(12)).replace(/\.0+$/, '');

function metric(id, level, value, unit, o = {}) {
  return { metric_id: id, level, value, unit, status: o.status || 'ok', tier: o.tier || null,
           failure_reason: o.reason || null, extra: o.extra || {}, uncertainty: o.uncertainty ?? null };
}

/* ------------------------------------------------------------------ parsers */
async function readText(file) {
  if (/\.gz$/i.test(file.name)) {
    if (!('DecompressionStream' in root)) throw new Error('this browser cannot read .gz files; upload the uncompressed file');
    const stream = file.stream().pipeThrough(new DecompressionStream('gzip'));
    return await new Response(stream).text();
  }
  return await file.text();
}

function parseInfo(line) {
  const info = {}; const re = /([A-Za-z_][\w.\-]*)=("[^"]*"|\S+)/g; let m;
  while ((m = re.exec(line))) {
    let v = m[2]; if (v[0] === '"') v = v.slice(1, -1);
    if (v.startsWith('_JSON ')) { try { info[m[1]] = JSON.parse(v.slice(6)).flat(); continue; } catch (e) { /* fall through */ } }
    const parts = v.trim().split(/\s+/);
    if (parts.length > 1 && parts.every(p => p !== '' && !isNaN(+p))) info[m[1]] = parts.map(Number);
    else if (v !== '' && !isNaN(+v)) info[m[1]] = +v;
    else info[m[1]] = v;
  }
  return info;
}

function parseExtxyz(text) {
  const L = text.split(/\r?\n/), frames = []; let i = 0;
  while (i < L.length) {
    if (!L[i].trim()) { i++; continue; }
    const n = parseInt(L[i], 10);
    if (!Number.isInteger(n) || n <= 0) throw new Error(`line ${i + 1}: expected an atom count, found "${L[i].slice(0, 40)}"`);
    const info = parseInfo(L[i + 1] || '');
    const props = String(info.Properties || 'species:S:1:pos:R:3').split(':');
    const cols = []; let c = 0;
    for (let k = 0; k + 2 < props.length; k += 3) { const w = +props[k + 2]; cols.push({ name: props[k], type: props[k + 1], start: c, width: w }); c += w; }
    const species = new Array(n), arrays = {};
    for (const col of cols) if (col.type !== 'S') arrays[col.name] = new Float64Array(n * col.width);
    for (let a = 0; a < n; a++) {
      const f = (L[i + 2 + a] || '').trim().split(/\s+/);
      if (f.length < c) throw new Error(`frame ${frames.length + 1}, atom ${a + 1}: ${f.length} columns, expected ${c}`);
      for (const col of cols) {
        if (col.type === 'S') { species[a] = f[col.start]; continue; }
        for (let w = 0; w < col.width; w++) arrays[col.name][a * col.width + w] = +f[col.start + w];
      }
    }
    frames.push({ n, info, species, arrays, line: i + 1 });
    i += n + 2;
  }
  if (!frames.length) throw new Error('no frames found');
  return frames;
}

function parseCSV(text) {
  const meta = {}, notes = [], rows = []; let header = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim(); if (!line) continue;
    if (line.startsWith('#')) { const t = line.slice(1).trim(); const m = t.match(/^([\w.\-]+)\s*[=:]\s*(.+)$/); if (m) meta[m[1]] = m[2].trim(); else notes.push(t); continue; }
    const cells = line.split(',').map(s => s.trim());
    if (!header) { header = cells; continue; }
    const o = {}; header.forEach((h, k) => { const v = cells[k]; o[h] = v === undefined || v === '' ? null : (isNaN(+v) ? v : +v); }); rows.push(o);
  }
  if (!header) throw new Error('no header row');
  return { header, rows, meta, notes };
}
function needCols(csv, cols, what) {
  const miss = cols.filter(c => !csv.header.includes(c));
  if (miss.length) throw new Error(`${what}: missing column(s) ${miss.join(', ')}`);
}

/* ------------------------------------------------------------------ L0 */
const pick = (o, keys) => { for (const k of keys) if (o[k] !== undefined) return [k, o[k]]; return [null, undefined]; };
const REF_E = ['REF_energy', 'DFT_energy', 'energy'], REF_F = ['REF_forces', 'DFT_forces', 'forces'], REF_S = ['REF_stress', 'DFT_stress', 'stress'];
const ML_E = ['MLIP_energy', 'energy'], ML_F = ['MLIP_forces', 'forces'], ML_S = ['MLIP_stress', 'stress'];

function checkL0(ref, ml) {
  const issues = [];
  if (ref.length !== ml.length) return { ok: false, issues: [`frame count differs: reference ${ref.length}, MLIP ${ml.length}`] };
  let maxDx = 0;
  for (let k = 0; k < ref.length; k++) {
    const r = ref[k], m = ml[k];
    if (r.n !== m.n) { issues.push(`frame ${k + 1}: ${r.n} atoms in the reference, ${m.n} in the MLIP file`); break; }
    if (r.species.join() !== m.species.join()) { issues.push(`frame ${k + 1}: species order differs`); break; }
    if (pick(r.info, REF_E)[0] === null) { issues.push(`frame ${k + 1}: reference energy missing (REF_energy or energy)`); break; }
    if (pick(m.info, ML_E)[0] === null) { issues.push(`frame ${k + 1}: MLIP energy missing (MLIP_energy or energy)`); break; }
    if (pick(r.arrays, REF_F)[0] === null) { issues.push(`frame ${k + 1}: reference forces missing (REF_forces or forces)`); break; }
    if (pick(m.arrays, ML_F)[0] === null) { issues.push(`frame ${k + 1}: MLIP forces missing (MLIP_forces or forces)`); break; }
    if (r.arrays.pos && m.arrays.pos) for (let j = 0; j < r.arrays.pos.length; j++) maxDx = Math.max(maxDx, Math.abs(r.arrays.pos[j] - m.arrays.pos[j]));
  }
  if (maxDx > 1e-3) issues.push(`positions differ by up to ${maxDx.toFixed(4)} Å: the predictions are not on the reference frames`);
  return { ok: !issues.length, issues, maxDx };
}

function composition(sp) { const c = {}; sp.forEach(s => c[s] = (c[s] || 0) + 1); return Object.keys(c).sort().map(k => k + c[k]).join(''); }

function evalL0(ref, ml, opt = {}) {
  const out = []; const add = (id, v, unit, o) => out.push(metric(id, 'L0', v, unit, o));
  const nF = ref.length;
  const eRef = ref.map(f => +pick(f.info, REF_E)[1]), eMl = ml.map(f => +pick(f.info, ML_E)[1]);
  const nAt = ref.map(f => f.n), comps = ref.map(f => composition(f.species));
  const nFail = eMl.filter(v => !isNum(v)).length;
  add('L0.model_eval_failures', nFail, 'count', { status: nFail ? 'failed' : 'ok', tier: 'S' });
  add('L0.nonfinite_energies', nFail, 'count', { status: nFail ? 'failed' : 'ok', tier: 'S' });
  // per-composition constant offset, per atom
  const offsets = {};
  for (const c of new Set(comps)) {
    let d = 0, n = 0; comps.forEach((cc, k) => { if (cc === c && isNum(eMl[k] - eRef[k])) { d += eMl[k] - eRef[k]; n += nAt[k]; } });
    offsets[c] = n ? d / n : NaN;
  }
  const errPa = eRef.map((e, k) => (eMl[k] - offsets[comps[k]] * nAt[k] - e) / nAt[k]);
  const se = errorStats(errPa.map(v => v * 1000));
  for (const k of ['mae', 'rmse', 'p99', 'max']) add(`L0.energy_${k}`, se[k], 'meV/atom', { status: 'provisional', tier: 'B', extra: { offsets_eV_per_atom: offsets } });
  const ov = Object.values(offsets);
  add('L0.energy_offset_spread', ov.length > 1 ? (Math.max(...ov) - Math.min(...ov)) * 1000 : 0, 'meV/atom', { tier: 'S' });
  // forces
  const fr = ref.map(f => pick(f.arrays, REF_F)[1]), fm = ml.map(f => pick(f.arrays, ML_F)[1]);
  const nComp = sum(fr.map(a => a.length)), rf = new Float64Array(nComp), mf = new Float64Array(nComp);
  { let o = 0; fr.forEach((a, k) => { rf.set(a, o); mf.set(fm[k], o); o += a.length; }); }
  const df = mf.map((v, i) => v - rf[i]);
  const sf = errorStats(df);
  for (const k of ['mae', 'rmse', 'p99', 'max']) add(`L0.force_${k}`, sf[k], 'eV/A', { status: 'provisional', tier: 'B' });
  { const mx = mean(rf), my = mean(mf); let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < nComp; i++) { const a = rf[i] - mx, b = mf[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
    add('L0.force_component_r2', (sxy / Math.sqrt(sxx * syy)) ** 2, '-', { tier: 'B' }); }
  // by species (component errors)
  const bySp = {};
  ref.forEach((f, k) => f.species.forEach((s, a) => { (bySp[s] = bySp[s] || []).push(fm[k][3 * a] - fr[k][3 * a], fm[k][3 * a + 1] - fr[k][3 * a + 1], fm[k][3 * a + 2] - fr[k][3 * a + 2]); }));
  for (const s of Object.keys(bySp).sort()) { const st = errorStats(bySp[s]); add(`L0.force_mae.species=${s}`, st.mae, 'eV/A', { status: 'provisional', tier: 'B', extra: { rmse: st.rmse, p99: st.p99, max: st.max, n: st.n } }); }
  // by reference force magnitude (per-atom vector errors)
  const edges = [0, 0.5, 1, 2, 4, 8, Infinity], byMag = {}, frameErr = [], frameMax = [];
  ref.forEach((f, k) => {
    const errs = [];
    for (let a = 0; a < f.n; a++) {
      const x = fr[k][3 * a], y = fr[k][3 * a + 1], z = fr[k][3 * a + 2];
      const mag = Math.hypot(x, y, z), err = Math.hypot(fm[k][3 * a] - x, fm[k][3 * a + 1] - y, fm[k][3 * a + 2] - z);
      let b = 0; for (let e = 1; e < edges.length - 1; e++) if (mag >= edges[e]) b = e;
      const lab = isFinite(edges[b + 1]) ? `[${fmtG(edges[b])},${fmtG(edges[b + 1])})` : `>=${fmtG(edges[b])}`;
      (byMag[lab] = byMag[lab] || []).push(err); errs.push(err);
    }
    frameErr.push(Math.sqrt(mean(errs.map(v => v * v)))); frameMax.push(Math.max(...errs));
  });
  for (const lab of Object.keys(byMag).sort()) { const st = errorStats(byMag[lab]); add(`L0.force_vector_mae.|F|=${lab}`, st.mae, 'eV/A', { status: 'provisional', tier: 'B', extra: { p99: st.p99, max: st.max, n: st.n } }); }
  // per tag (per-frame force RMSE over components; energy error after the set-wide offset)
  const perFrameRmse = fr.map((a, k) => Math.sqrt(mean(Array.from(a, (v, i) => (fm[k][i] - v) ** 2))));
  for (const tag of (opt.tags || ['pool', 'state_kind'])) {
    const vals = ref.map(f => f.info[tag]); if (vals.some(v => v === undefined || typeof v === 'number')) continue;
    const groups = [...new Set(vals.map(String))].sort();
    for (const g of groups) { const idx = vals.map((v, k) => String(v) === g ? k : -1).filter(k => k >= 0);
      add(`L0.force_rmse.${tag}=${g}`, mean(idx.map(k => perFrameRmse[k])), 'eV/A', { status: 'provisional', tier: 'B', extra: { n_frames: idx.length } }); }
    for (const g of groups) { const idx = vals.map((v, k) => String(v) === g ? k : -1).filter(k => k >= 0);
      add(`L0.energy_mae.${tag}=${g}`, nanmean(idx.map(k => Math.abs(errPa[k]) * 1000)), 'meV/atom', { status: 'provisional', tier: 'B', extra: { n_frames: idx.length } }); }
  }
  // stress (ASE sign, eV/A^3, 3x3)
  const rs = ref.map(f => pick(f.info, REF_S)[1]), ms = ml.map(f => pick(f.info, ML_S)[1]);
  const withS = rs.map((s, k) => Array.isArray(s) && s.length === 9 && Array.isArray(ms[k]) && ms[k].length === 9);
  if (withS.some(Boolean)) {
    const d = []; rs.forEach((s, k) => { if (withS[k]) for (let j = 0; j < 9; j++) d.push((ms[k][j] - s[j]) * 1000); });
    const st = errorStats(d);
    for (const k of ['mae', 'rmse', 'max']) add(`L0.stress_${k}`, st[k], 'meV/A^3', { status: 'provisional', tier: 'B', extra: { n_frames: withS.filter(Boolean).length } });
  } else add('L0.stress_rmse', null, 'meV/A^3', { status: 'skipped', tier: 'C', reason: 'no stresses on both sides' });
  // plot data in the page's shape
  const take = []; { let s = 12345; const n = Math.min(700, nComp); const seen = new Set();
    while (take.length < n) { s = (s * 1103515245 + 12345) % 2147483648; const i = s % nComp; if (!seen.has(i)) { seen.add(i); take.push(i); } } take.sort((a, b) => a - b); }
  const r4 = v => +v.toFixed(4);
  return { metrics: out, plot: { parity: [take.map(i => r4(rf[i])), take.map(i => r4(mf[i]))], frame_err: frameErr.map(r4), frame_max: frameMax.map(r4) },
           pools: ref.map(f => String(f.info.pool ?? f.info.config_type ?? '')) };
}

/* ------------------------------------------------------------------ L1 */
const T = { pec_min_repulsion_r: 1.6, pec_force_mismatch_rel: 0.05, pec_convergence_ratio: 2.5, scan_smoothness_ratio: 25.0,
            force_consistency_rel: 5e-3, translation_energy_tol: 1e-6, rotation_energy_tol: 1e-5, net_force_tol: 1e-6,
            stress_consistency_rel: 5e-2, nve_drift_tol: 5.0, com_drift_tol: 1e-3, min_distance_floor: 1.6 };
const NOISE_MARGIN = 5.0;
const L1m = (id, v, unit, o = {}) => metric(id, 'L1', v, unit, Object.assign({ tier: 'S' }, o));

function diagnosePec(pair, r, E, F, noiseResidual) {
  const n = r.length, dr = r[1] - r[0];
  const nNonfinite = E.filter(v => !isNum(v)).length + F.filter(v => !isNum(v)).length;
  let kMin = 0; for (let k = 1; k < n; k++) if (E[k] < E[kMin]) kMin = k;
  const shortIdx = r.map((v, k) => v <= T.pec_min_repulsion_r ? k : -1).filter(k => k >= 0);
  let mono = true; if (shortIdx.length > 1) for (let j = 1; j < shortIdx.length; j++) if (!(E[shortIdx[j]] - E[shortIdx[j - 1]] <= 1e-9)) mono = false;
  const scale = Math.max(maxAbs(F), 1e-8);
  const residual = (step, trim = 0) => {
    const rr = [], ee = [], ff = []; for (let k = 0; k < n; k += step) { rr.push(r[k]); ee.push(E[k]); ff.push(F[k]); }
    if (rr.length < 5) return NaN;
    const h = rr[1] - rr[0]; let best = -1;
    for (let k = 1; k < rr.length - 1; k++) {
      const fn = -(ee[k + 1] - ee[k - 1]) / (2 * h);
      if (!isNum(fn) || !isNum(ff[k])) continue;
      if (trim > 0 && !(rr[k] >= rr[0] + trim && rr[k] <= rr[rr.length - 1] - trim)) continue;
      best = Math.max(best, Math.abs(fn - ff[k]));
    }
    return best < 0 ? Infinity : best / scale;
  };
  const mismatch = residual(1), trim = 3 * 2 * dr, fine = residual(1, trim), coarse = residual(2, trim);
  const ratio = fine > 0 && isNum(coarse) && isNum(fine) ? coarse / fine : NaN;
  return { pair, r_min: r[kMin], e_min: E[kMin], mono, mismatch, ratio, fine, noise: noiseResidual, nNonfinite };
}
function pecMetric(d) {
  const reasons = [];
  if (!d.mono) reasons.push('no monotonic short-range repulsion');
  if (d.nNonfinite > 0) reasons.push('non-finite energy or force on the curve');
  if (d.mismatch > T.pec_force_mismatch_rel) reasons.push('energy/force derivative mismatch');
  if (isNum(d.ratio) && d.ratio < T.pec_convergence_ratio && d.fine > d.noise) reasons.push('derivative residual does not converge as O(h^2): possible discontinuity');
  return L1m(`L1.pec.${d.pair}`, { r_min_A: d.r_min, e_min_eV: d.e_min }, 'eV', { status: reasons.length ? 'failed' : 'ok', reason: reasons.join('; ') || null,
    extra: { monotonic_repulsive: d.mono, max_force_mismatch_rel: d.mismatch, derivative_convergence_ratio: d.ratio } });
}
function probeMetrics(p) {
  const noise = +p.noise_eV || 0, out = [];
  const tolT = Math.max(T.translation_energy_tol, NOISE_MARGIN * noise), tolR = Math.max(T.rotation_energy_tol, NOISE_MARGIN * noise);
  const tolF = Math.max(T.net_force_tol, NOISE_MARGIN * noise / p.n_atoms);
  out.push(L1m('L1.translation_invariance', p.d_translation_eV, 'eV', { status: p.d_translation_eV > tolT ? 'failed' : 'ok', extra: { tolerance_eV: tolT } }));
  out.push(L1m('L1.rotation_invariance', p.d_rotation_eV, 'eV', { status: p.d_rotation_eV > tolR ? 'failed' : 'ok', extra: { tolerance_eV: tolR } }));
  out.push(L1m('L1.net_force', p.net_force_eV_per_A_per_atom, 'eV/A/atom', { status: p.net_force_eV_per_A_per_atom > tolF ? 'failed' : 'ok', extra: { tolerance_eV_per_A_per_atom: tolF } }));
  out.push(L1m('L1.net_torque', p.net_torque_eV_per_atom, 'eV/atom'));
  const f = p.force, tf = Math.max(T.force_consistency_rel, NOISE_MARGIN * noise / (2 * f.delta_A) / f.max_abs_force);
  out.push(L1m('L1.force_vs_dE_dx', f.rel, '-', { status: f.rel > tf ? 'failed' : 'ok', extra: { tolerance_rel: tf } }));
  const s = p.stress, an = s.analytic_voigt_eV_per_A3, nu = s.numeric_voigt_eV_per_A3, sc = Math.max(maxAbs(an), 1e-8);
  const rel = Math.max(...an.map((v, i) => Math.abs(v - nu[i]))) / sc, ts = Math.max(T.stress_consistency_rel, NOISE_MARGIN * noise / (2 * s.eps) / p.volume_A3 / sc);
  out.push(L1m('L1.stress_vs_dE_dstrain', rel, '-', { status: rel > ts ? 'failed' : 'ok', extra: { tolerance_rel: ts } }));
  return out;
}
function latticeMetrics(scan) {
  const E = scan.map(r => r.energy_eV_per_atom), ok = E.filter(isNum), n = E.length;
  let ratio = NaN, maxSecond = NaN;
  if (ok.length > 4) { const s2 = []; for (let k = 2; k < ok.length; k++) s2.push(Math.abs(ok[k] - 2 * ok[k - 1] + ok[k - 2]));
    const med = percentile(s2, 50); ratio = med > 0 ? Math.max(...s2) / med : Infinity; maxSecond = Math.max(...s2) * 1000; }
  let rises = false; if (ok.length > 3) { const third = ok.slice(0, Math.max(1, Math.floor(ok.length / 3))); rises = third.every((v, k) => k === 0 || v - third[k - 1] < 0); }
  return [L1m('L1.lattice_scan_finite', ok.length, 'count', { status: ok.length < n ? 'failed' : 'ok' }),
          L1m('L1.lattice_scan_smoothness', ratio, 'max/median |d2E|', { status: isNum(ratio) && ratio > T.scan_smoothness_ratio ? 'failed' : 'ok', extra: { max_second_difference_meV_per_atom: maxSecond } }),
          L1m('L1.lattice_scan_compression_monotonic', rises, '-', { status: rises ? 'ok' : 'failed' })];
}
function mdRun(csv) {
  const t = csv.rows.map(r => r.time_fs), e = csv.rows.map(r => r.etot_eV_per_atom);
  const completed = String(csv.meta.completed ?? 'true').toLowerCase() !== 'false';
  return { t, tps: t.map(v => v / 1000), e, T: csv.rows.map(r => r.temperature_K), dmin: csv.rows.map(r => r.min_distance_A),
           com: csv.rows.map(r => r.com_drift_A), completed, drift: slope(t.map(v => v / 1000), e.map(v => v * 1000)) };
}
function evalL1(f, opt = {}) {
  const out = [], plot = {};
  if (f.pec) {
    const byPair = {}; for (const row of f.pec.rows) (byPair[row.pair] = byPair[row.pair] || []).push(row);
    const noise = (f.probes && f.probes.pec_energy_noise_eV) || {};
    const pairs = Object.keys(byPair);
    plot.pec = { r: [], pairs, e: [], mono: [] };
    for (const p of pairs) {
      const rows = byPair[p].slice().sort((a, b) => a.r_A - b.r_A), r = rows.map(x => x.r_A), E = rows.map(x => x.energy_eV), F = rows.map(x => x.force_eV_per_A);
      const scale0 = Math.max(maxAbs(F.filter(isNum)), 1e-8), dr = r[1] - r[0];
      const d = diagnosePec(p, r, E, F, NOISE_MARGIN * Math.SQRT2 * (+noise[p] || 0) / (2 * dr) / scale0);
      out.push(pecMetric(d));
      if (!plot.pec.r.length) plot.pec.r = r.filter((_, k) => k % 2 === 0).map(v => +v.toFixed(3));
      plot.pec.e.push(E.filter((_, k) => k % 2 === 0).map(v => isNum(v) ? +v.toFixed(4) : null)); plot.pec.mono.push(d.mono);
    }
  }
  if (f.probes) out.push(...probeMetrics(f.probes));
  if (f.lattice) { out.push(...latticeMetrics(f.lattice.rows)); const R = f.lattice.rows;
    plot.scan = { s: R.map(r => +r.scale.toFixed(3)), e: R.map(r => +r.energy_eV_per_atom.toFixed(4)), d: R.map(r => +r.min_distance_A.toFixed(3)) }; }
  if (f.stability) { const m = mdRun(f.stability), dmin = nanmin(m.dmin);
    out.push(L1m('L1.stability_2000K.completed', m.completed, '-', { status: m.completed ? 'ok' : 'failed' }));
    out.push(L1m('L1.stability_2000K.min_distance', dmin, 'A', { status: isNum(dmin) && dmin < T.min_distance_floor ? 'failed' : 'ok' }));
    out.push(L1m('L1.stability_2000K.mean_temperature', nanmean(m.T), 'K', { extra: { target_K: 2000 } })); }
  const runs = {}; if (f.nve3) runs['3'] = mdRun(f.nve3); if (f.nve15) runs['1.5'] = mdRun(f.nve15);
  if (Object.keys(runs).length) {
    const drifts = Object.values(runs).map(r => r.drift).filter(isNum);
    const spread = drifts.length > 1 ? Math.max(...drifts) - Math.min(...drifts) : NaN;
    out.push(L1m('L1.timestep_sensitivity.drift_spread', spread, 'meV/atom/ps', { status: isNum(spread) && spread > T.nve_drift_tol ? 'failed' : 'ok' }));
    for (const dt of Object.keys(runs)) out.push(L1m(`L1.timestep_sensitivity.completed.dt=${dt}fs`, runs[dt].completed, '-', { status: runs[dt].completed ? 'ok' : 'failed' }));
  }
  if (runs['3']) { const r = runs['3'], ref = opt.aimdDrift, com = Math.max(...r.com.filter(isNum));
    out.push(L1m('L1.nve_drift.dt=3fs', r.drift, 'meV/atom/ps', { status: !isNum(r.drift) || Math.abs(r.drift) > T.nve_drift_tol ? 'failed' : 'ok', tier: isNum(ref) ? 'A' : 'S',
      extra: { aimd_reference_drift_meV_per_atom_per_ps: isNum(ref) ? ref : null, ratio_to_reference: isNum(ref) && ref !== 0 ? Math.abs(r.drift) / Math.abs(ref) : null } }));
    out.push(L1m('L1.com_drift.dt=3fs', com, 'A', { status: isNum(com) && com > T.com_drift_tol ? 'failed' : 'ok' }));
    plot.nve = { t: r.tps.map(v => +v.toFixed(4)), e: r.e.map(v => +((v - r.e[0]) * 1000).toFixed(5)) }; }
  const failed = out.filter(m => m.status === 'failed').map(m => m.metric_id);
  const gate = { pass: !failed.length, n_checks: out.length, n_failed: failed.length, failed_metrics: failed,
                 decision: failed.length ? 'L1 FAILED: do not spend compute on L4/L5 until these are resolved' : 'L2-L5 may run' };
  return { metrics: out, plot, gate };
}

/* ------------------------------------------------------------------ L2 */
const GPA = 0.006241509125883258;           // ase.units.GPa, eV/A^3
function bm(v, p) { const [e0, v0, b0, bp] = p, eta = (v0 / v) ** (2 / 3); return e0 + 9 * v0 * b0 / 16 * ((eta - 1) ** 3 * bp + (eta - 1) ** 2 * (6 - 4 * eta)); }
function fitEOS(V, E) {                       // per-atom Birch–Murnaghan, Levenberg–Marquardt
  const ok = V.map((v, i) => isNum(v) && isNum(E[i])), v = V.filter((_, i) => ok[i]), e = E.filter((_, i) => ok[i]);
  const nan = { v0: NaN, e0: NaN, b0: NaN, bp: NaN, rmse: NaN, n: v.length };
  if (v.length < 4) return nan;
  const c = polyfit(v, e, 2), v0g = -c[1] / (2 * c[0]), b0g = 2 * c[0] * v0g;
  let p = [Math.min(...e), v0g, Math.max(b0g, 1e-3), 4.0];
  const res = q => v.map((x, i) => bm(x, q) - e[i]), cost = r => sum(r.map(x => x * x));
  let r = res(p), C = cost(r), lam = 1e-3;
  for (let it = 0; it < 500; it++) {
    const J = v.map(() => new Array(4).fill(0));
    for (let j = 0; j < 4; j++) { const h = 1e-7 * Math.max(Math.abs(p[j]), 1e-6), q = p.slice(); q[j] += h; const r2 = res(q);
      for (let i = 0; i < v.length; i++) J[i][j] = (r2[i] - r[i]) / h; }
    const A = [0, 1, 2, 3].map(a => [0, 1, 2, 3].map(b => sum(J.map(row => row[a] * row[b])))), g = [0, 1, 2, 3].map(a => sum(J.map((row, i) => row[a] * r[i])));
    let improved = false;
    for (let tries = 0; tries < 30; tries++) {
      const Ad = A.map((row, a) => row.map((x, b) => a === b ? x * (1 + lam) + 1e-30 : x));
      const step = solve(Ad, g.map(x => -x)); const q = p.map((x, j) => x + step[j]); const rq = res(q), Cq = cost(rq);
      if (isNum(Cq) && Cq < C) { const rel = (C - Cq) / Math.max(C, 1e-300); p = q; r = rq; C = Cq; lam = Math.max(lam / 3, 1e-12); improved = true;
        if (rel < 1e-15 || Math.max(...step.map((s, j) => Math.abs(s) / Math.max(Math.abs(p[j]), 1e-12))) < 1e-13) it = 1e9; break; }
      lam *= 4;
    }
    if (!improved) break;
  }
  return { e0: p[0], v0: p[1], b0: p[2] / GPA, bp: p[3], rmse: Math.sqrt(C / v.length) * 1000, n: v.length };
}
const L2m = (id, v, unit, o = {}) => metric(id, 'L2', v, unit, Object.assign({ tier: 'B', status: 'provisional' }, o));

function checkL2(refJson, relaxCsv) {
  const issues = []; const keys = new Set(relaxCsv.rows.map(r => r.key));
  for (const s of refJson.structures) if (!keys.has(s.key)) issues.push(`MLIP relaxation missing for "${s.key}" (${s.name})`);
  return { ok: !issues.length, issues };
}
function evalL2(refJson, refEos, relaxCsv, mlEos) {
  const out = [], plot = {}; const S = refJson.structures, byKey = Object.fromEntries(S.map(s => [s.key, s]));
  const M = Object.fromEntries(relaxCsv.rows.map(r => [r.key, r]));
  const relax = key => { const s = byKey[key], m = M[key]; if (!s || !m) return;
    const conv = String(m.converged).toLowerCase() === 'true' || m.converged === 1;
    out.push(L2m(`L2.relax.${s.relax_label}.energy_per_atom`, m.energy_eV_per_atom, 'eV/atom', { status: conv ? 'ok' : 'failed', tier: 'S',
      reason: conv ? null : `fmax ${(+m.max_force_eV_per_A).toFixed(4)} > ${m.fmax_eV_per_A}` }));
    out.push(L2m(`L2.relax.${s.relax_label}.volume_error`, 100 * (m.volume_A3_per_atom - s.volume_A3_per_atom) / s.volume_A3_per_atom, '%',
      { extra: { model_A3_per_atom: m.volume_A3_per_atom, dft_A3_per_atom: s.volume_A3_per_atom } }));
    if (s.cell_lengths_A) { const ml = [m.a_A, m.b_A, m.c_A], err = s.cell_lengths_A.map((a, i) => 100 * (ml[i] - a) / a);
      out.push(L2m(`L2.relax.${s.relax_label}.cell_length_error`, Math.max(...err.map(Math.abs)), '%', { extra: { per_axis_pct: err } })); } };
  const po = refJson.phase_order || null, phases = S.filter(s => s.role === 'phase').map(s => s.key);
  const base = po ? po.base : phases[0], other = po ? po.phase : phases[1];
  const name = k => po ? (k === other ? po.names[0] : k === base ? po.names[1] : k) : k;
  relax(base); relax(other);
  if (M[base] && M[other]) { const d = (M[other].energy_eV_per_atom - M[base].energy_eV_per_atom) * 1000, dd = (byKey[other].energy_eV_per_atom - byKey[base].energy_eV_per_atom) * 1000;
    const same = sign(d) === sign(dd);
    out.push(L2m(`L2.phase_order.${name(other)}_minus_${name(base)}`, d, 'meV/atom', { status: same ? 'provisional' : 'failed', reason: same ? null : 'phase ordering inverted vs DFT',
      extra: { dft_meV_per_atom: dd, error_meV_per_atom: d - dd, ordering_reproduced: same } })); }
  const ladder = [base, other].concat(phases.filter(k => k !== base && k !== other)).filter(k => M[k]);
  for (const k of ladder.slice(2)) relax(k);
  // phase ladder over every pair
  if (ladder.length >= 2) {
    const dftE = Object.fromEntries(ladder.map(k => [k, byKey[k].energy_eV_per_atom])), b = ladder.reduce((a, k) => dftE[k] < dftE[a] ? k : a, ladder[0]);
    const md = Object.fromEntries(ladder.map(k => [k, (M[k].energy_eV_per_atom - M[b].energy_eV_per_atom) * 1000])), dd = Object.fromEntries(ladder.map(k => [k, (dftE[k] - dftE[b]) * 1000]));
    for (const k of ladder.slice().sort((x, y) => dd[x] - dd[y])) if (k !== b) out.push(L2m(`L2.phase_ladder.dE_error.${k}`, md[k] - dd[k], 'meV/atom', { extra: { model_meV_per_atom: md[k], dft_meV_per_atom: dd[k], reference_phase: b } }));
    const pairs = []; ladder.forEach((a, i) => ladder.slice(i + 1).forEach(c => pairs.push([a, c])));
    const inv = pairs.filter(([a, c]) => dd[a] !== dd[c] && sign(md[a] - md[c]) !== sign(dd[a] - dd[c])), nOrd = pairs.filter(([a, c]) => dd[a] !== dd[c]).length;
    out.push(L2m('L2.phase_ladder.inversions', inv.length, 'count', { status: inv.length ? 'failed' : 'ok', reason: inv.length ? inv.map(([a, c]) => `${a} vs ${c} ordered backwards`).join('; ') : null, extra: { n_pairs: nOrd } }));
    out.push(L2m('L2.phase_ladder.kendall_tau', nOrd ? 1 - 2 * inv.length / nOrd : NaN, '-', { status: 'ok' }));
    out.push(L2m('L2.phase_ladder.max_dE_error', Math.max(...ladder.filter(k => k !== b).map(k => Math.abs(md[k] - dd[k]))), 'meV/atom'));
  }
  // element references and formation energies
  const els = S.filter(s => s.role === 'element' && M[s.key]); els.forEach(s => relax(s.key));
  if (els.length && ladder.length) {
    const muM = Object.fromEntries(els.map(s => [s.key, M[s.key].energy_eV_per_atom])), muD = Object.fromEntries(els.map(s => [s.key, s.energy_eV_per_atom]));
    const form = (e, comp, mu) => { const tot = sum(Object.values(comp)); return e - sum(Object.entries(comp).map(([el, n]) => n / tot * mu[el])); };
    const keys = ladder.filter(k => Object.keys(byKey[k].composition).every(el => el in muM));
    const hM = Object.fromEntries(keys.map(k => [k, form(M[k].energy_eV_per_atom, byKey[k].composition, muM)])), hD = Object.fromEntries(keys.map(k => [k, form(byKey[k].energy_eV_per_atom, byKey[k].composition, muD)]));
    for (const k of keys.slice().sort((x, y) => hD[x] - hD[y])) out.push(L2m(`L2.formation_energy.dE_error.${k}`, (hM[k] - hD[k]) * 1000, 'meV/atom', { extra: { model_meV_per_atom: hM[k] * 1000, dft_meV_per_atom: hD[k] * 1000 } }));
    if (keys.includes(base) && keys.includes(other)) { const g = (hM[other] - hM[base]) * 1000, gd = (hD[other] - hD[base]) * 1000, same = sign(g) === sign(gd);
      out.push(L2m(`L2.formation_energy.${name(other)}_minus_${name(base)}`, g, 'meV/atom', { status: same ? 'provisional' : 'failed', reason: same ? null : 'composition-corrected phase ordering inverted vs DFT',
        extra: { dft_meV_per_atom: gd, error_meV_per_atom: g - gd, ordering_reproduced: same } })); }
  }
  // EOS
  const groups = (csv) => { const g = {}; for (const r of csv.rows) { const k = r.group, m = r.member ?? ''; ((g[k] = g[k] || {})[m] = g[k][m] || { v: [], e: [] }); g[k][m].v.push(r.volume_A3_per_atom); g[k][m].e.push(r.energy_eV_per_atom); } return g; };
  const gD = groups(refEos), gM = groups(mlEos);
  for (const grp of Object.keys(gD)) {
    if (!gM[grp]) continue; const members = Object.keys(gD[grp]).filter(m => gM[grp][m]).sort();
    if (!members.length) continue;
    const curve = c => { const o = c.v.map((v, i) => [v, c.e[i]]).sort((a, b) => a[0] - b[0]); const emin = Math.min(...o.map(x => x[1])); return { v: o.map(x => +x[0].toFixed(3)), e: o.map(x => +((x[1] - emin) * 1000).toFixed(4)) }; };
    const firstMember = members.includes('s01') ? 's01' : members[0];
    plot[grp] = { dft: curve(gD[grp][firstMember]), mlip: curve(gM[grp][firstMember]) };
    if (members.length === 1 && Object.keys(gD[grp]).length === 1) {
      const fm = fitEOS(gM[grp][members[0]].v, gM[grp][members[0]].e), fd = fitEOS(gD[grp][members[0]].v, gD[grp][members[0]].e);
      out.push(L2m(`L2.eos.${grp}.V0`, fm.v0, 'A^3/atom', { extra: { fit: fm } }), L2m(`L2.eos.${grp}.B0`, fm.b0, 'GPa', { extra: { fit: fm } }));
      if (isNum(fd.v0)) out.push(L2m(`L2.eos.${grp}.V0_error`, 100 * (fm.v0 - fd.v0) / fd.v0, '%', { extra: { dft: fd, mlip: fm } }), L2m(`L2.eos.${grp}.B0_error`, 100 * (fm.b0 - fd.b0) / fd.b0, '%', { extra: { dft: fd, mlip: fm } }));
    } else {
      const fits = members.map(m => [fitEOS(gM[grp][m].v, gM[grp][m].e), fitEOS(gD[grp][m].v, gD[grp][m].e)]).filter(([a, b]) => isNum(a.v0) && isNum(b.v0) && isNum(b.b0));
      if (!fits.length) continue;
      const dv = fits.map(([a]) => a.v0), db = fits.map(([a]) => a.b0), Dv = fits.map(([, b]) => b.v0), Db = fits.map(([, b]) => b.b0);
      const ve = dv.map((x, i) => 100 * (x - Dv[i]) / Dv[i]), be = db.map((x, i) => 100 * (x - Db[i]) / Db[i]);
      out.push(L2m(`L2.eos.${grp}.V0`, mean(dv), 'A^3/atom', { uncertainty: sd1(dv) }), L2m(`L2.eos.${grp}.B0`, mean(db), 'GPa', { uncertainty: sd1(db) }),
               L2m(`L2.eos.${grp}.V0_error`, mean(ve), '%', { uncertainty: sd1(ve), extra: { n_seeds: fits.length } }), L2m(`L2.eos.${grp}.B0_error`, mean(be), '%', { uncertainty: sd1(be), extra: { n_seeds: fits.length } }),
               L2m(`L2.eos.${grp}.seed_ensemble.dft_V0_spread`, sd1(Dv), 'A^3/atom', { tier: null }), L2m(`L2.eos.${grp}.seed_ensemble.dft_B0_spread`, sd1(Db), 'GPa', { tier: null }));
      for (const [nm, err, refs] of [['V0', ve, Dv], ['B0', be, Db]]) { const rs = 100 * sd1(refs) / mean(refs);
        out.push(L2m(`L2.eos.${grp}.seed_ensemble.${nm}_error_vs_seed_spread`, isNum(rs) && rs > 0 ? Math.abs(mean(err)) / rs : NaN, 'ratio', { tier: null })); }
    }
  }
  return { metrics: out, plot };
}

/* ------------------------------------------------------------------ overlap + ranking */
function applyOverlap(metrics, ids) {
  const set = new Set(ids || []);
  for (const m of metrics) if (set.has(m.metric_id)) { m.extra.training_leakage = true; if (m.status === 'ok') m.status = 'provisional'; }
  return metrics;
}
const mAbs = a => mean(a.map(Math.abs)), mMax = a => Math.max(...a.map(Math.abs)), mMean = a => mean(a);
// [key, level, regex, field, aggregate, transform]
const IND = [
  ['force MAE', 'L0', /^L0\.force_mae$/, 'value', mAbs], ['energy MAE', 'L0', /^L0\.energy_mae$/, 'value', mAbs],
  ['stress RMSE', 'L0', /^L0\.stress_rmse$/, 'value', mAbs], ['force 1-R2', 'L0', /^L0\.force_component_r2$/, 'value', mAbs, 'one_minus'],
  ['force MAE worst stage', 'L0', /^L0\.force_rmse\.pool=/, 'value', mMax],
  ['L1 gate', 'L1', null, 'gate', mMean], ['PEC curves passing L1', 'L1', /^L1\.pec\./, 'status_ok', mMean],
  ['NVE drift', 'L1', /^L1\.nve_drift\.dt=3fs$/, 'value', mAbs], ['energy-force consistency', 'L1', /^L1\.(force_vs_dE_dx|stress_vs_dE_dstrain)$/, 'value', mAbs],
  ['PEC repulsive walls', 'L1', /^L1\.pec\./, 'extra.monotonic_repulsive', mMean], ['2000 K min distance', 'L1', /^L1\.stability_2000K\.min_distance$/, 'value', mAbs],
  ['lattice scan smoothness', 'L1', /^L1\.lattice_scan_smoothness$/, 'value', mAbs], ['timestep drift spread', 'L1', /^L1\.timestep_sensitivity\.drift_spread$/, 'value', mAbs],
  ['relaxed volume error', 'L2', /^L2\.relax\..*\.volume_error$/, 'value', mAbs], ['relaxed cell-length error', 'L2', /^L2\.relax\..*\.cell_length_error$/, 'value', mAbs],
  ['EOS V0 error', 'L2', /^L2\.eos\..*\.V0_error$/, 'value', mAbs], ['EOS B0 error', 'L2', /^L2\.eos\..*\.B0_error$/, 'value', mAbs],
  ['phase order', 'L2', /^L2\.phase_order\.cubic_minus_hexagonal$/, 'sign_positive', mAbs], ['phase ladder ordering', 'L2', /^L2\.phase_ladder\.inversions$/, 'status_ok', mAbs],
  ['phase ladder dE error', 'L2', /^L2\.phase_ladder\.dE_error\./, 'value', mAbs], ['stacking gap error', 'L2', /^L2\.phase_ladder\.dE_error\.mp3534$/, 'value', mAbs],
  ['formation energy error', 'L2', /^L2\.formation_energy\.dE_error\./, 'value', mAbs], ['corrected phase order', 'L2', /^L2\.formation_energy\.cubic_minus_hexagonal$/, 'extra.ordering_reproduced', mAbs],
];
function fieldValue(m, field) {
  if (field === 'status_ok') return m.status === 'ok' ? 1 : 0;
  if (field === 'sign_positive') return isNum(m.value) ? (m.value > 0 ? 1 : 0) : null;
  if (field.startsWith('extra.')) { const v = m.extra[field.slice(6)]; return typeof v === 'boolean' ? +v : isNum(v) ? v : null; }
  const v = m.value; return typeof v === 'boolean' ? +v : isNum(v) ? v : null;
}
function indicators(metrics, gate) {
  const raw = {};
  for (const [key, , rx, field, agg, tr] of IND) {
    let vals;
    if (field === 'gate') vals = gate ? [gate.pass ? 1 : 0] : [];
    else vals = metrics.filter(m => rx.test(m.metric_id)).map(m => fieldValue(m, field)).filter(v => v !== null);
    if (tr === 'one_minus') vals = vals.map(v => 1 - v);
    raw[key] = vals.length ? agg(vals) : null;
  }
  return raw;
}
function scoreAgainst(D, key, raw, levels) {      // add one model to the page's indicator table
  const lv = {}, rows = [];
  for (const ind of D.indicators) {
    if (!levels.includes(ind.level) || !(ind.key in raw)) continue;
    const all = Object.assign({}, ind.raw, { [key]: raw[ind.key] }), present = Object.entries(all).filter(([, v]) => v !== null && v !== undefined);
    let score = null; const x = raw[ind.key];
    if (x !== null && present.length) {
      const floor = Math.max(ind.floor, 1e-12);
      if (ind.dir === 'pass') score = 100 * Math.min(Math.max(x, 0), 1);
      else if (ind.dir === 'gap') { const best = Math.max(...present.map(([, v]) => v)); score = 100 * floor / ((best - x) + floor); }
      else if (ind.dir === 'higher') { const best = Math.max(...present.map(([, v]) => v)); score = 100 * (x + floor) / (best + floor); }
      else { const best = Math.min(...present.map(([, v]) => Math.abs(v))); score = 100 * (best + floor) / (Math.abs(x) + floor); }
    }
    rows.push({ key: ind.key, level: ind.level, raw: x, score, w: ind.w, unit: ind.unit });
  }
  for (const l of levels) { const r = rows.filter(q => q.level === l && q.score !== null); lv[l] = r.length ? sum(r.map(q => q.w * q.score)) / sum(r.map(q => q.w)) : null; }
  return { level: lv, rows };
}

/* ------------------------------------------------------------------ comparison with expected */
function compare(metrics, expected, rtol = 1e-6) {
  const by = Object.fromEntries(metrics.map(m => [m.metric_id, m])); let n = 0, agree = 0; const diffs = [];
  for (const [id, [v, status]] of Object.entries(expected)) {
    if (id.includes(':')) continue; const m = by[id]; if (!m) continue;
    if (typeof v !== 'number' && typeof v !== 'boolean' && v !== null) continue; n++;
    const tol = /^L2\.eos\./.test(id) ? 1e-4 : rtol, a = m.value;
    const same = (typeof v === 'boolean' || typeof a === 'boolean') ? Boolean(a) === Boolean(v)
      : (v === null || !isNum(v)) ? !isNum(a) : isNum(a) && Math.abs(a - v) <= tol * Math.max(Math.abs(v), 1e-12) + 1e-12;
    if (same && m.status === status) agree++; else diffs.push({ id, browser: [a, m.status], hpc: [v, status] });
  }
  return { n, agree, diffs };
}

root.MLIPEngine = { readText, parseExtxyz, parseCSV, needCols, checkL0, evalL0, evalL1, evalL2, checkL2, fitEOS,
                    applyOverlap, indicators, scoreAgainst, compare, IND_LEVELS: ['L0', 'L1', 'L2'] };
})(typeof window !== 'undefined' ? window : globalThis);
