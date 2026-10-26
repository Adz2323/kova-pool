// phone-worker.js: one Web Worker = one group of 24 kangaroos in kangaroo.wasm (kangaroo/cpu/kangaroo_wasm.cpp).
// Messages in:  {type:"start", job, species (0 tame, 1 wild, 2 both), seed}   {type:"stop"}
// Messages out: {type:"ready"} {type:"stats", jumps, loops, regen} {type:"dps", lines:[...]} {type:"error", text}
let ex = null, running = false, shareMask = 0n, nt = 24, jumps = 0;
const hex = v => v.toString(16).padStart(16, "0");
const words = (hexStr, off, mem) => { const v = BigInt("0x" + hexStr); for (let i = 0; i < 4; i++) mem[off + i] = (v >> BigInt(64 * i)) & 0xFFFFFFFFFFFFFFFFn; };
async function load() {
  const r = await fetch("/kangaroo.wasm", { cache: "no-cache" });
  const { instance } = await WebAssembly.instantiate(await r.arrayBuffer(), {});
  ex = instance.exports; postMessage({ type: "ready" });
}
function start(job, species, seed) {
  const mem = new BigUint64Array(ex.memory.buffer); const off = Number(ex.kw_buf()) / 8;
  for (let i = 0; i < 16; i++) mem[off + i] = 0n;
  words(job.pubkey.slice(2), off, mem); mem[off + 4] = job.pubkey.startsWith("03") ? 1n : 0n;
  const st = BigInt("0x" + (job.start || "0")); for (let i = 0; i < 4; i++) mem[off + 5 + i] = (st >> BigInt(64 * i)) & 0xFFFFFFFFFFFFFFFFn;
  if (species === 1 && job.blind) { words(job.blind.slice(2), off + 9, mem); mem[off + 13] = job.blind.startsWith("03") ? 1n : 0n; mem[off + 14] = 1n; }
  const cand = Number(job.cand_dp || job.dp || 32), share = Number(job.share_dp || job.dp || 32);
  shareMask = share > cand ? (1n << BigInt(share - cand)) - 1n : 0n;
  const rc = ex.kw_init(Number(job.range), cand, species, BigInt(seed[0]), BigInt(seed[1]));
  if (rc) { postMessage({ type: "error", text: "init failed (" + rc + ")" }); return; }
  nt = ex.kw_nt(); running = true; jumps = 0; loop();
}
function loop() {
  if (!running) return;
  const t0 = Date.now(); const lines = [];
  while (Date.now() - t0 < 900) {                       // ~1 s of work, then yield so a "stop" can arrive
    const n = ex.kw_run(1000); jumps += 24 * 1000;
    const mem = new BigUint64Array(ex.memory.buffer); const o0 = Number(ex.kw_out()) / 8;
    for (let i = 0; i < n; i++) {
      const o = o0 + 6 * i; const x0 = mem[o];
      if (shareMask && (x0 & shareMask)) continue;        // a candidate, not a share point
      lines.push(hex(x0) + hex(mem[o + 1]) + hex(mem[o + 4]) + hex(mem[o + 3]) + hex(mem[o + 2]) + (Number(mem[o + 5] & 0xFFFFFFFFn) < nt ? "0" : "1"));
    }
  }
  if (lines.length) postMessage({ type: "dps", lines });
  postMessage({ type: "stats", jumps, loops: Number(ex.kw_loops()), regen: Number(ex.kw_regen()) }); jumps = 0;
  setTimeout(loop, 0);
}
onmessage = e => {
  const m = e.data;
  if (m.type === "start") { if (!ex) { postMessage({ type: "error", text: "wasm not loaded" }); return; } start(m.job, m.species, m.seed); }
  else if (m.type === "stop") running = false;
};
load().catch(err => postMessage({ type: "error", text: String(err) }));
