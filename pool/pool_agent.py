#!/usr/bin/env python3
"""
pool_agent.py: the Kova Pool worker. Runs inside the pool image (or straight on a Linux box with a GPU) and joins the
pool's herd with a POOL_TOKEN from the account page on pool.kova23.com.

  1. asks the pool which public key + range + DP bits to search, and what this token is: species (tame or wild),
     mode (pooled / solo), power mode (full / gentle / idle), and for wild tokens the blind point (GET /api/worker/job)
  2. starts one `kangaroo` per GPU with ONLY that species (--species; --blind R for wild). A pool worker never resolves
     collisions itself: it cannot (one species; wild blinded). It streams every DP to the pool (POST /api/worker/dps),
     the pool forwards them into the herd's table and the pool's server derives any key.
  3. heartbeat every 15 s (POST /api/worker/heartbeat): GPU names and jumps/s for the account page. The answer carries
     the current job version and the token's mode/power, so changes made on the site reach the worker within 15 s.
  4. power modes: full = flat out; gentle = GPU power cap at POWER_CAP_PERCENT of the card's maximum when nvidia-smi
     lets us (needs root), otherwise a duty cycle (3 s running, 2 s paused, SIGSTOP/SIGCONT); idle = on Linux same as
     gentle (the "only when the PC is idle" rule is the Windows agent's job).

Only the Python standard library. Environment: POOL_URL (default https://pool.kova23.com), POOL_TOKEN (required),
KANG_ARGS (extra kangaroo options), KANG_RC=auto|cubin|cuda|off, POWER_CAP_PERCENT (60), KANG_CPU=1 (testing without
a GPU), HEARTBEAT_S (15).
"""
import collections, json, os, re, shlex, signal, subprocess, sys, threading, time, urllib.request, urllib.error

APP = os.path.dirname(os.path.abspath(__file__))
POOL_URL = os.environ.get("POOL_URL", "https://pool.kova23.com").rstrip("/")
TOKEN = os.environ.get("POOL_TOKEN", "").strip()
BIN = os.environ.get("KANG_BIN", os.path.join(APP, "kangaroo"))
CPU_MODE = os.environ.get("KANG_CPU", "0") == "1"
EXTRA = shlex.split(os.environ.get("KANG_ARGS", ""))
RC_MODE = os.environ.get("KANG_RC", "auto")
HEARTBEAT_S = float(os.environ.get("HEARTBEAT_S", "15"))
CAP_PERCENT = int(os.environ.get("POWER_CAP_PERCENT", "60"))
BATCH_MAX = 5000; BATCH_S = 2.0; QUEUE_MAX = 2_000_000
DUTY_ON, DUTY_OFF = 3.0, 2.0
T0 = time.time()

def log(*a): print(time.strftime("%H:%M:%S"), "[pool-agent]", *a, flush=True)

def call(path, payload=None, timeout=20):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(POOL_URL + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}", "User-Agent": "kova-pool-agent"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try: body = json.loads(e.read() or b"{}")
        except Exception: body = {}
        return e.code, body

st_lock = threading.Lock()
st = {"per": {}, "job": None, "found": False, "power": "full", "mode": "pooled", "sent": 0, "dropped": 0, "stale": 0, "suspended": ""}
dpq = collections.deque(); dpq_lock = threading.Lock()

def gpu_list():
    if CPU_MODE: return ["CPU (test mode)"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return [x.strip() for x in out if x.strip()] or ["?"]
    except Exception: return ["?"]

def fetch_job():
    while True:
        code, j = call("/api/worker/job")
        if code == 200 and j.get("pubkey") and j.get("range"): return j
        if code in (401, 403):
            log(f"the pool refused this token ({j.get('error', code)}); check POOL_TOKEN on your account page. Retrying in 60 s")
            with st_lock: st["suspended"] = str(j.get("error", code))
            time.sleep(60); continue
        if code == 200: log("the pool has no target yet; checking again in 30 s")
        else: log(f"could not reach the pool ({code} {j.get('error', '')}); retrying in 30 s")
        time.sleep(30)

STAT = re.compile(r"^STAT (.*)$"); DP = re.compile(r"^DP ([0-9a-f]{80}[01])$")

def reader(idx, proc):
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode(errors="replace").rstrip()
        m = DP.match(line)
        if m:
            with dpq_lock:
                dpq.append(m.group(1))
                if len(dpq) > QUEUE_MAX: dpq.popleft(); st["dropped"] += 1
            continue
        m = STAT.match(line)
        if m:
            kv = dict(x.split("=", 1) for x in m.group(1).split())
            with st_lock:
                st["per"].setdefault(idx, {}).update(jumps=float(kv.get("jumps", 0)), dps=int(kv.get("dps", 0)), kangs=int(kv.get("kangs", 0)),
                                                     loops=int(kv.get("loops", 0)), status="searching")
            continue
        if re.search(r"\b[0-9a-fA-F]{64}\b", line) and "pubkey" not in line: line = "(line hidden)"
        print(f"[gpu{idx}] {line}", flush=True)

def dp_loop():
    last = time.time()
    while True:
        with dpq_lock: n = len(dpq)
        if n < BATCH_MAX: time.sleep(0.2)
        with dpq_lock: n = len(dpq)
        if not n or (n < BATCH_MAX and time.time() - last < BATCH_S): continue
        with dpq_lock: batch = [dpq.popleft() for _ in range(min(n, BATCH_MAX))]
        with st_lock: ver = (st["job"] or {}).get("version", "")
        code, r = call("/api/worker/dps", {"version": ver, "dps": batch}, timeout=60)
        if code == 200:
            with st_lock:
                st["sent"] += len(batch)
                if r.get("dropped"): st["stale"] += len(batch)
                if r.get("found"): st["found"] = True
            last = time.time()
        elif code == 400:
            log(f"the pool rejected a DP batch ({r.get('error')}); dropping it"); last = time.time()
        else:
            with dpq_lock:
                dpq.extendleft(reversed(batch))
                while len(dpq) > QUEUE_MAX: dpq.popleft(); st["dropped"] += 1
            log(f"DP upload failed ({code} {r.get('error', '')}); {len(dpq)} queued, retrying"); time.sleep(5); last = time.time()

# ---- power modes
def power_limit(pct):
    """GPU power cap at pct% of each card's maximum. True if nvidia-smi accepted it (needs root / admin)."""
    if CPU_MODE: return False
    try:
        mx = subprocess.run(["nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.split()
        ok = True
        for i, w in enumerate(mx):
            lim = max(100, int(float(w) * pct / 100))
            r = subprocess.run(["nvidia-smi", "-i", str(i), "-pl", str(lim)], capture_output=True, text=True, timeout=30)
            ok &= r.returncode == 0
        return ok
    except Exception: return False

class Duty(threading.Thread):
    """SIGSTOP / SIGCONT duty cycle for the kangaroo processes when the power cap is not available"""
    def __init__(self, procs): super().__init__(daemon=True); self.procs = procs; self.on = True; self.stop_flag = False
    def run(self):
        while not self.stop_flag:
            time.sleep(DUTY_ON if self.on else DUTY_OFF)
            if self.stop_flag: break
            self.on = not self.on
            for p in self.procs:
                if p.poll() is None:
                    try: p.send_signal(signal.SIGCONT if self.on else signal.SIGSTOP)
                    except Exception: pass
    def end(self):
        self.stop_flag = True
        for p in self.procs:
            if p.poll() is None:
                try: p.send_signal(signal.SIGCONT)
                except Exception: pass

duty = None
def apply_power(mode, procs):
    global duty
    if duty: duty.end(); duty = None
    if mode == "full":
        if not CPU_MODE: power_limit(100)
        log("power: full"); return
    if power_limit(CAP_PERCENT): log(f"power: {mode}, GPU power cap {CAP_PERCENT}% (nvidia-smi)"); return
    duty = Duty(procs); duty.start()
    log(f"power: {mode}, duty cycle {DUTY_ON:.0f} s on / {DUTY_OFF:.0f} s off (no permission for a power cap)")

def start(job, gpus):
    procs = []
    with st_lock: st["per"] = {}; st["found"] = False
    for idx in range(len(gpus)):
        cmd = [BIN, "--pubkey", job["pubkey"], "--start", str(job.get("start") or "0"), "--range", str(job["range"]),
               "--agent", "--dpstream", "--dplocal", "0", "--report", "5", "--species", job["species"]]
        if job.get("blind"): cmd += ["--blind", job["blind"]]
        if int(job.get("dp") or 0): cmd += ["--dp", str(job["dp"])]
        cmd += ["--cpu", "--threads", "2"] if CPU_MODE else ["--gpu", str(idx)]
        if not CPU_MODE and RC_MODE != "off" and "--rc" not in EXTRA: cmd += ["--rc", RC_MODE]
        cmd += EXTRA
        log(f"GPU {idx}: " + " ".join(shlex.quote(c) for c in cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=APP)
        threading.Thread(target=reader, args=(idx, p), daemon=True).start()
        procs.append(p)
    return procs

def stop(procs):
    global duty
    if duty: duty.end(); duty = None
    for p in procs:
        if p.poll() is None: p.send_signal(signal.SIGTERM)
    for p in procs:
        try: p.wait(20)
        except subprocess.TimeoutExpired: p.kill()

def heartbeat_loop(gpus):
    while True:
        with st_lock:
            jps = sum(v.get("jumps", 0) for v in st["per"].values())
            kangs = sum(v.get("kangs", 0) for v in st["per"].values()); loops = sum(v.get("loops", 0) for v in st["per"].values())
        code, r = call("/api/worker/heartbeat", {"gpus": gpus, "jps": jps, "uptime": time.time() - T0, "kangs": kangs, "loops": loops})
        if code == 200:
            with st_lock:
                st["hb"] = r
                if r.get("found"): st["found"] = True
        elif code in (401, 403):
            with st_lock: st["suspended"] = str(r.get("error", code))
        else: log(f"heartbeat failed: {code} {r.get('error', '')}")
        time.sleep(HEARTBEAT_S)

def main():
    if len(TOKEN) < 20 or not TOKEN.startswith("pk_"):
        log("POOL_TOKEN must be set (make one on your account page at " + POOL_URL + ")"); sys.exit(2)
    if not os.path.exists(BIN) and not CPU_MODE: log(f"kangaroo binary not found at {BIN}"); sys.exit(2)
    gpus = gpu_list()
    log(f"pool {POOL_URL} | token {TOKEN[:10]}... | {len(gpus)} GPU(s): {', '.join(gpus)}")
    threading.Thread(target=heartbeat_loop, args=(gpus,), daemon=True).start()
    threading.Thread(target=dp_loop, daemon=True).start()
    job = fetch_job()
    while True:
        with st_lock: st["job"] = job; st["power"] = job.get("power", "full"); st["mode"] = job.get("mode", "pooled"); st["suspended"] = ""
        log(f"target {job['pubkey'][:20]}... range 2^{job['range']} dp {job['dp']} | species {job['species']}{' (blinded)' if job.get('blind') else ''} | "
            f"mode {job.get('mode')} | power {job.get('power')} | job {job['version']}")
        procs = start(job, gpus)
        apply_power(job.get("power", "full"), procs)
        switched = False
        while any(p.poll() is None for p in procs):
            time.sleep(2)
            with st_lock: hb = dict(st.get("hb") or {}); found = st["found"]; susp = st["suspended"]
            if susp:
                log(f"token suspended by the pool ({susp}): stopping. Fix it on your account page; retrying in 5 min")
                stop(procs); time.sleep(300); job = fetch_job(); switched = True; break
            if found:
                log("the pool reports the key is found; stopping until a new target is set")
                stop(procs); break
            if hb.get("job_version") and hb["job_version"] != job["version"]:
                log(f"target changed {job['version']} -> {hb['job_version']}: restarting")
                stop(procs); job = fetch_job(); switched = True; break
            if hb.get("power") and hb["power"] != st["power"]:
                with st_lock: st["power"] = hb["power"]
                job["power"] = hb["power"]; apply_power(hb["power"], procs)
            if hb.get("mode") and hb["mode"] != st["mode"]:
                with st_lock: st["mode"] = hb["mode"]
                log(f"mode is now {hb['mode']}")
        if switched: continue
        if st["found"]:
            while True:
                time.sleep(15)
                with st_lock: hb = dict(st.get("hb") or {})
                if hb.get("job_version") and hb["job_version"] != job["version"]: job = fetch_job(); break
            continue
        log(f"kangaroo exited (codes {[p.returncode for p in procs]}), restarting in 10 s"); time.sleep(10)

if __name__ == "__main__":
    main()
