package main

// native.go: the "native" engine. Instead of Docker Desktop, KovaPool downloads kangaroo.exe (built by the
// windows-worker GitHub Actions workflow and published on the rolling release "windows-latest") and runs it directly,
// doing itself what pool_agent.py does in the Docker image: fetch the job, run one solver per GPU, stream DPs to the
// pool in batches, heartbeat, restart on a target change, stop on a find. Pause/resume (idle-only and gentle's duty
// cycle) use NtSuspendProcess / NtResumeProcess on the solver processes.

import (
	"archive/zip"
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const (
	releaseBase = "https://github.com/Adz2323/kova-pool/releases/download/windows-latest/"
	batchMax    = 5000
	batchEvery  = 2 * time.Second
	queueMax    = 2_000_000
	heartbeatS  = 15 * time.Second
)

var buildSHA = "dev" // set by the workflow: -X main.buildSHA=<sha>

var (
	ntdll        = syscall.NewLazyDLL("ntdll.dll")
	pNtSuspend   = ntdll.NewProc("NtSuspendProcess")
	pNtResume    = ntdll.NewProc("NtResumeProcess")
	pOpenProcess = kernel32.NewProc("OpenProcess")
	pCloseHandle = kernel32.NewProc("CloseHandle")
	reDP         = regexp.MustCompile(`^DP ([0-9a-f]{80}[01])$`)
	reSTAT       = regexp.MustCompile(`^STAT (.*)$`)
	reHex64      = regexp.MustCompile(`\b[0-9a-fA-F]{64}\b`)
)

type gpuStat struct {
	jumps             float64
	dps, kangs, loops int64
}

type nativeEngine struct {
	mu       sync.Mutex
	procs    []*exec.Cmd
	per      map[int]gpuStat
	job      map[string]any
	gpus     []string
	dpq      []string
	shareMsk uint64 // low bits of x that must be zero for a share point (job share_dp - cand_dp); 0 = every DP
	dropped  int64
	sent     int64
	found    bool
	susp     string
	hb       map[string]any
	paused   bool
	running  bool
	stopping bool
	log      []string
	t0       time.Time
	dir      string // %LOCALAPPDATA%\KovaPool\solver
}

var native = &nativeEngine{per: map[int]gpuStat{}, t0: time.Now()}

// one token runs ONE kind of worker (this app is always a GPU worker); wid names this process so several PCs on
// one token are counted and shown separately by the pool
const myTier = "gpu"

var wid = func() string { b := make([]byte, 4); rand.Read(b); return hex.EncodeToString(b) }()

func (n *nativeEngine) logf(f string, a ...any) {
	n.mu.Lock()
	n.log = append(n.log, time.Now().Format("15:04:05")+" "+fmt.Sprintf(f, a...))
	if len(n.log) > 60 {
		n.log = n.log[len(n.log)-60:]
	}
	n.mu.Unlock()
}
func (n *nativeEngine) logs() string {
	n.mu.Lock()
	defer n.mu.Unlock()
	return strings.Join(n.log, "\n")
}

// ---- pool API (same endpoints as pool_agent.py)
func poolCall(path string, payload any, timeout time.Duration) (int, map[string]any) {
	var body io.Reader
	method := "GET"
	if payload != nil {
		b, _ := json.Marshal(payload)
		body = bytes.NewReader(b)
		method = "POST"
	}
	req, _ := http.NewRequest(method, strings.TrimRight(cfg.PoolURL, "/")+path, body)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.Token)
	req.Header.Set("User-Agent", "kovapool-native/"+buildSHA)
	r, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return 0, map[string]any{"error": err.Error()}
	}
	defer r.Body.Close()
	var j map[string]any
	b, _ := io.ReadAll(r.Body)
	json.Unmarshal(b, &j)
	if j == nil {
		j = map[string]any{}
	}
	return r.StatusCode, j
}

// ---- solver download: kangaroo-windows.zip + VERSION.txt from the rolling release; re-fetched when VERSION changes
func (n *nativeEngine) ensureSolver() error {
	n.dir = filepath.Join(os.Getenv("LOCALAPPDATA"), appName, "solver")
	os.MkdirAll(n.dir, 0o700)
	exe := filepath.Join(n.dir, "kangaroo.exe")
	have, _ := os.ReadFile(filepath.Join(n.dir, "VERSION.txt"))
	c := &http.Client{Timeout: 30 * time.Second}
	want := ""
	if r, err := c.Get(releaseBase + "VERSION.txt"); err == nil && r.StatusCode == 200 {
		b, _ := io.ReadAll(r.Body)
		r.Body.Close()
		want = strings.TrimSpace(string(b))
	}
	if _, err := os.Stat(exe); err == nil && (want == "" || want == strings.TrimSpace(string(have))) {
		return nil
	}
	if want == "" {
		return fmt.Errorf("the solver isn't downloaded yet and the download site can't be reached (github.com)")
	}
	n.logf("downloading the solver (kangaroo-windows.zip, build %s)…", want[:8])
	mu.Lock()
	state.Status = "downloading the solver (first time, ~40 MB)…"
	mu.Unlock()
	r, err := (&http.Client{Timeout: 10 * time.Minute}).Get(releaseBase + "kangaroo-windows.zip")
	if err != nil || r.StatusCode != 200 {
		return fmt.Errorf("solver download failed (%v)", err)
	}
	defer r.Body.Close()
	tmp := filepath.Join(n.dir, "download.zip")
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err = io.Copy(f, r.Body); err != nil {
		f.Close()
		return err
	}
	f.Close()
	z, err := zip.OpenReader(tmp)
	if err != nil {
		return fmt.Errorf("bad solver zip: %v", err)
	}
	for _, zf := range z.File {
		name := filepath.Clean(zf.Name)
		if strings.HasPrefix(name, "..") || filepath.IsAbs(name) {
			continue
		}
		dst := filepath.Join(n.dir, name)
		if zf.FileInfo().IsDir() {
			os.MkdirAll(dst, 0o700)
			continue
		}
		os.MkdirAll(filepath.Dir(dst), 0o700)
		in, err := zf.Open()
		if err != nil {
			z.Close()
			return err
		}
		out, err := os.Create(dst)
		if err != nil {
			in.Close()
			z.Close()
			return err
		}
		io.Copy(out, in)
		out.Close()
		in.Close()
	}
	z.Close()
	os.Remove(tmp)
	os.WriteFile(filepath.Join(n.dir, "VERSION.txt"), []byte(want), 0o600)
	n.logf("solver ready")
	return nil
}

func gpuNames() []string {
	out, err := run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
	if err != nil || out == "" {
		return nil
	}
	var g []string
	for _, l := range strings.Split(out, "\n") {
		if s := strings.TrimSpace(l); s != "" {
			g = append(g, s)
		}
	}
	return g
}

// ---- processes
func (n *nativeEngine) launch(job map[string]any) error {
	n.mu.Lock()
	n.per = map[int]gpuStat{}
	n.found = false
	n.procs = nil
	n.mu.Unlock()
	for idx := range n.gpus {
		args := []string{"--pubkey", str(job["pubkey"]), "--start", strOr(job["start"], "0"), "--range", fmt.Sprint(num(job["range"])),
			"--agent", "--dpstream", "--dplocal", "0", "--report", "5", "--species", str(job["species"]), "--gpu", fmt.Sprint(idx), "--rc", "auto"}
		if b := str(job["blind"]); b != "" {
			args = append(args, "--blind", b)
		}
		if dp := numOr(job["cand_dp"], num(job["dp"])); dp > 0 { // the solver runs at cand_dp (<= 32); the pool pays for share points
			args = append(args, "--dp", fmt.Sprint(dp))
		}
		cmd := exec.Command(filepath.Join(n.dir, "kangaroo.exe"), args...)
		cmd.Dir = n.dir                                                                     // rc/*.cubin are found relative to the working directory
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000} // CREATE_NO_WINDOW
		out, err := cmd.StdoutPipe()
		if err != nil {
			return err
		}
		cmd.Stderr = cmd.Stdout
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("could not start kangaroo.exe: %v", err)
		}
		n.logf("GPU %d (%s): solver started, species %s%s, dp %v", idx, n.gpus[idx], str(job["species"]), map[bool]string{true: " (blinded)", false: ""}[str(job["blind"]) != ""], job["dp"])
		n.mu.Lock()
		n.procs = append(n.procs, cmd)
		n.mu.Unlock()
		go n.reader(idx, out)
	}
	return nil
}

func (n *nativeEngine) reader(idx int, r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		if m := reDP.FindStringSubmatch(line); m != nil {
			if x0, err := strconv.ParseUint(m[1][:16], 16, 64); err == nil && x0&n.shareMsk != 0 {
				continue // a candidate, not a share point: stays on this machine
			}
			n.mu.Lock()
			if len(n.dpq) >= queueMax {
				n.dpq = n.dpq[1:]
				n.dropped++
			}
			n.dpq = append(n.dpq, m[1])
			n.mu.Unlock()
			continue
		}
		if m := reSTAT.FindStringSubmatch(line); m != nil {
			g := gpuStat{}
			for _, kv := range strings.Fields(m[1]) {
				k, v, _ := strings.Cut(kv, "=")
				switch k {
				case "jumps":
					fmt.Sscanf(v, "%g", &g.jumps)
				case "dps":
					fmt.Sscanf(v, "%d", &g.dps)
				case "kangs":
					fmt.Sscanf(v, "%d", &g.kangs)
				case "loops":
					fmt.Sscanf(v, "%d", &g.loops)
				}
			}
			n.mu.Lock()
			n.per[idx] = g
			n.mu.Unlock()
			continue
		}
		if reHex64.MatchString(line) && !strings.Contains(line, "pubkey") {
			line = "(line hidden)"
		}
		if strings.TrimSpace(line) != "" {
			n.logf("[gpu%d] %s", idx, line)
		}
	}
}

func (n *nativeEngine) alive() bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	for _, p := range n.procs {
		if p.ProcessState == nil {
			return true
		}
	}
	return false
}

func (n *nativeEngine) killAll() {
	n.mu.Lock()
	procs := n.procs
	n.mu.Unlock()
	for _, p := range procs {
		if p.Process != nil {
			n.setSuspended(p, false)
			p.Process.Kill()
		}
	}
	for _, p := range procs {
		p.Wait()
	}
	n.mu.Lock()
	n.procs = nil
	n.paused = false
	n.mu.Unlock()
}

func (n *nativeEngine) setSuspended(p *exec.Cmd, suspend bool) bool {
	if p.Process == nil {
		return false
	}
	h, _, _ := pOpenProcess.Call(0x0800, 0, uintptr(p.Process.Pid)) // PROCESS_SUSPEND_RESUME
	if h == 0 {
		return false
	}
	defer pCloseHandle.Call(h)
	if suspend {
		r, _, _ := pNtSuspend.Call(h)
		return r == 0
	}
	r, _, _ := pNtResume.Call(h)
	return r == 0
}
func (n *nativeEngine) pause() bool {
	n.mu.Lock()
	procs := n.procs
	n.mu.Unlock()
	ok := len(procs) > 0
	for _, p := range procs {
		ok = n.setSuspended(p, true) && ok
	}
	n.mu.Lock()
	n.paused = ok
	n.mu.Unlock()
	return ok
}
func (n *nativeEngine) resume() bool {
	n.mu.Lock()
	procs := n.procs
	n.mu.Unlock()
	ok := true
	for _, p := range procs {
		ok = n.setSuspended(p, false) && ok
	}
	n.mu.Lock()
	n.paused = false
	n.mu.Unlock()
	return ok
}

// ---- loops
func (n *nativeEngine) dpLoop() {
	last := time.Now()
	for {
		time.Sleep(200 * time.Millisecond)
		n.mu.Lock()
		q := len(n.dpq)
		if q == 0 || (q < batchMax && time.Since(last) < batchEvery) {
			n.mu.Unlock()
			continue
		}
		k := q
		if k > batchMax {
			k = batchMax
		}
		batch := append([]string(nil), n.dpq[:k]...)
		n.dpq = n.dpq[k:]
		ver := ""
		if n.job != nil {
			ver = str(n.job["version"])
		}
		n.mu.Unlock()
		code, r := poolCall("/api/worker/dps", map[string]any{"version": ver, "dps": batch, "tier": myTier, "wid": wid}, 60*time.Second)
		last = time.Now()
		switch {
		case code == 200:
			n.mu.Lock()
			n.sent += int64(len(batch))
			if b, _ := r["found"].(bool); b {
				n.found = true
			}
			n.mu.Unlock()
		case code == 400 || code == 409:
			n.logf("the pool rejected a DP batch (%v); dropping it", r["error"])
			if code == 409 { // the token changed tier under us: the main loop stops and re-asks
				n.mu.Lock()
				n.susp = fmt.Sprint(r["error"])
				n.mu.Unlock()
			}
		default:
			n.mu.Lock()
			n.dpq = append(batch, n.dpq...)
			for len(n.dpq) > queueMax {
				n.dpq = n.dpq[1:]
				n.dropped++
			}
			n.mu.Unlock()
			n.logf("DP upload failed (%d %v); %d queued, retrying", code, r["error"], len(batch))
			time.Sleep(5 * time.Second)
		}
	}
}

func (n *nativeEngine) heartbeatLoop() {
	for {
		n.mu.Lock()
		var jps float64
		var kangs, loops int64
		for _, g := range n.per {
			jps += g.jumps
			kangs += g.kangs
			loops += g.loops
		}
		gpus := n.gpus
		running := n.running
		n.mu.Unlock()
		if running {
			code, r := poolCall("/api/worker/heartbeat", map[string]any{"gpus": gpus, "jps": jps, "uptime": time.Since(n.t0).Seconds(), "kangs": kangs, "loops": loops, "tier": myTier, "wid": wid}, 20*time.Second)
			n.mu.Lock()
			if code == 200 {
				n.hb = r
				if b, _ := r["found"].(bool); b {
					n.found = true
				}
			} else if code == 401 || code == 403 || code == 409 {
				n.susp = fmt.Sprint(r["error"])
			}
			n.mu.Unlock()
			mu.Lock()
			state.Jps = jps
			mu.Unlock()
		}
		time.Sleep(heartbeatS)
	}
}

var loopsOnce sync.Once

func (n *nativeEngine) fetchJob() (map[string]any, bool) {
	for {
		n.mu.Lock()
		stopping := n.stopping
		n.mu.Unlock()
		if stopping {
			return nil, false
		}
		code, j := poolCall("/api/worker/job?tier="+myTier, nil, 20*time.Second)
		if code == 200 && str(j["pubkey"]) != "" && num(j["range"]) > 0 {
			return j, true
		}
		if code == 401 || code == 403 || code == 409 { // 409: the token is in use by a phone / CPU worker: it needs its own token
			setErr(fmt.Sprintf("The pool refused this token (%v). Check it on your account page.", j["error"]))
			n.logf("token refused (%v); retrying in 60 s", j["error"])
			time.Sleep(60 * time.Second)
			continue
		}
		if code == 200 {
			n.logf("the pool has no target yet; checking again in 30 s")
		} else {
			n.logf("could not reach the pool (%d %v); retrying in 30 s", code, j["error"])
		}
		time.Sleep(30 * time.Second)
	}
}

// start: called from startWorker when the engine is native. Runs the agent loop in the background.
func (n *nativeEngine) start() {
	if err := n.ensureSolver(); err != nil {
		setErr(err.Error())
		return
	}
	n.gpus = gpuNames()
	if len(n.gpus) == 0 {
		setErr("No NVIDIA GPU found (nvidia-smi missing). Install the current NVIDIA driver from nvidia.com/drivers, then click Start.")
		return
	}
	loopsOnce.Do(func() { go n.dpLoop(); go n.heartbeatLoop() })
	n.mu.Lock()
	n.running = true
	n.stopping = false
	n.susp = ""
	n.mu.Unlock()
	mu.Lock()
	state.Running = true
	state.Paused = false
	state.Status = "worker running"
	state.LastErr = ""
	mu.Unlock()
	cfg.Enabled = true
	saveConfig()
	go n.agent()
}

func (n *nativeEngine) agent() {
	defer func() { n.mu.Lock(); n.running = false; n.mu.Unlock() }()
	job, ok := n.fetchJob()
	if !ok {
		return
	}
	for {
		n.mu.Lock()
		n.job = job
		if extra := numOr(job["share_dp"], num(job["dp"])) - numOr(job["cand_dp"], num(job["dp"])); extra > 0 {
			n.shareMsk = (uint64(1) << uint(extra)) - 1
		} else {
			n.shareMsk = 0
		}
		n.susp = ""
		n.hb = nil
		n.mu.Unlock()
		n.logf("target %s… range 2^%v dp %v | job %s | mode %s | power %s", str(job["pubkey"])[:20], job["range"], job["dp"], str(job["version"]), str(job["mode"]), str(job["power"]))
		mu.Lock()
		state.Species = str(job["species"])
		state.Mode = str(job["mode"])
		state.Power = str(job["power"])
		mu.Unlock()
		if err := n.launch(job); err != nil {
			setErr(err.Error())
			n.killAll()
			return
		}
		switched := false
		for n.alive() {
			time.Sleep(2 * time.Second)
			n.mu.Lock()
			hb := n.hb
			found := n.found
			susp := n.susp
			stopping := n.stopping
			n.mu.Unlock()
			if stopping {
				n.killAll()
				return
			}
			if susp != "" {
				n.logf("token suspended by the pool (%s): stopping; fix it on your account page", susp)
				n.killAll()
				setErr("Token suspended by the pool: " + susp)
				return
			}
			if found {
				n.logf("the pool reports the key is found; stopping until a new target is set")
				n.killAll()
				mu.Lock()
				state.Status = "key found: waiting for a new target"
				mu.Unlock()
				for {
					time.Sleep(15 * time.Second)
					n.mu.Lock()
					hb2 := n.hb
					stopping := n.stopping
					n.mu.Unlock()
					if stopping {
						return
					}
					if v := str(hb2["job_version"]); v != "" && v != str(job["version"]) {
						break
					}
				}
				job, ok = n.fetchJob()
				if !ok {
					return
				}
				switched = true
				break
			}
			if v := str(hb["job_version"]); v != "" && v != str(job["version"]) {
				n.logf("target changed %s -> %s: restarting", str(job["version"]), v)
				n.killAll()
				job, ok = n.fetchJob()
				if !ok {
					return
				}
				switched = true
				break
			}
			if b, _ := hb["restart"].(bool); b { // the console's Restart button, passed through by the pool
				n.logf("the pool asked this worker to restart: restarting the solver")
				n.mu.Lock()
				n.hb = nil
				n.mu.Unlock()
				n.killAll()
				job, ok = n.fetchJob()
				if !ok {
					return
				}
				switched = true
				break
			}
			if p := str(hb["power"]); p != "" && p != str(job["power"]) {
				job["power"] = p
				mu.Lock()
				state.Power = p
				mu.Unlock()
			}
			if m := str(hb["mode"]); m != "" && m != str(job["mode"]) {
				job["mode"] = m
				mu.Lock()
				state.Mode = m
				mu.Unlock()
			}
		}
		if switched {
			continue
		}
		n.mu.Lock()
		stopping := n.stopping
		n.mu.Unlock()
		if stopping {
			return
		}
		n.logf("the solver exited; restarting in 10 s (log above says why)")
		n.killAll()
		time.Sleep(10 * time.Second)
	}
}

func (n *nativeEngine) stop() {
	n.mu.Lock()
	n.stopping = true
	n.mu.Unlock()
	n.killAll()
	n.mu.Lock()
	n.running = false
	n.mu.Unlock()
}
func (n *nativeEngine) isRunning() bool { n.mu.Lock(); defer n.mu.Unlock(); return n.running }
func (n *nativeEngine) isPaused() bool  { n.mu.Lock(); defer n.mu.Unlock(); return n.paused }

// small helpers for the loosely typed job JSON
func str(v any) string { s, _ := v.(string); return s }
func strOr(v any, d string) string {
	if s := str(v); s != "" {
		return s
	}
	return d
}
func numOr(v any, d int64) int64 {
	if v == nil {
		return d
	}
	return num(v)
}

func num(v any) int64 {
	switch t := v.(type) {
	case float64:
		return int64(t)
	case int:
		return int64(t)
	case string:
		var x int64
		fmt.Sscanf(t, "%d", &x)
		return x
	}
	return 0
}

var _ = unsafe.Pointer(nil)
