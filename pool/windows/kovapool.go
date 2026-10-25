//go:build windows

// KovaPool for Windows: a tray app that runs the pool worker (the adz23/kova-pool Docker image) through Docker Desktop
// and adds what Windows needs: a token setup page, start / stop, "idle only" (pauses the moment you touch the PC,
// resumes after N idle minutes), the gentle power cap, and start-with-Windows. Standard library only; the tray is
// plain Win32 (Shell_NotifyIcon). Settings page: http://127.0.0.1:47716 (opens in the browser).
//
// Build (from Linux or anywhere): GOOS=windows GOARCH=amd64 go build -ldflags "-H windowsgui -s -w" -o KovaPool.exe .
// Needs on the PC: NVIDIA driver, Docker Desktop with the WSL 2 backend (GPU support comes with it).
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const (
	appName     = "KovaPool"
	container   = "kovapool"
	image       = "adz23/kova-pool:5"
	settings    = "127.0.0.1:47716"
	defaultPool = "https://pool.kova23.com"
)

// ------------------------------------------------------------------ config
type Config struct {
	Token       string `json:"token"`
	PoolURL     string `json:"pool_url"`
	IdleMinutes int    `json:"idle_minutes"`
	Autostart   bool   `json:"autostart"`
	Enabled     bool   `json:"enabled"` // start the worker when the app starts
	Engine      string `json:"engine"`  // "native" (kangaroo.exe, default) or "docker" (Docker Desktop)
}

func nativeMode() bool { return cfg.Engine != "docker" }

var (
	cfg     Config
	cfgPath string
	mu      sync.Mutex
	state   = struct {
		Docker, GPU, Running, Paused bool
		Status, Species, Mode, Power string
		Engine                       string
		LastErr                      string
		Jps                          float64
		Log                          string
	}{}
	pauseReason string
)

func loadConfig() {
	dir := filepath.Join(os.Getenv("APPDATA"), appName)
	os.MkdirAll(dir, 0o700)
	cfgPath = filepath.Join(dir, "config.json")
	cfg = Config{PoolURL: defaultPool, IdleMinutes: 5, Enabled: true}
	if b, err := os.ReadFile(cfgPath); err == nil {
		json.Unmarshal(b, &cfg)
	}
	if cfg.PoolURL == "" {
		cfg.PoolURL = defaultPool
	}
	if cfg.Engine == "" {
		cfg.Engine = "native"
	}
}
func saveConfig() {
	b, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(cfgPath, b, 0o600)
}

// ------------------------------------------------------------------ win32
var (
	user32   = syscall.NewLazyDLL("user32.dll")
	shell32  = syscall.NewLazyDLL("shell32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	advapi32 = syscall.NewLazyDLL("advapi32.dll")

	pRegisterClassEx     = user32.NewProc("RegisterClassExW")
	pCreateWindowEx      = user32.NewProc("CreateWindowExW")
	pDefWindowProc       = user32.NewProc("DefWindowProcW")
	pGetMessage          = user32.NewProc("GetMessageW")
	pTranslateMessage    = user32.NewProc("TranslateMessage")
	pDispatchMessage     = user32.NewProc("DispatchMessageW")
	pPostQuitMessage     = user32.NewProc("PostQuitMessage")
	pCreatePopupMenu     = user32.NewProc("CreatePopupMenu")
	pAppendMenu          = user32.NewProc("AppendMenuW")
	pTrackPopupMenu      = user32.NewProc("TrackPopupMenu")
	pGetCursorPos        = user32.NewProc("GetCursorPos")
	pSetForegroundWindow = user32.NewProc("SetForegroundWindow")
	pDestroyMenu         = user32.NewProc("DestroyMenu")
	pGetLastInputInfo    = user32.NewProc("GetLastInputInfo")
	pCreateIconFromRes   = user32.NewProc("CreateIconFromResourceEx")
	pMessageBox          = user32.NewProc("MessageBoxW")
	pShellNotifyIcon     = shell32.NewProc("Shell_NotifyIconW")
	pShellExecute        = shell32.NewProc("ShellExecuteW")
	pGetModuleHandle     = kernel32.NewProc("GetModuleHandleW")
	pGetTickCount        = kernel32.NewProc("GetTickCount")
	pCreateMutex         = kernel32.NewProc("CreateMutexW")
	pRegCreateKeyEx      = advapi32.NewProc("RegCreateKeyExW")
	pRegSetValueEx       = advapi32.NewProc("RegSetValueExW")
	pRegDeleteValue      = advapi32.NewProc("RegDeleteValueW")
	pRegCloseKey         = advapi32.NewProc("RegCloseKey")
)

const (
	wmDestroy       = 0x0002
	wmCommand       = 0x0111
	wmApp           = 0x8000
	wmTray          = wmApp + 1
	wmLButtonDblClk = 0x0203
	wmRButtonUp     = 0x0205
	nimAdd          = 0
	nimModify       = 1
	nimDelete       = 2
	nifMessage      = 1
	nifIcon         = 2
	nifTip          = 4
	mfString        = 0
	mfSeparator     = 0x800
	mfGrayed        = 1
	tpmReturnCmd    = 0x100
	tpmBottomAlign  = 0x20
	idOpen          = 1
	idPause         = 2
	idStart         = 3
	idQuit          = 4
	idSite          = 5
)

type wndClassEx struct {
	Size, Style                        uint32
	WndProc                            uintptr
	ClsExtra, WndExtra                 int32
	Instance, Icon, Cursor, Background syscall.Handle
	MenuName, ClassName                *uint16
	IconSm                             syscall.Handle
}
type point struct{ X, Y int32 }
type msg struct {
	Hwnd           syscall.Handle
	Message        uint32
	WParam, LParam uintptr
	Time           uint32
	Pt             point
}
type notifyIconData struct {
	Size                       uint32
	Hwnd                       syscall.Handle
	ID, Flags, CallbackMessage uint32
	Icon                       syscall.Handle
	Tip                        [128]uint16
	State, StateMask           uint32
	Info                       [256]uint16
	Version                    uint32
	InfoTitle                  [64]uint16
	InfoFlags                  uint32
	GuidItem                   [16]byte
	BalloonIcon                syscall.Handle
}
type lastInputInfo struct{ Size, Time uint32 }

func utf16(s string) *uint16 { p, _ := syscall.UTF16PtrFromString(s); return p }

var (
	hwnd  syscall.Handle
	nid   notifyIconData
	hIcon syscall.Handle
)

func setTip(s string) {
	if len(s) > 120 {
		s = s[:120]
	}
	u, _ := syscall.UTF16FromString(s)
	for i := range nid.Tip {
		nid.Tip[i] = 0
	}
	copy(nid.Tip[:], u)
	pShellNotifyIcon.Call(nimModify, uintptr(unsafe.Pointer(&nid)))
}

func wndProc(h syscall.Handle, m uint32, wp, lp uintptr) uintptr {
	switch m {
	case wmTray:
		switch lp {
		case wmLButtonDblClk:
			openSettings()
		case wmRButtonUp:
			showMenu()
		}
		return 0
	case wmCommand:
		switch wp & 0xFFFF {
		case idOpen:
			openSettings()
		case idSite:
			shellOpen(cfg.PoolURL + "/#/account")
		case idPause:
			go func() {
				mu.Lock()
				if state.Paused {
					resume("menu")
				} else {
					pause("paused from the menu", true)
				}
				mu.Unlock()
			}()
		case idStart:
			go func() {
				if isRunning() {
					stopWorker()
				} else {
					startWorker()
				}
			}()
		case idQuit:
			pShellNotifyIcon.Call(nimDelete, uintptr(unsafe.Pointer(&nid)))
			pPostQuitMessage.Call(0)
		}
		return 0
	case wmDestroy:
		pShellNotifyIcon.Call(nimDelete, uintptr(unsafe.Pointer(&nid)))
		pPostQuitMessage.Call(0)
		return 0
	}
	r, _, _ := pDefWindowProc.Call(uintptr(h), uintptr(m), wp, lp)
	return r
}

func showMenu() {
	menu, _, _ := pCreatePopupMenu.Call()
	add := func(id uintptr, text string, flags uintptr) {
		pAppendMenu.Call(menu, mfString|flags, id, uintptr(unsafe.Pointer(utf16(text))))
	}
	mu.Lock()
	running, paused, st := state.Running, state.Paused, state.Status
	mu.Unlock()
	add(0, st, mfGrayed)
	pAppendMenu.Call(menu, mfSeparator, 0, 0)
	add(idOpen, "Open settings", 0)
	add(idSite, "My account on pool.kova23.com", 0)
	if running {
		if paused {
			add(idPause, "Resume now", 0)
		} else {
			add(idPause, "Pause", 0)
		}
		add(idStart, "Stop worker", 0)
	} else {
		add(idStart, "Start worker", 0)
	}
	pAppendMenu.Call(menu, mfSeparator, 0, 0)
	add(idQuit, "Quit KovaPool (stops the worker)", 0)
	var pt point
	pGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	pSetForegroundWindow.Call(uintptr(hwnd))
	pTrackPopupMenu.Call(menu, tpmBottomAlign, uintptr(pt.X), uintptr(pt.Y), 0, uintptr(hwnd), 0)
	pDestroyMenu.Call(menu)
}

func shellOpen(url string) {
	pShellExecute.Call(0, uintptr(unsafe.Pointer(utf16("open"))), uintptr(unsafe.Pointer(utf16(url))), 0, 0, 1)
}
func openSettings() { shellOpen("http://" + settings + "/") }
func msgBox(text, title string) {
	pMessageBox.Call(0, uintptr(unsafe.Pointer(utf16(text))), uintptr(unsafe.Pointer(utf16(title))), 0x40)
}

func idleSeconds() float64 {
	var lii lastInputInfo
	lii.Size = uint32(unsafe.Sizeof(lii))
	pGetLastInputInfo.Call(uintptr(unsafe.Pointer(&lii)))
	tick, _, _ := pGetTickCount.Call()
	return float64(uint32(tick)-lii.Time) / 1000
}

// a 32x32 icon: teal disc with an amber dot (the pool's mark), as a BMP-in-ICO resource built at start-up
func makeIcon() syscall.Handle {
	const n = 32
	var b bytes.Buffer
	// BITMAPINFOHEADER for a 32-bit ARGB icon image (height doubled for the AND mask)
	hdr := []uint32{40, n, n * 2, 1 | 32<<16, 0, n * n * 4, 0, 0, 0, 0}
	for _, v := range hdr {
		b.Write([]byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)})
	}
	for y := n - 1; y >= 0; y-- {
		for x := 0; x < n; x++ {
			dx, dy := float64(x)+0.5-16, float64(y)+0.5-16
			d := dx*dx + dy*dy
			var r, g, bl, a byte
			switch {
			case d <= 25:
				r, g, bl, a = 0xE2, 0xA4, 0x1E, 255 // amber dot
			case d <= 196:
				r, g, bl, a = 0x0F, 0x7C, 0x78, 255 // teal disc
			}
			b.Write([]byte{bl, g, r, a})
		}
	}
	b.Write(make([]byte, n*n/8)) // AND mask
	h, _, _ := pCreateIconFromRes.Call(uintptr(unsafe.Pointer(&b.Bytes()[0])), uintptr(b.Len()), 1, 0x30000, n, n, 0)
	return syscall.Handle(h)
}

func trayMain() {
	inst, _, _ := pGetModuleHandle.Call(0)
	cls := utf16("KovaPoolTray")
	wc := wndClassEx{Size: uint32(unsafe.Sizeof(wndClassEx{})), WndProc: syscall.NewCallback(wndProc), Instance: syscall.Handle(inst), ClassName: cls}
	pRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))
	h, _, _ := pCreateWindowEx.Call(0, uintptr(unsafe.Pointer(cls)), uintptr(unsafe.Pointer(utf16(appName))), 0, 0, 0, 0, 0, 0, 0, inst, 0)
	hwnd = syscall.Handle(h)
	hIcon = makeIcon()
	nid = notifyIconData{Size: uint32(unsafe.Sizeof(nid)), Hwnd: hwnd, ID: 1, Flags: nifMessage | nifIcon | nifTip, CallbackMessage: wmTray, Icon: hIcon}
	u, _ := syscall.UTF16FromString("KovaPool: starting")
	copy(nid.Tip[:], u)
	pShellNotifyIcon.Call(nimAdd, uintptr(unsafe.Pointer(&nid)))
	var m msg
	for {
		r, _, _ := pGetMessage.Call(uintptr(unsafe.Pointer(&m)), 0, 0, 0)
		if int32(r) <= 0 {
			break
		}
		pTranslateMessage.Call(uintptr(unsafe.Pointer(&m)))
		pDispatchMessage.Call(uintptr(unsafe.Pointer(&m)))
	}
}

// ------------------------------------------------------------------ docker
func run(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}
func dockerOK() bool {
	_, err := run("docker", "version", "--format", "{{.Server.Version}}")
	return err == nil
}
func gpuOK() bool { _, err := run("nvidia-smi", "-L"); return err == nil }
func isRunning() bool {
	if nativeMode() {
		return native.isRunning()
	}
	out, err := run("docker", "inspect", "-f", "{{.State.Status}}", container)
	return err == nil && (out == "running" || out == "paused")
}
func isPaused() bool {
	if nativeMode() {
		return native.isPaused()
	}
	out, _ := run("docker", "inspect", "-f", "{{.State.Status}}", container)
	return out == "paused"
}

func startWorker() {
	mu.Lock()
	state.LastErr = ""
	state.Status = "starting the worker…"
	mu.Unlock()
	if cfg.Token == "" {
		setErr("No token yet: open settings and paste the token from your account page.")
		return
	}
	if nativeMode() {
		native.start()
		return
	}
	if !dockerOK() {
		setErr("Docker Desktop is not running. Start it (or install it), then click Start.")
		return
	}
	run("docker", "rm", "-f", container)
	mu.Lock()
	state.Status = "downloading the worker image (first time: ~2 GB)…"
	mu.Unlock()
	setTip("KovaPool: downloading the worker image")
	if out, err := run("docker", "pull", image); err != nil {
		setErr("Could not download the worker image: " + last(out))
		return
	}
	out, err := run("docker", "run", "-d", "--name", container, "--gpus", "all", "--restart", "unless-stopped", "--cap-add", "SYS_ADMIN",
		"-e", "POOL_TOKEN="+cfg.Token, "-e", "POOL_URL="+cfg.PoolURL, image)
	if err != nil {
		if strings.Contains(out, "could not select device driver") || strings.Contains(out, "nvidia") {
			setErr("Docker can't see the GPU. Make sure Docker Desktop uses the WSL 2 backend and the NVIDIA driver is current. Details: " + last(out))
		} else {
			setErr("Could not start the worker: " + last(out))
		}
		return
	}
	cfg.Enabled = true
	saveConfig()
	mu.Lock()
	state.Running = true
	state.Paused = false
	state.Status = "worker running"
	mu.Unlock()
}
func stopWorker() {
	if nativeMode() {
		native.stop()
	} else {
		run("docker", "rm", "-f", container)
	}
	cfg.Enabled = false
	saveConfig()
	mu.Lock()
	state.Running, state.Paused = false, false
	state.Status = "worker stopped"
	mu.Unlock()
	setTip("KovaPool: stopped")
}
func pause(reason string, manual bool) { // mu held
	if !state.Running || state.Paused {
		return
	}
	ok := false
	if nativeMode() {
		ok = native.pause()
	} else {
		_, err := run("docker", "pause", container)
		ok = err == nil
	}
	if ok {
		state.Paused = true
		pauseReason = reason
		if manual {
			pauseReason = "manual"
		}
		state.Status = "paused: " + reason
		setTip("KovaPool: " + state.Status)
	}
}
func resume(reason string) { // mu held
	if !state.Paused {
		return
	}
	ok := false
	if nativeMode() {
		ok = native.resume()
	} else {
		_, err := run("docker", "unpause", container)
		ok = err == nil
	}
	if ok {
		state.Paused = false
		pauseReason = ""
		state.Status = "worker running"
		setTip("KovaPool: running")
	}
}
func setErr(e string) {
	mu.Lock()
	state.LastErr = e
	state.Status = "not running"
	state.Running = false
	mu.Unlock()
	setTip("KovaPool: " + e)
}
func last(s string) string {
	l := strings.Split(strings.TrimSpace(s), "\n")
	return l[len(l)-1]
}

// power cap for "gentle": nvidia-smi -pl needs admin; returns false if refused (then we fall back to pausing cycles)
func powerCap(pct int) bool {
	out, err := run("nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits")
	if err != nil {
		return false
	}
	ok := true
	for i, w := range strings.Fields(out) {
		var mx float64
		fmt.Sscanf(w, "%f", &mx)
		lim := int(mx * float64(pct) / 100)
		if lim < 100 {
			lim = 100
		}
		if _, err := run("nvidia-smi", "-i", fmt.Sprint(i), "-pl", fmt.Sprint(lim)); err != nil {
			ok = false
		}
	}
	return ok
}

// ------------------------------------------------------------------ pool
func poolJob() (map[string]any, error) {
	req, _ := http.NewRequest("GET", strings.TrimRight(cfg.PoolURL, "/")+"/api/worker/job", nil)
	req.Header.Set("Authorization", "Bearer "+cfg.Token)
	c := &http.Client{Timeout: 15 * time.Second}
	r, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	var j map[string]any
	b, _ := io.ReadAll(r.Body)
	json.Unmarshal(b, &j)
	if r.StatusCode != 200 {
		return nil, fmt.Errorf("%v", j["error"])
	}
	return j, nil
}

// the supervisor: keeps state fresh, applies the power mode from the pool, handles idle-only
func supervise() {
	capped := false
	gentleTick := 0
	for {
		running := isRunning()
		mu.Lock()
		state.Engine = cfg.Engine
		state.Docker = !nativeMode() && dockerOK()
		state.GPU = gpuOK()
		state.Running = running
		if running {
			state.Paused = isPaused()
			if !state.Paused && state.Status != "worker running" && !strings.HasPrefix(state.Status, "paused") {
				state.Status = "worker running"
			}
		} else if state.LastErr == "" {
			state.Status = "not running"
		}
		mu.Unlock()
		if cfg.Token != "" {
			if j, err := poolJob(); err == nil {
				mu.Lock()
				state.Species, _ = j["species"].(string)
				state.Mode, _ = j["mode"].(string)
				state.Power, _ = j["power"].(string)
				mu.Unlock()
			} else {
				mu.Lock()
				state.LastErr = "pool: " + err.Error()
				mu.Unlock()
			}
		}
		if running {
			var out string
			if nativeMode() {
				out = native.logs()
			} else {
				out, _ = run("docker", "logs", "--tail", "25", container)
			}
			mu.Lock()
			state.Log = out
			power := state.Power
			switch power {
			case "idle":
				idle := idleSeconds()
				if idle < float64(cfg.IdleMinutes)*60 {
					if !state.Paused {
						pause(fmt.Sprintf("you're using the PC (resumes after %d idle minutes)", cfg.IdleMinutes), false)
					}
				} else if state.Paused && pauseReason != "manual" {
					resume("idle")
				}
			case "gentle":
				if !capped {
					capped = powerCap(60)
				}
				if !capped { // no admin rights for the power cap: 3 s running / 2 s paused
					gentleTick++
					if gentleTick%5 < 3 {
						if state.Paused && pauseReason != "manual" {
							resume("gentle")
						}
					} else if !state.Paused {
						pause("gentle mode (duty cycle; run KovaPool as administrator for a real power cap)", false)
					}
				}
			default:
				if capped {
					powerCap(100)
					capped = false
				}
				if state.Paused && pauseReason != "manual" {
					resume("full")
				}
			}
			if !state.Paused {
				setTip(fmt.Sprintf("KovaPool: running (%s, %s, %s)", state.Species, state.Mode, state.Power))
			}
			mu.Unlock()
		}
		time.Sleep(time.Second)
	}
}

// ------------------------------------------------------------------ autostart (registry Run key)
func setAutostart(on bool) {
	var key syscall.Handle
	sub := utf16(`Software\Microsoft\Windows\CurrentVersion\Run`)
	pRegCreateKeyEx.Call(0x80000001, uintptr(unsafe.Pointer(sub)), 0, 0, 0, 0xF003F, 0, uintptr(unsafe.Pointer(&key)), 0)
	defer pRegCloseKey.Call(uintptr(key))
	name := utf16(appName)
	if on {
		exe, _ := os.Executable()
		val, _ := syscall.UTF16FromString(`"` + exe + `"`)
		pRegSetValueEx.Call(uintptr(key), uintptr(unsafe.Pointer(name)), 0, 1, uintptr(unsafe.Pointer(&val[0])), uintptr(len(val)*2))
	} else {
		pRegDeleteValue.Call(uintptr(key), uintptr(unsafe.Pointer(name)))
	}
}

// ------------------------------------------------------------------ settings page
const page = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KovaPool</title>
<style>body{font:15px/1.5 system-ui,Segoe UI,sans-serif;max-width:720px;margin:2rem auto;padding:0 16px;color:#172029;background:#EDF0F4}
h1{font-size:1.6rem}label{display:block;font-weight:600;margin:1rem 0 .3rem}input[type=text],input[type=number]{width:100%;padding:.55rem;border:1px solid #D6DCE3;border-radius:6px;font:inherit}
button{background:#0F7C78;color:#fff;border:0;border-radius:6px;padding:.6rem 1rem;font:inherit;font-weight:700;cursor:pointer;margin-right:.5rem}button.quiet{background:#fff;color:#172029;border:1px solid #D6DCE3}
.box{background:#fff;border:1px solid #D6DCE3;border-radius:6px;padding:1rem 1.2rem;margin:1rem 0}.bad{background:#F7DEDC;color:#B3261E;padding:.6rem .8rem;border-radius:6px;font-weight:600}
.ok{color:#2B7A3D;font-weight:700}.no{color:#B3261E;font-weight:700}pre{background:#172029;color:#EEF2F5;padding:.8rem;border-radius:6px;font-size:12px;max-height:280px;overflow:auto;white-space:pre-wrap}
.pill{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.85rem;font-weight:700;background:#D9EEEC;color:#0F7C78}.pill.wild{background:#F8EACB;color:#C98A12}</style>
<h1>KovaPool</h1>
<div class="box" id="status">Loading…</div>
<form class="box" method="post" action="/save">
<label>Pool token <span style="font-weight:400;color:#7A858F">(from your account page on pool.kova23.com → Connect a GPU)</span></label>
<input type="text" name="token" value="{{TOKEN}}" placeholder="pk_…" autocomplete="off" spellcheck="false">
<label>Idle only: start after the PC has been idle for</label><input type="number" name="idle" value="{{IDLE}}" min="1" max="120" style="width:6rem"> minutes
<label><input type="checkbox" name="autostart" {{AUTO}}> Start KovaPool when Windows starts</label>
<label>Engine</label>
<label style="font-weight:400"><input type="radio" name="engine" value="native" {{ENG_N}}> Built-in solver (recommended: no Docker needed; downloads kangaroo.exe once, ~40 MB)</label>
<label style="font-weight:400"><input type="radio" name="engine" value="docker" {{ENG_D}}> Docker Desktop (runs the same adz23/kova-pool image as Linux)</label>
<p><button>Save</button> <button class="quiet" formaction="/start">Save and start worker</button> <button class="quiet" formaction="/stop">Stop worker</button></p>
<p style="color:#7A858F;font-size:.9rem">Mode (pooled / solo) and power (full / gentle / idle) are set on your account page; the worker picks changes up within 15 seconds. Gentle's power cap needs KovaPool run as administrator; otherwise it alternates 3 s running, 2 s paused.</p>
</form>
<div class="box"><h3 style="margin-top:0">Worker log</h3><pre id="log">…</pre></div>
<script>
async function tick(){const r=await fetch('/state');const s=await r.json();
document.getElementById('status').innerHTML=(s.LastErr?'<div class="bad">'+s.LastErr+'</div>':'')+
(s.Engine==='docker'?'<p>Docker Desktop: <span class="'+(s.Docker?'ok">running':'no">not running')+'</span> · ':'<p>Engine: built-in solver · ')+'NVIDIA GPU: <span class="'+(s.GPU?'ok">found':'no">not found')+'</span></p>'+
'<p>Worker: <b>'+s.Status+'</b>'+(s.Species?' · species <span class="pill '+s.Species+'">'+s.Species+'</span> · mode '+s.Mode+' · power '+s.Power:'')+'</p>'+
(s.Engine==='docker'&&!s.Docker?'<p>Install <a href="https://www.docker.com/products/docker-desktop/">Docker Desktop</a> (keep the WSL 2 backend it suggests) and the current <a href="https://www.nvidia.com/drivers">NVIDIA driver</a>, then come back here and click Start.</p>':'')+
(!s.GPU?'<p>No NVIDIA GPU seen: install the current <a href="https://www.nvidia.com/drivers">NVIDIA driver</a>.</p>':'');
document.getElementById('log').textContent=s.Log||'(no output yet)';}
tick();setInterval(tick,3000);
</script>`

func settingsServer() {
	ln, err := net.Listen("tcp", settings)
	if err != nil {
		return
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		auto := ""
		if cfg.Autostart {
			auto = "checked"
		}
		en, ed := "checked", ""
		if !nativeMode() {
			en, ed = "", "checked"
		}
		p := strings.NewReplacer("{{TOKEN}}", cfg.Token, "{{IDLE}}", fmt.Sprint(cfg.IdleMinutes), "{{AUTO}}", auto, "{{ENG_N}}", en, "{{ENG_D}}", ed).Replace(page)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		io.WriteString(w, p)
	})
	mux.HandleFunc("/state", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		b, _ := json.Marshal(state)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Write(b)
	})
	save := func(r *http.Request) {
		r.ParseForm()
		cfg.Token = strings.TrimSpace(r.FormValue("token"))
		fmt.Sscanf(r.FormValue("idle"), "%d", &cfg.IdleMinutes)
		if cfg.IdleMinutes < 1 {
			cfg.IdleMinutes = 5
		}
		cfg.Autostart = r.FormValue("autostart") != ""
		if e := r.FormValue("engine"); e == "docker" || e == "native" {
			if e != cfg.Engine && isRunning() {
				stopWorker()
			}
			cfg.Engine = e
		}
		setAutostart(cfg.Autostart)
		saveConfig()
	}
	mux.HandleFunc("/save", func(w http.ResponseWriter, r *http.Request) { save(r); http.Redirect(w, r, "/", 303) })
	mux.HandleFunc("/start", func(w http.ResponseWriter, r *http.Request) { save(r); go startWorker(); http.Redirect(w, r, "/", 303) })
	mux.HandleFunc("/stop", func(w http.ResponseWriter, r *http.Request) { go stopWorker(); http.Redirect(w, r, "/", 303) })
	http.Serve(ln, mux)
}

func main() {
	runtime.LockOSThread() // the tray's message loop must stay on the main thread
	h, _, e := pCreateMutex.Call(0, 0, uintptr(unsafe.Pointer(utf16("Local\\KovaPoolSingleInstance"))))
	if h == 0 || e == syscall.Errno(183) { // ERROR_ALREADY_EXISTS: just show the running instance's page
		openSettings()
		return
	}
	loadConfig()
	go settingsServer()
	go supervise()
	if cfg.Token == "" {
		go func() { time.Sleep(600 * time.Millisecond); openSettings() }()
	} else if cfg.Enabled {
		go startWorker()
	}
	trayMain()
	if nativeMode() {
		native.stop()
	} else if isRunning() {
		run("docker", "rm", "-f", container)
	}
}
