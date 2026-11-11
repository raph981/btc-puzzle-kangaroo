#!/usr/bin/env python3
"""Local web interface for the kangaroo solver.

    python3 server.py            then open http://127.0.0.1:8070

Uses only the Python standard library. The server starts the `kangaroo`
binary as a child process and relays its JSON progress to the browser.
"""
import argparse
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(ROOT, "kangaroo")
HEX = re.compile(r"^(0x)?[0-9a-fA-F]+$")
ADDR = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{25,40}$")

with open(os.path.join(ROOT, "puzzles.json")) as f:
    PUZZLES = json.load(f)


def ensure_binary():
    if not os.path.exists(BINARY):
        print("building the solver (make) ...")
        subprocess.run(["make", "-C", ROOT], check=True)


def solver_info():
    try:
        out = subprocess.run([BINARY, "--info"], capture_output=True, text=True, timeout=20).stdout
        return json.loads(out)
    except Exception as e:  # noqa: BLE001 - shown in the UI
        return {"cuda_build": False, "threads": os.cpu_count() or 1, "devices": [], "error": str(e)}


class Job:
    """At most one solver process at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.run_id = 0
        self.state = {"status": "idle", "log": [], "history": []}

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self, req):
        brute = bool(req.get("brute"))
        target = {"brute": brute}
        if "puzzle" in req:
            p = next((p for p in PUZZLES if p["bits"] == req["puzzle"]), None)
            if not p:
                raise ValueError("unknown puzzle")
            if not brute and p["bits"] > 125:
                raise ValueError("puzzle #%d is beyond the solver's 125-bit range limit" % p["bits"])
            if not brute and not p.get("pubkey"):
                raise ValueError("the public key of puzzle #%d has never been revealed on-chain - "
                                 "only the address-only search is possible" % p["bits"])
            target.update(label="Puzzle #%d" % p["bits"], pubkey=p.get("pubkey"), address=p["address"], bits=p["bits"])
        else:
            c = req.get("custom") or {}
            target["label"] = "Custom range"
            for k in ("start", "end"):
                if not HEX.match(str(c.get(k, ""))):
                    raise ValueError("%s must be a hex value" % k)
                target[k] = c[k]
            if brute:
                if not ADDR.match(str(c.get("address", ""))):
                    raise ValueError("brute force needs a target Bitcoin address")
                target["address"] = c["address"]
            else:
                if not HEX.match(str(c.get("pubkey", ""))):
                    raise ValueError("pubkey must be a hex value")
                target["pubkey"] = c["pubkey"]
                if c.get("address"):
                    if not ADDR.match(c["address"]):
                        raise ValueError("address does not look like a Bitcoin address")
                    target["address"] = c["address"]

        if brute and "bits" in target and target["bits"] > 56:
            raise ValueError("brute force over a 2^%d range is not feasible - use a known-public-key puzzle" % target["bits"])

        cmd = [BINARY, "--json"]
        if brute:
            cmd += ["--brute", "--address", target["address"]]
        else:
            cmd += ["--pubkey", target["pubkey"]]
            if target.get("address"):
                cmd += ["--address", target["address"]]
        if "bits" in target:
            cmd += ["--bits", str(target["bits"])]
        else:
            cmd += ["--start", target["start"], "--end", target["end"]]
        backend = "gpu" if (req.get("backend") == "gpu" and not brute) else "cpu"
        if backend == "gpu":
            cmd += ["--gpu", "--device", str(int(req.get("device", 0)))]
        else:
            cmd += ["--threads", str(max(1, min(256, int(req.get("threads", os.cpu_count() or 1)))))]
        if not brute:
            if req.get("kangaroos"):
                cmd += ["--kangaroos", str(max(2, min(1 << 26, int(req["kangaroos"]))))]
            if req.get("dp") not in (None, ""):
                cmd += ["--dp", str(max(0, min(60, int(req["dp"]))))]

        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise ValueError("a search is already running")
            self.run_id += 1
            self.state = {"status": "running", "target": target, "backend": backend, "log": [], "history": [],
                          "run_id": self.run_id}
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            threading.Thread(target=self._read, args=(self.proc, self.run_id), daemon=True).start()

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()

    def _log(self, text):
        self.state["log"].append(text)
        del self.state["log"][:-200]

    def _read(self, proc, run_id):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {"event": "log", "message": line}
            with self.lock:
                if run_id != self.run_id:
                    return
                st = self.state
                kind = ev.get("event")
                if kind == "start":
                    st["start"] = ev
                    if ev.get("mode") == "brute":
                        self._log("started brute force on %s over 2^%.1f keys" % (ev["backend"], ev["range_bits"]))
                    else:
                        self._log("started on %s: %d kangaroos, %d DP bits" % (ev["backend"], ev["kangaroos"], ev["dp_bits"]))
                elif kind == "progress":
                    st["progress"] = ev
                    st["history"].append([ev["elapsed"], ev["rate"]])
                    del st["history"][:-900]
                elif kind == "found":
                    st["result"] = ev
                    st["status"] = "found"
                    unit = "keys" if (st.get("start") or {}).get("mode") == "brute" else "jumps"
                    self._log("key found after %d %s" % (ev["ops"], unit))
                elif kind == "exhausted":
                    st["exhausted"] = ev
                    st["status"] = "exhausted"
                    self._log("range exhausted: the key is not in this range (%d keys checked)" % ev["ops"])
                elif kind == "stopped":
                    st["status"] = "stopped"
                    self._log("stopped by user")
                elif kind == "error":
                    st["status"] = "error"
                    st["error"] = ev.get("message", "unknown error")
                    self._log("error: " + st["error"])
                else:
                    self._log(ev.get("message", line))
        code = proc.wait()
        with self.lock:
            if run_id == self.run_id and self.state["status"] == "running":
                self.state["status"] = "error"
                self.state["error"] = "solver exited unexpectedly (code %d)" % code
                self._log(self.state["error"])


JOB = Job()
INFO = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "web", "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._send(200, {"puzzles": PUZZLES, "info": INFO})
        elif path == "/api/state":
            self._send(200, JOB.snapshot())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        # only accept requests made by this page (blocks cross-site form posts)
        if self.headers.get("X-Requested-With") != "kangaroo-ui":
            return self._send(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/start":
                JOB.start(req)
            elif self.path == "/api/stop":
                JOB.stop()
            else:
                return self._send(404, {"error": "not found"})
            self._send(200, {"ok": True})
        except (ValueError, TypeError, KeyError) as e:
            self._send(400, {"error": str(e)})


def main():
    global INFO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    ensure_binary()
    INFO = solver_info()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("kangaroo UI running on http://%s:%d  (Ctrl+C to quit)" % (args.host, args.port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOB.stop()


if __name__ == "__main__":
    main()
