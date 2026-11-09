#!/usr/bin/env python3
"""End-to-end tests: the solver is given only a public key and a range and must
recover the private key of several already-solved low Bitcoin puzzles.

    python3 tests/test_solver.py            (or: make test)
    python3 tests/test_solver.py 55 60      (pick the puzzles to run)
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "kangaroo")

with open(os.path.join(ROOT, "puzzles.json")) as f:
    PUZZLES = {p["bits"]: p for p in json.load(f)}


def run(args):
    out = subprocess.run([BIN] + args, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def solve(bits):
    p = PUZZLES[bits]
    t = time.time()
    events = run(["--json", "--bits", str(bits), "--pubkey", p["pubkey"], "--address", p["address"]])
    found = [e for e in events if e["event"] == "found"]
    assert found, "puzzle %d: no key found" % bits
    r = found[0]
    key = int(r["key"], 16)
    assert 1 << (bits - 1) <= key < 1 << bits, "key outside the puzzle range"
    assert r["address_match"] and r["address"] == p["address"], "address mismatch"
    print("  [ ok ] puzzle #%d  key=%s  address=%s  (%.1fs, %d jumps)" % (bits, r["key"], r["address"], time.time() - t, r["ops"]))


def main():
    which = [int(a) for a in sys.argv[1:]] or [20, 25, 30, 35, 40, 45, 50]
    print("self test")
    st = subprocess.run([BIN, "--selftest"], capture_output=True, text=True)
    print(st.stdout, end="")
    assert st.returncode == 0, "self test failed"

    print("custom range (key at the first and last position, non power-of-two width)")
    pub = PUZZLES[20]["pubkey"]  # key 0xd2c55
    for start, end in (("d2c55", "f0000"), ("90000", "d2c55"), ("d2c00", "d2cff")):
        r = [e for e in run(["--json", "--pubkey", pub, "--start", start, "--end", end]) if e["event"] == "found"]
        assert r and r[0]["key"] == "d2c55", "range %s..%s failed" % (start, end)
        print("  [ ok ] range %s..%s" % (start, end))

    print("brute force by address (public key unknown)")
    # full low puzzle, given only its address
    p = PUZZLES[20]
    t = time.time()
    r = [e for e in run(["--json", "--brute", "--bits", "20", "--address", p["address"]]) if e["event"] == "found"]
    assert r and r[0]["key"] == "d2c55" and r[0]["address"] == p["address"], "brute puzzle 20 failed"
    print("  [ ok ] puzzle #20  key=%s  (%.1fs, %d keys)" % (r[0]["key"], time.time() - t, r[0]["ops"]))
    # a self-derived target in a custom range, found by address and by raw hash160
    der = subprocess.run([BIN, "--derive", "6bc1f"], capture_output=True, text=True).stdout
    addr = [ln.split(":")[1].strip() for ln in der.splitlines() if ln.startswith("address")][0]
    r = [e for e in run(["--json", "--brute", "--start", "60000", "--end", "70000", "--address", addr]) if e["event"] == "found"]
    assert r and r[0]["key"] == "6bc1f" and r[0]["address"] == addr, "brute custom range failed"
    print("  [ ok ] custom range by address  key=%s" % r[0]["key"])

    print("solving puzzles from their public keys")
    for bits in which:
        solve(bits)
    print("all tests passed")


if __name__ == "__main__":
    main()
