# Kangaroo – Bitcoin puzzle solver

A Pollard's kangaroo solver for the [Bitcoin puzzle transaction](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)
keys, written in C++ with an optional NVIDIA (CUDA) backend and a local web interface.

The puzzle is a public challenge: puzzle *N* is an address whose private key was deliberately
chosen in the range `[2^(N-1), 2^N)`. Puzzle 70 has a 2^69 key range. Trying every key is
hopeless (centuries on a GPU), but its **public key is known**, and that allows the kangaroo
method, which needs only about `2·sqrt(range)` ≈ 2^35.5 group operations.

> Puzzle 70 was solved and its coins were claimed years ago, so there is nothing to win here –
> it is a good, verifiable benchmark. The method only works on puzzles whose public key has been
> revealed on-chain; for the others only brute force over addresses is possible.

## Quick start

```bash
make                 # CPU build (needs g++)
make test            # self test + solves puzzles 20–50 from their public keys
python3 server.py    # web interface on http://127.0.0.1:8070
```

Command line:

```bash
./kangaroo --bits 70 \
  --pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483 \
  --address 19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR
```

`--bits N` means "puzzle N"; an arbitrary interval can be given with `--start HEX --end HEX`.
`--address` is optional and only used to double-check the recovered key. Run `./kangaroo --help`
for all options.

## NVIDIA GPU

```bash
make gpu             # needs the CUDA toolkit (nvcc)
./kangaroo --gpu --bits 70 --pubkey 0290e6…
```

The web interface enables its GPU switch automatically when the binary was built with CUDA and
a device is present.

**Status of the GPU backend:** the machine this was developed on has no NVIDIA card, so
`src/gpu_cuda.cu` has not been compiled or run yet. To keep that risk small, the whole
arithmetic and stepping kernel (`src/kernel.h`) is shared: the GPU runs the same `kang_step()`
function the CPU backend and the tests use, and the CUDA file is only a thin wrapper around it
(memory transfers and the kernel launch). It is a straightforward port, not a tuned one.

## Web interface

`python3 server.py` (standard library only) serves a single page where you pick a puzzle or a
custom public key and range, choose CPU or GPU, and watch speed, progress against the expected
work, distinguished points and the result. It listens on localhost only.

## How it works

* The target `Q = k·G` is shifted so the unknown `k'` lies in `[0, W)`.
* Half of the kangaroos are **tame** (they start at known scalars), half are **wild** (they start
  at `Q + d·G`). All of them hop along the curve; the hop length depends only on the current
  point, so two kangaroos that ever land on the same point stay together from then on.
* Points whose x coordinate ends in `dp` zero bits are **distinguished** and are stored in a hash
  table together with the distance travelled. When a tame and a wild kangaroo report the same
  distinguished point, `k' = tame_distance − wild_distance`.
* Each hop is one affine point addition; a batch of kangaroos shares a single field inversion
  (Montgomery's trick).

Measured over 30 runs of puzzle 40 the average cost was 1.19 × `2·sqrt(W)` jumps. A single run
can take anything from about half to a few times the average – the search is probabilistic.

## Performance

| Hardware | Speed | Puzzle 50 | Puzzle 70 (estimate) |
|---|---|---|---|
| Intel i7-7500U, 4 threads | ~6.5 M jumps/s | ~10 s | ~2.5–3 hours on average |

## Layout

| File | Purpose |
|---|---|
| `src/kernel.h` | secp256k1 field arithmetic and the kangaroo stepping kernel (shared by CPU and CUDA) |
| `src/main.cpp` | command line, herd setup, CPU threads, distinguished point table |
| `src/gpu_cuda.cu`, `src/gpu.h` | CUDA backend |
| `src/hash.h` | SHA-256, RIPEMD-160, Base58Check (address derivation) |
| `server.py`, `web/index.html` | web interface |
| `puzzles.json` | addresses and public keys of some solved puzzles (no private keys) |
| `tests/test_solver.py` | end-to-end tests |
