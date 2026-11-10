#!/usr/bin/env bash
# Build the solver and start the web interface.
#
#   ./run.sh                 CPU build, UI on http://127.0.0.1:8070
#   ./run.sh --gpu           CUDA build (needs nvcc)
#   ./run.sh --port 9000     use another port
#   ./run.sh --test          run the test suite before starting
#   ./run.sh --no-browser    do not open a browser window
set -euo pipefail

cd "$(dirname "$0")"

gpu=0
test=0
browser=1
port=8070

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) gpu=1 ;;
        --test) test=1 ;;
        --no-browser) browser=0 ;;
        --port)
            [ $# -ge 2 ] || { echo "error: --port needs a value" >&2; exit 1; }
            port="$2"; shift ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

for tool in make g++ python3; do
    command -v "$tool" >/dev/null || { echo "error: $tool is not installed" >&2; exit 1; }
done

if [ "$gpu" = 1 ]; then
    command -v nvcc >/dev/null || { echo "error: --gpu needs the CUDA toolkit (nvcc not found)" >&2; exit 1; }
    echo "==> building solver with CUDA"
    make clean >/dev/null
    make gpu
else
    # a previous CUDA build leaves gpu_cuda.o behind; rebuild cleanly for CPU
    if [ -f gpu_cuda.o ]; then make clean >/dev/null; fi
    echo "==> building solver (CPU)"
    make
fi

if [ "$test" = 1 ]; then
    echo "==> running tests"
    python3 tests/test_solver.py
fi

url="http://127.0.0.1:$port"
if [ "$browser" = 1 ]; then
    opener=""
    command -v xdg-open >/dev/null && opener=xdg-open
    command -v open >/dev/null && [ -z "$opener" ] && opener=open
    if [ -n "$opener" ]; then
        # open once the server answers
        ( for _ in $(seq 1 50); do
              if python3 -c "import urllib.request; urllib.request.urlopen('$url', timeout=1)" 2>/dev/null; then
                  "$opener" "$url" >/dev/null 2>&1; exit 0
              fi
              sleep 0.2
          done ) &
    fi
fi

echo "==> starting web interface on $url (Ctrl+C to stop)"
exec python3 server.py --port "$port"
