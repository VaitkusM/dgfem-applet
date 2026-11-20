#!/usr/bin/env bash
# Smoke-test the site with headless Chrome.
#
#   tools/smoke.sh [--gpu] [base-url]
#
# Default mode (also run by CI, .github/workflows/test.yml):
#   - loads index.html and every chapters/*.html with ?selftest=1 and requires the JSON verdict
#     in <pre id="selftest-result"> to say "ok":true (see lib/ui/chapter.js for its fields);
#   - runs the whole test suite in the browser (tests/test.html) and requires "failed":0.
#   Chrome runs with --disable-gpu, so WebGPU is absent: the GPU tests are reported as SKIPPED
#   (the summary line shows how many) and chapter 12 exercises its CPU fallback.
# GPU mode (--gpu; local machines with a real GPU, not CI):
#   - runs tests/gpu/gpu-tests.html and chapter 12 with WebGPU enabled and FAILS if a GPU test
#     is skipped or failed, or if chapter 12 did not validate the GPU against the CPU twin.
#   Needs Deno (tools/gpu-smoke.js drives Chrome over the DevTools protocol, because GPU work
#   is not covered by Chrome's virtual time budget).
#
# base-url: an already running static server; default: start `python3 -m http.server` on a free port.
# Environment: CHROME = browser binary (default: autodetect google-chrome / chromium / macOS Chrome),
#              CHROME_FLAGS = extra flags (CI uses --no-sandbox), SMOKE_PORT = port of the local server (default 8765).
# Exit status: 0 if everything passed, 1 otherwise.
set -u
cd "$(dirname "$0")/.."

GPU=0; BASE=""
for a in "$@"; do
  case "$a" in
    --gpu) GPU=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) BASE="$a" ;;
  esac
done

# ---- find Chrome ---------------------------------------------------------------
if [ -z "${CHROME:-}" ]; then
  for c in google-chrome google-chrome-stable chromium chromium-browser chrome \
           "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" "/Applications/Chromium.app/Contents/MacOS/Chromium"; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then CHROME="$c"; break; fi
  done
fi
if [ -z "${CHROME:-}" ]; then echo "FAIL  no Chrome/Chromium found (set CHROME=/path/to/chrome)"; exit 1; fi
export CHROME

# ---- a portable hard timeout (macOS has no timeout(1)) -----------------------
if command -v timeout >/dev/null 2>&1; then limit() { timeout -k 5 "$@"; }
elif command -v gtimeout >/dev/null 2>&1; then limit() { gtimeout -k 5 "$@"; }
else limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }
fi

# ---- static server ---------------------------------------------------------------
SERVER=""; PROFILE=""
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null && wait "$SERVER" 2>/dev/null; [ -n "$PROFILE" ] && rm -rf "$PROFILE"; }
trap cleanup EXIT
if [ -z "$BASE" ]; then
  PORT="${SMOKE_PORT:-8765}"
  python3 -m http.server "$PORT" >/dev/null 2>&1 &
  SERVER=$!
  BASE="http://localhost:$PORT"
  ok=0
  sleep 0.5   # a server that cannot bind the port exits right away; give it time to do so before probing the port
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if ! kill -0 "$SERVER" 2>/dev/null; then break; fi   # the server died (port in use?)
    if curl -s -o /dev/null "$BASE/index.html" 2>/dev/null; then ok=1; break; fi
    sleep 0.25
  done
  if [ "$ok" != 1 ] || ! kill -0 "$SERVER" 2>/dev/null; then
    echo "FAIL  could not start a static server on port $PORT (in use? set SMOKE_PORT or pass a base URL)"; SERVER=""; exit 1
  fi
fi
BASE="${BASE%/}"

if [ "$GPU" = 1 ]; then
  if ! command -v deno >/dev/null 2>&1; then echo "FAIL  --gpu needs deno"; exit 1; fi
  deno run --allow-run --allow-net --allow-read --allow-write --allow-env tools/gpu-smoke.js "$BASE"
  exit $?
fi

PROFILE="$(mktemp -d 2>/dev/null || mktemp -d -t smoke)"
fail=0
# dump <url> <virtual-time-budget ms>: the DOM after the page has run for that much (virtual) time
dump() {
  # shellcheck disable=SC2086
  limit 300 "$CHROME" --headless=new --disable-gpu --no-first-run --user-data-dir="$PROFILE" ${CHROME_FLAGS:-} \
    --virtual-time-budget="$2" --dump-dom "$1" 2>/dev/null
}
verdict() { sed -n 's/.*<pre id="selftest-result"[^>]*>\(.*\)<\/pre>.*/\1/p' | head -n 1; }

pages="index.html"
for f in chapters/*.html; do pages="$pages $f"; done
for f in $pages; do
  out=$(dump "$BASE/$f?selftest=1" 15000 | verdict)
  if echo "$out" | grep -q '"ok":true'; then echo "ok    $f"; else echo "FAIL  $f  ${out:-(no self-test verdict: the page script did not run to the end)}"; fail=1; fi
done

out=$(dump "$BASE/tests/test.html" 120000 | sed -n 's/.*<pre id="summary"[^>]*>\(.*\)<\/pre>.*/\1/p' | head -n 1)
skipped=$(echo "$out" | sed -n 's/.*"skipped":\([0-9]*\).*/\1/p')
if echo "$out" | grep -q '"failed":0' && echo "$out" | grep -q '"passed":[1-9]'; then
  echo "ok    tests/test.html  $(echo "$out" | sed -n 's/.*\("passed":[0-9]*,"failed":[0-9]*,"skipped":[0-9]*\).*/\1/p')"
  if [ -n "$skipped" ] && [ "$skipped" != 0 ]; then
    echo "note  $skipped test(s) SKIPPED in this browser (no WebGPU with --disable-gpu); run tools/smoke.sh --gpu on a machine with a GPU:"
    echo "      $(echo "$out" | sed -n 's/.*"skippedTests":\(\[[^]]*\]\).*/\1/p')"
  fi
else echo "FAIL  tests/test.html  ${out:-(no summary)}"; fail=1; fi
exit $fail
