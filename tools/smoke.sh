#!/usr/bin/env bash
# Smoke-test every page with headless Chrome: loads each chapter with ?selftest=1
# and checks the JSON verdict; also runs tests/test.html.
# Usage: tools/smoke.sh [base-url]   (default: starts a local server on :8765)
set -u
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
BASE="${1:-}"
if [ -z "$BASE" ]; then
  python3 -m http.server 8765 >/dev/null 2>&1 &
  SERVER=$!
  trap 'kill $SERVER' EXIT
  sleep 1
  BASE="http://localhost:8765"
fi
fail=0
dump() { perl -e 'alarm shift; exec @ARGV' 300 "$CHROME" --headless=new --disable-gpu --virtual-time-budget="$2" --dump-dom "$1" 2>/dev/null; }
for f in chapters/*.html; do
  out=$(dump "$BASE/$f?selftest=1" 15000 | sed -n 's/.*<pre id="selftest-result">\(.*\)<\/pre>.*/\1/p')
  if echo "$out" | grep -q '"ok":true'; then echo "ok    $f"; else echo "FAIL  $f  $out"; fail=1; fi
done
out=$(dump "$BASE/tests/test.html" 120000 | sed -n 's/.*<pre id="summary">\(.*\)<\/pre>.*/\1/p')
if echo "$out" | grep -q '"failed":0'; then echo "ok    tests/test.html $out"; else echo "FAIL  tests/test.html $out"; fail=1; fi
exit $fail
