#!/usr/bin/env bash
#
# One-command demo launcher for the Edge Naval AI Sandbox.
#
# Brings up everything a recording needs and refuses to open the browser until
# the server is genuinely serving -- a demo should never start on a stack trace.
#
#   ./demo.sh                 agent containment console (the one to film)
#   ./demo.sh --harness       watchstander dashboard instead
#   ./demo.sh --claude        drive the agent with Claude
#   ./demo.sh --check         run preflight only, start nothing
#
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV=".venv"
PY="$VENV/bin/python"
TARGET="console"
BACKEND="fallback"
INTERVAL="2.2"
PORT=""
OPEN_BROWSER=1
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness)   TARGET="harness" ;;
    --console)   TARGET="console" ;;
    --claude)    BACKEND="anthropic" ;;
    --offline)   BACKEND="fallback" ;;
    --interval)  INTERVAL="$2"; shift ;;
    --port)      PORT="$2"; shift ;;
    --no-open)   OPEN_BROWSER=0 ;;
    --check)     CHECK_ONLY=1 ;;
    -h|--help)   sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

[[ -z "$PORT" ]] && { [[ "$TARGET" == "console" ]] && PORT=8788 || PORT=8787; }

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YEL=$'\033[33m'; CYA=$'\033[36m'; OFF=$'\033[0m'
step(){ printf "  %s->%s %s\n" "$CYA" "$OFF" "$1"; }
ok(){   printf "  %sok%s   %s\n" "$GRN" "$OFF" "$1"; }
warn(){ printf "  %swarn%s %s\n" "$YEL" "$OFF" "$1"; }
die(){  printf "\n  %sstopped:%s %s\n\n" "$RED" "$OFF" "$1" >&2; exit 1; }

printf "\n%s  EDGE NAVAL AI SANDBOX -- DEMO LAUNCHER%s\n" "$BOLD" "$OFF"
printf "%s  ------------------------------------------------------------%s\n" "$DIM" "$OFF"

# ---------------------------------------------------------------- interpreter
# A bare `python` on macOS is usually conda or the system build, neither of
# which has these dependencies. The venv is not optional.
if [[ ! -x "$PY" ]]; then
  step "creating $VENV"
  command -v python3 >/dev/null || die "python3 not found on PATH."
  python3 -m venv "$VENV" || die "could not create $VENV"
fi
ok "interpreter $($PY -V 2>&1) at $PY"

# ---------------------------------------------------------------- packages
if ! "$PY" - <<'PYCHECK' >/dev/null 2>&1
import fastembed, faiss, sklearn, pydantic, numpy, scipy
PYCHECK
then
  step "installing dependencies (first run, ~1-2 min)"
  "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
  "$PY" -m pip install -q -r requirements.txt || die "dependency install failed."
fi
ok "dependencies present"

# ---------------------------------------------------------------- model cache
MODEL_DIR="$("$PY" - <<'PYCACHE' 2>/dev/null || true
from edge_embedding import DEFAULT_MODEL, locate_staged_model, resolve_cache_dir
found = locate_staged_model(resolve_cache_dir(), DEFAULT_MODEL)
print(found or "")
PYCACHE
)"
if [[ -z "$MODEL_DIR" ]]; then
  warn "embedding model is not staged yet -- downloading (~90 MB, needs network)"
  "$PY" edge_embedding.py --stage || die "model staging failed. Run it on a networked host, then copy ~/.cache/edge_embedding across."
  ok "model staged"
else
  ok "model staged (offline-ready)"
fi

# ---------------------------------------------------------------- inference
if [[ "$BACKEND" == "anthropic" ]]; then
  if "$PY" - <<'PYKEY' >/dev/null 2>&1
import sys
from agent_brain import _ensure_env
import os
_ensure_env()
sys.exit(0 if os.getenv("ANTHROPIC_API_KEY") else 1)
PYKEY
  then ok "Claude inference: ANTHROPIC_API_KEY found"
  else
    warn "no ANTHROPIC_API_KEY (checked environment and .env) -- falling back to the offline brain"
    BACKEND="fallback"
  fi
else
  ok "inference: deterministic offline brain (no key needed, no network)"
fi

# ---------------------------------------------------------------- port
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  warn "port $PORT is in use -- stopping the previous demo"
  lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  sleep 1
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "port $PORT is still held by another process."
fi
ok "port $PORT free"

if [[ "$CHECK_ONLY" == "1" ]]; then
  printf "\n  %spreflight passed -- nothing started (--check)%s\n\n" "$GRN" "$OFF"
  exit 0
fi

# ---------------------------------------------------------------- launch
LOG="$(mktemp -t navydemo)"
if [[ "$TARGET" == "console" ]]; then
  APP="agent console"; URL="http://127.0.0.1:$PORT"
  "$PY" -u agent_console.py --port "$PORT" --backend "$BACKEND" --interval "$INTERVAL" >"$LOG" 2>&1 &
  READY="console  :"
else
  APP="watchstander dashboard"; URL="http://127.0.0.1:$PORT"
  "$PY" -u main_harness.py --port "$PORT" >"$LOG" 2>&1 &
  READY="dashboard :"
fi
SERVER_PID=$!

cleanup(){
  printf "\n  stopping %s...\n" "$APP"
  kill "$SERVER_PID" 2>/dev/null || true
  # Give it a moment, then insist. A demo that leaves the port held means the
  # next take fails on "address already in use".
  for _ in 1 2 3 4 5 6; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$SERVER_PID" 2>/dev/null && kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$LOG"
  printf "  done.\n\n"
}
trap cleanup EXIT INT TERM

step "starting the $APP (calibrating the baseline, ~10-20 s)"
# Wait for the server's own ready banner. Polling the log rather than the port
# means a crash during boot surfaces immediately instead of after a timeout.
for _ in $(seq 1 120); do
  grep -q "$READY" "$LOG" 2>/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; sed 's/^/  | /' "$LOG" | tail -20; die "the server exited during startup."; }
  sleep 1
done
grep -q "$READY" "$LOG" 2>/dev/null || { sed 's/^/  | /' "$LOG" | tail -20; die "the server did not become ready in 120 s."; }
ok "serving at $URL"

if [[ "$OPEN_BROWSER" == "1" ]] && command -v open >/dev/null 2>&1; then
  open "$URL" >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------- the script
printf "\n%s  READY TO RECORD  %s%s\n" "$GRN$BOLD" "$URL" "$OFF"
if [[ "$TARGET" == "console" ]]; then
cat <<'BEATS'

  Suggested beats, about 90 seconds:

    1. Let it run ~15 s with the sandbox on. Blocked commands hit the
       boundary; allowed ones reach the vessel. Vessel stays "Within limits".
    2. Press FAULTY. The agent starts probing -- radar, speed, EMCON.
       Blocked climbs. Escaped stays 0.
    3. Read one row of the test-case table aloud: the command, the gate that
       stopped it, the reason.
    4. Click SANDBOX ON to drop the shield. The run resets and the boundary
       goes red.
    5. Wait ~15 s. Escaped starts counting. The vessel goes "Out of bounds"
       -- radar radiating, off route.
    6. Click again to raise the shield. Containment resumes, run resets.

  The point to land: the agent never changed. Only the thing in front of it.

BEATS
else
cat <<'BEATS'

  Suggested beats:

    1. Let the unit patrol nominally for ~10 s.
    2. Press "Activate surface radar" -- contained instantly by doctrine.
    3. Press "Erratic 45 divergence" -- the behaviour layer contains it a
       step before any hard limit is crossed.
    4. Turn Wi-Fi off on camera. Nothing changes.

BEATS
fi
printf "  %sCtrl-C to stop.%s\n\n" "$DIM" "$OFF"

wait "$SERVER_PID"
