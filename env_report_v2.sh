#!/usr/bin/env bash
# Emit a JSON record of the software/hardware environment for a run manifest.
# usage: bash scripts/env_report_v2.sh > build/env_report_v2.json
set -u
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr -d '\n\r'; }
py="$(python3 --version 2>&1)"
pk="$(python3 - <<'PY' 2>/dev/null
import importlib, json
out = {}
for m in ("numpy", "scipy", "yaml", "pandas", "matplotlib", "sklearn", "torch", "ripser", "gudhi"):
    try:
        out[m] = getattr(importlib.import_module(m), "__version__", "present")
    except Exception:
        pass
print(json.dumps(out))
PY
)"
[ -z "$pk" ] && pk='{}'
cpu="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
[ -z "$cpu" ] && cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
cores="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
mem="$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo unknown)"
gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
cc="$( (cc --version 2>/dev/null || echo none) | head -1)"
node="$( (node --version 2>/dev/null) || echo none)"
cat <<JSON
{
  "date_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "os": "$(esc "$(uname -srm)")",
  "cpu": "$(esc "$cpu")",
  "cores": "$(esc "$cores")",
  "mem_mb": "$(esc "$mem")",
  "gpu": "$(esc "${gpu:-none}")",
  "python": "$(esc "$py")",
  "python_packages": $pk,
  "cc": "$(esc "$cc")",
  "node": "$(esc "$node")",
  "git_commit": "$(esc "$commit")",
  "git_uncommitted_files": "$dirty",
  "PYTHONHASHSEED": "${PYTHONHASHSEED:-unset}"
}
JSON
