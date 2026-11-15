#!/usr/bin/env bash
# Full software-reproducible pass: tests, consistency checks, tables, demos.
# Hardware-bound results (Loihi, A100, ChipWhisperer) are NOT regenerated;
# see docs/HARDWARE_DEPENDENCY_MATRIX_v2.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
export PYTHONHASHSEED=0
bash scripts/env_report_v2.sh > build/env_report_v2.json
python3 -m unittest discover -s tests -p 'test_*_v2.py' -t . 2>&1 | tee build/test_py_v2.log | tail -3
make -s -C c test | tee build/test_c_v2.log
node --test js/doh_wire_codec_v2.test.js > build/test_js_v2.log 2>&1 && echo "js tests: ok"
python3 scripts/consistency_checks_v2.py | tee build/consistency_v2.log
python3 scripts/generate_tables_v2.py
python3 scripts/tda_pipeline_demo_v2.py --windows 20 --out build/tda_demo_point_v2.json
python3 scripts/tda_pipeline_demo_v2.py --representation delay --width 60 --eps 0.6 --max-scale 1.5 --windows 20 --out build/tda_demo_delay_v2.json
python3 scripts/validate_manifest_v2.py configs/run_manifest_v2.yaml || true
echo "done; artifacts in build/"
