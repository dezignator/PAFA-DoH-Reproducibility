# extended_v2 build/test entry points.  `make help` lists targets.
PY ?= python3

help:
	@echo "targets: experiments report test test-py test-c test-js tables tex consistency demo demo-delay manifest-check env all clean"

test: test-py test-c test-js

test-py:
	$(PY) -m unittest discover -s tests -p 'test_*_v2.py' -t . -v

test-c:
	$(MAKE) -C c test

test-js:
	node --test js/doh_wire_codec_v2.test.js

tables:
	$(PY) scripts/generate_tables_v2.py

tex: tables
	cd latex && pdflatex -interaction=nonstopmode -halt-on-error chapter4_tables_v2.tex

consistency:
	$(PY) scripts/consistency_checks_v2.py

demo:
	$(PY) scripts/tda_pipeline_demo_v2.py --out build/tda_demo_point_v2.json

demo-delay:
	$(PY) scripts/tda_pipeline_demo_v2.py --representation delay --width 60 --eps 0.6 --max-scale 1.5 --out build/tda_demo_delay_v2.json

manifest-check:
	$(PY) scripts/validate_manifest_v2.py configs/run_manifest_v2.yaml

env:
	bash scripts/env_report_v2.sh > build/env_report_v2.json

experiments:
	$(PY) experiments/pafa_federated_experiments_v2.py --out results_v2/federated
	$(PY) experiments/pafa_attack_scenarios_v2.py --out results_v2/attack

report:
	$(PY) experiments/pafa_report_builder_v2.py --fed results_v2/federated --attack results_v2/attack --out results_v2/report

all: test consistency tables

clean:
	rm -rf build latex/*.aux latex/*.log latex/*.out
	$(MAKE) -C c clean

.PHONY: experiments report help test test-py test-c test-js tables tex consistency demo demo-delay manifest-check env all clean
