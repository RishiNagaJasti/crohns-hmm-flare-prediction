SHELL := /bin/bash
ROOT := $(abspath .)
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
OUT ?= final_outputs
THREAD_ENV := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN_ARGS := --out $(OUT) --n-seeds 10 --n-total 120 --n-train 65 --n-val 15 \
            --n-days 120 --future-days 240 --pmf-horizon 180 --hazard-horizon 120 \
            --n-param-draws 16 --n-hazard-draws 16 --n-starts 3 --max-em-iter 35 \
            --n-jobs 2 --performance-bootstrap 2000 --calibration-bootstrap 500 \
            --base-seed 1729

.PHONY: environment test simulate paper metadata manifest verify reproduce quick clean-build

environment:
	@if [ ! -x "$(PY)" ]; then \
		python3 -m venv $(VENV); \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.lock.txt; \
	fi

test: environment
	PYTHONPATH=$(ROOT) $(THREAD_ENV) $(PY) -m pytest -q tests/test_pipeline.py

simulate: environment
	rm -rf $(OUT)
	$(THREAD_ENV) $(PY) crohns_hmm_pipeline.py $(RUN_ARGS)

paper: environment
	mkdir -p figures
	cp $(OUT)/figures/*.pdf figures/
	$(PY) fill_manuscript.py --root $(ROOT) --outputs $(OUT)
	command -v latexmk >/dev/null || { echo 'latexmk/TeX Live is required to build the PDF.'; exit 1; }
	latexmk -pdf -interaction=nonstopmode -halt-on-error Crohns_HMM_Time_to_Flare_Study.tex
	latexmk -c Crohns_HMM_Time_to_Flare_Study.tex

metadata: environment
	$(PY) build_release_metadata.py --root $(ROOT) --outputs $(OUT)

manifest: environment
	$(PY) build_manifest.py --root $(ROOT) --output MANIFEST.sha256

verify: environment
	PYTHONPATH=$(ROOT) $(THREAD_ENV) $(PY) verify_release.py --root $(ROOT) --outputs $(OUT)

reproduce: test simulate paper metadata manifest verify
	@echo 'Full reproducibility build completed successfully.'

quick: environment
	rm -rf quick_outputs
	PYTHONPATH=$(ROOT) $(THREAD_ENV) $(PY) -m pytest -q tests/test_pipeline.py
	$(THREAD_ENV) $(PY) crohns_hmm_pipeline.py --out quick_outputs --quick
	@echo 'Quick smoke build completed. It is not the manuscript result.'

clean-build:
	rm -rf final_outputs quick_outputs smoke_outputs profile_seed profile_seed1
	rm -f Crohns_HMM_Time_to_Flare_Study.aux Crohns_HMM_Time_to_Flare_Study.fdb_latexmk \
	      Crohns_HMM_Time_to_Flare_Study.fls Crohns_HMM_Time_to_Flare_Study.log \
	      Crohns_HMM_Time_to_Flare_Study.out
