SHELL := /bin/bash
ROOT := $(abspath .)
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
OUT ?= final_outputs
THREAD_ENV := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LATEXMK ?= latexmk

# --- IEEE Access class assets -------------------------------------------
# vendor/ieeeaccess/upstream/ holds the official package, byte-for-byte and
# never edited.  The documented font-only compatibility change is applied to a
# generated copy under build/ieeeaccess/, which is not committed.
IEEE_UPSTREAM_DIR := vendor/ieeeaccess/upstream
IEEE_BUILD_DIR    := build/ieeeaccess
IEEE_PATCH        := patches/ieeeaccess-font-fallback.patch
IEEE_CLASS        := $(IEEE_UPSTREAM_DIR)/ieeeaccess.cls
IEEE_STAMP        := $(IEEE_BUILD_DIR)/.prepared

# --- standalone Figure 1 -------------------------------------------------
FIG_DIR  := figures
FIG1_SRC := fig1_graphicalmodel.tex
FIG1_PDF := $(FIG_DIR)/fig1_graphicalmodel.pdf

RUN_ARGS := --out $(OUT) --n-seeds 10 --n-total 120 --n-train 65 --n-val 15 \
            --n-days 120 --future-days 240 --pmf-horizon 180 --hazard-horizon 120 \
            --n-param-draws 16 --n-hazard-draws 16 --n-starts 3 --max-em-iter 35 \
            --n-jobs 2 --performance-bootstrap 2000 --calibration-bootstrap 500 \
            --base-seed 1729

.PHONY: environment test simulate ieeeaccess-assets figure1 paper metadata manifest \
        verify reproduce quick clean-build clean-paper

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

ieeeaccess-assets: $(IEEE_STAMP)

$(IEEE_STAMP): $(IEEE_CLASS) $(IEEE_PATCH)
	rm -rf $(IEEE_BUILD_DIR)
	mkdir -p $(IEEE_BUILD_DIR)
	cp -R $(IEEE_UPSTREAM_DIR)/. $(IEEE_BUILD_DIR)/
	patch --batch --forward --directory=$(IEEE_BUILD_DIR) -p1 < $(abspath $(IEEE_PATCH))
	touch $(IEEE_STAMP)

figure1: $(FIG1_PDF)

$(FIG_DIR):
	mkdir -p $(FIG_DIR)

$(FIG1_PDF): $(FIG1_SRC) | $(FIG_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(FIG_DIR) $(FIG1_SRC)
	$(LATEXMK) -c -outdir=$(FIG_DIR) $(FIG1_SRC)

paper: environment ieeeaccess-assets figure1
	mkdir -p figures
	cp $(OUT)/figures/*.pdf figures/
	$(PY) fill_manuscript.py --root $(ROOT) --outputs $(OUT)
	command -v latexmk >/dev/null || { echo 'latexmk/TeX Live is required to build the PDF.'; exit 1; }
	TEXINPUTS="$(abspath $(IEEE_BUILD_DIR))//:$$TEXINPUTS" \
	  $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error Crohns_HMM_Time_to_Flare_Study.tex
	TEXINPUTS="$(abspath $(IEEE_BUILD_DIR))//:$$TEXINPUTS" \
	  $(LATEXMK) -c Crohns_HMM_Time_to_Flare_Study.tex

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

clean-paper:
	$(LATEXMK) -C Crohns_HMM_Time_to_Flare_Study.tex || true
	rm -f $(FIG1_PDF)

clean-build:
	rm -rf $(IEEE_BUILD_DIR)
	rm -f $(FIG1_PDF)
	rm -f Crohns_HMM_Time_to_Flare_Study.pdf
	$(LATEXMK) -C Crohns_HMM_Time_to_Flare_Study.tex || true
	rm -rf final_outputs quick_outputs
