SHELL := /bin/bash
ROOT := $(abspath .)
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
OUT ?= final_outputs
THREAD_ENV := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LATEXMK ?= latexmk

# --- IEEE Access class assets -------------------------------------------
# The official IEEE Access template is not committed to this repository.
# A checksum-verified copy is extracted under build/ieeeaccess/, together
# with the fonts shipped in the template archive; the class is used unmodified.
# The user must download the IEEE Access template from
# https://template-selector.ieee.org/ (search "IEEE Access", pick Research Article,
# download LaTeX template) and place the zip at $(IEEE_ARCHIVE). Override with:
#   make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip
IEEE_ARCHIVE      ?= $(HOME)/Downloads/IEEE LaTeX Template.zip
IEEE_CHECKSUMS    := vendor/ieeeaccess/UPSTREAM_SHA256.txt
IEEE_BUILD_DIR    := build/ieeeaccess
IEEE_STAMP        := $(IEEE_BUILD_DIR)/.prepared
# pdfTeX finds class files through TEXINPUTS but fonts, metrics, and map files
# through separate search paths; point all of them at the extracted template.
IEEE_DIR_ABS      := $(abspath $(IEEE_BUILD_DIR))
IEEE_TEX_ENV      = TEXINPUTS="$(IEEE_DIR_ABS)//:$$TEXINPUTS" TFMFONTS="$(IEEE_DIR_ABS)//:$$TFMFONTS" T1FONTS="$(IEEE_DIR_ABS)//:$$T1FONTS" TEXFONTMAPS="$(IEEE_DIR_ABS)//:$$TEXFONTMAPS"

# --- standalone Figure 1 -------------------------------------------------
FIG_DIR  := figures
FIG1_SRC := fig1_graphicalmodel.tex
FIG1_PDF := $(FIG_DIR)/fig1_graphicalmodel.pdf

RUN_ARGS := --out $(OUT) --n-seeds 10 --n-total 120 --n-train 65 --n-val 15 \
            --n-days 120 --future-days 240 --pmf-horizon 180 --hazard-horizon 120 \
            --n-param-draws 16 --n-hazard-draws 16 --n-starts 3 --max-em-iter 35 --em-tol 1e-4 \
            --n-jobs 2 --performance-bootstrap 2000 --calibration-bootstrap 500 \
            --base-seed 1729

.PHONY: environment preflight test simulate results ieeeaccess-assets figure1 paper metadata \
        manifest verify reproduce quick clean-build clean-paper

PYTHON ?= python3.13
PY_REQUIRED := 3.13.5

environment:
	@if [ ! -x "$(PY)" ]; then \
		command -v $(PYTHON) >/dev/null || { echo "$(PYTHON) not found; install Python 3.13 or pass PYTHON=/path/to/python3.13"; exit 1; }; \
		$(PYTHON) -m venv $(VENV); \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.lock.txt; \
	fi
	@v=$$($(PY) -c 'import platform; print(platform.python_version())'); \
	case "$$v" in \
	  3.13.*) ;; \
	  *) echo "ERROR: $(VENV) uses Python $$v; the locked environment requires Python 3.13.x."; \
	     echo "Remove $(VENV) and rerun with PYTHON=/path/to/python3.13."; \
	     exit 1;; \
	esac; \
	if [ "$$v" != "$(PY_REQUIRED)" ]; then \
	  echo "WARNING: the lock file was generated with Python $(PY_REQUIRED); $(VENV) uses Python $$v."; \
	fi

test: environment
	PYTHONPATH=$(ROOT) $(THREAD_ENV) $(PY) -m pytest -q tests/test_pipeline.py

simulate: environment
	rm -rf $(OUT)
	$(THREAD_ENV) $(PY) crohns_hmm_pipeline.py $(RUN_ARGS)

ieeeaccess-assets: $(IEEE_STAMP)

$(IEEE_STAMP): $(IEEE_CHECKSUMS)
	@if [ ! -f "$(IEEE_ARCHIVE)" ]; then \
	  echo "ERROR: IEEE Access template not found at:"; \
	  echo "  $(IEEE_ARCHIVE)"; \
	  echo ""; \
	  echo "Download it from https://template-selector.ieee.org/"; \
	  echo "  (search 'IEEE Access', pick Research Article, LaTeX template)"; \
	  echo "and place the zip at the path above, or override with:"; \
	  echo "  make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip"; \
	  exit 1; \
	fi
	rm -rf $(IEEE_BUILD_DIR)
	mkdir -p $(IEEE_BUILD_DIR)
	@echo "Extracting IEEE Access template from $(IEEE_ARCHIVE)"
	unzip -j -q -o "$(IEEE_ARCHIVE)" \
	  '*/ieeeaccess.cls' '*/spotcolor.sty' '*/*.png' '*/*.pfb' '*/*.tfm' '*/*.map' '*/*.fd' \
	  -d $(IEEE_BUILD_DIR)/ 2>/dev/null || \
	unzip -j -q -o "$(IEEE_ARCHIVE)" \
	  'ieeeaccess.cls' 'spotcolor.sty' '*.png' '*.pfb' '*.tfm' '*.map' '*.fd' \
	  -d $(IEEE_BUILD_DIR)/
	@echo "Verifying SHA-256 checksums against $(IEEE_CHECKSUMS)"
	@while read expected_sha rel_path; do \
	  filename=$$(basename "$$rel_path"); \
	  actual_sha=$$(shasum -a 256 "$(IEEE_BUILD_DIR)/$$filename" | awk '{print $$1}'); \
	  if [ "$$actual_sha" != "$$expected_sha" ]; then \
	    echo "ERROR: SHA mismatch for $$filename"; \
	    echo "  expected: $$expected_sha"; \
	    echo "  actual:   $$actual_sha"; \
	    echo "The template archive at $(IEEE_ARCHIVE) may be a different version"; \
	    echo "than the one this repository was built against. Download ACCESS_latex_template_20260513-1-1.zip,"; \
	    echo "linked from https://ieeeaccess.ieee.org/authors/preparing-your-article/, or update $(IEEE_CHECKSUMS)."; \
	    exit 1; \
	  fi; \
	done < $(IEEE_CHECKSUMS)
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
	$(THREAD_ENV) $(PY) fill_manuscript.py --root $(ROOT) --outputs $(OUT)
	command -v latexmk >/dev/null || { echo 'latexmk/TeX Live is required to build the PDF.'; exit 1; }
	$(IEEE_TEX_ENV) \
	  $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error Crohns_HMM_Time_to_Flare_Study.tex
	mv Crohns_HMM_Time_to_Flare_Study.pdf Crohns_HMM_Time_to_Flare_Study_v2.3.pdf
	$(IEEE_TEX_ENV) \
	  $(LATEXMK) -c Crohns_HMM_Time_to_Flare_Study.tex

metadata: environment
	$(PY) build_release_metadata.py --root $(ROOT) --outputs $(OUT)

manifest: environment
	$(PY) build_manifest.py --root $(ROOT) --output MANIFEST.sha256

verify: environment
	PYTHONPATH=$(ROOT) $(THREAD_ENV) $(PY) verify_release.py --root $(ROOT) --outputs $(OUT) --release-mode $$($(PY) -c 'import json; print(json.load(open("release_config.json"))["release_mode"])')

# Check the main external tools and the IEEE template before the long simulation
# starts, so a missing prerequisite fails in seconds, not after it. (make verify also
# needs git and a clean checkout of the release tag; those are checked at the end.)
preflight:
	@if [ ! -f "$(IEEE_STAMP)" ] && [ ! -f "$(IEEE_ARCHIVE)" ]; then \
	  echo "ERROR: IEEE Access template not found at: $(IEEE_ARCHIVE)"; \
	  echo "Download it (see README) or pass IEEE_ARCHIVE=/path/to/template.zip."; \
	  echo "To run the analysis and check the results tables without TeX or the template, use: make results"; \
	  exit 1; \
	fi
	@command -v $(LATEXMK) >/dev/null || { echo "ERROR: latexmk (TeX Live) not found. Use 'make results' to run the analysis without building the PDF."; exit 1; }
	@command -v pdftotext >/dev/null || { echo "ERROR: pdftotext (poppler) not found; 'make verify' needs it. Use 'make results' to run the analysis without it."; exit 1; }
	@kpsewhich IEEEtran.cls >/dev/null || { echo "ERROR: IEEEtran.cls not found (Debian/Ubuntu: install texlive-publishers). Use 'make results' to run the analysis without it."; exit 1; }
	@command -v unzip >/dev/null || { echo "ERROR: unzip not found; it is needed to extract the IEEE template. Use 'make results' to run the analysis without it."; exit 1; }
	@command -v shasum >/dev/null || { echo "ERROR: shasum not found; it is needed to check the IEEE template files. Use 'make results' to run the analysis without it."; exit 1; }

# Analysis only: tests, full simulation, and regeneration of the performance,
# paired-difference, calibration and non-current-flare tables by rerunning the
# aggregation code on the archived outputs. Needs no TeX and no IEEE template.
results: test simulate
	PYTHONPATH=$(ROOT) $(THREAD_ENV) $(PY) verify_release.py --root $(ROOT) --outputs $(OUT) --numbers-only --release-mode $$($(PY) -c 'import json; print(json.load(open("release_config.json"))["release_mode"])')

reproduce: preflight test simulate paper metadata manifest verify
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
	rm -f Crohns_HMM_Time_to_Flare_Study_v2.3.pdf
	$(LATEXMK) -C Crohns_HMM_Time_to_Flare_Study.tex || true
	rm -rf final_outputs quick_outputs
