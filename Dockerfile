FROM python:3.13.5-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git make latexmk texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-pictures poppler-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.lock.txt /work/requirements.lock.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.lock.txt
COPY . /work
CMD ["make", "reproduce", "VENV=/usr/local"]
