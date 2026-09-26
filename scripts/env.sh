#!/usr/bin/env bash
UAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UAG_ROOT
export PYTHONPATH="$UAG_ROOT:$UAG_ROOT/.deps${PYTHONPATH:+:$PYTHONPATH}"
export UAG_PYTHON="${UAG_PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
cd "$UAG_ROOT"
