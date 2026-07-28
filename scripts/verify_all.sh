#!/usr/bin/env bash
set -eu

PYTHON_BIN="${PYTHON:-python3}"

export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/verify_manifest.py
"${PYTHON_BIN}" src/verify_boolean_phase_robustness.py
"${PYTHON_BIN}" -m pytest -p no:cacheprovider tests -q

echo "verification_status=PASS"
