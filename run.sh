#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    echo "Usage: $0 [csv_file]"
    echo "  csv_file  Optional path to a ratings CSV exported from index.html."
    echo "            Copied into place as user.csv before generating charts."
    echo "            If omitted, an existing user.csv in this directory is used."
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

if [[ $# -ge 1 ]]; then
    csv_file="$1"
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: CSV file not found: $csv_file" >&2
        exit 1
    fi
    cp "$csv_file" user.csv
elif [[ ! -f user.csv ]]; then
    echo "Error: no user.csv found. Export it from index.html or pass a CSV path." >&2
    usage
fi

python3 -c "import numpy, plotly, pandas, openpyxl" 2>/dev/null || {
    echo "Installing dependencies: numpy plotly pandas openpyxl"
    pip install numpy plotly pandas openpyxl
}

python3 PolarAreaChart.py

if command -v open >/dev/null 2>&1; then
    open Assessments.html
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open Assessments.html
else
    echo "Report generated: Assessments.html"
fi
