#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/collect_dsm.log"
DATA_DIR=/data/satellite/seoul/gangnam/samsung/260406_Samseong_gwarp
RESULTS_DIR="${DATA_DIR}/dsm_results"

> "$LOGFILE"

mkdir -p "${RESULTS_DIR}"

echo "Collecting DSM results to ${RESULTS_DIR}/" | stdbuf -oL tee -a "$LOGFILE"

for pair_type in sync dia; do
    for method in sgm diachronic monster foundation; do
        src="${DATA_DIR}/${pair_type}_${method}/dsm.tif"
        dst="${RESULTS_DIR}/samsung_${pair_type}_${method}.tif"
        if [ -f "$src" ]; then
            cp "$src" "$dst"
            echo "  ${pair_type}_${method} -> $(basename $dst)" | stdbuf -oL tee -a "$LOGFILE"
        else
            echo "  ${pair_type}_${method}: not found yet" | stdbuf -oL tee -a "$LOGFILE"
        fi
    done
done

if [ -f "${DATA_DIR}/dsm.tif" ]; then
    cp "${DATA_DIR}/dsm.tif" "${RESULTS_DIR}/samsung_existing_gwarp.tif"
    echo "  existing dsm -> samsung_existing_gwarp.tif" | stdbuf -oL tee -a "$LOGFILE"
fi

echo "" | stdbuf -oL tee -a "$LOGFILE"
echo "Contents:" | stdbuf -oL tee -a "$LOGFILE"
ls -lh "${RESULTS_DIR}/" | stdbuf -oL tee -a "$LOGFILE"
