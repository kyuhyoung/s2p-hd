#!/bin/bash
# Daegu (K3) FoundationStereo — south + north
# 호스트에서 실행. 각 side마다 단발 docker container 띄워서 s2p 실행 후 종료.
#
# Usage:
#   ./run_daegu_foundation.sh                # 양쪽 다
#   SIDE=south ./run_daegu_foundation.sh     # south만
#   SIDE=north ./run_daegu_foundation.sh     # north만
#   ID_GPU=0 ./run_daegu_foundation.sh       # GPU 선택 (default: 0)
#   FILL_GAPS=0 ./run_daegu_foundation.sh    # dabeeo gap-fill 건너뜀 (default: 1)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_daegu_foundation.log"
: > "$LOGFILE"
exec > >(stdbuf -oL tee "$LOGFILE") 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# === sudo keep-alive (명시적 PID 추적해서 wait 멈춤 방지) ===
sudo -v
( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) &
SUDO_PID=$!
trap "kill $SUDO_PID 2>/dev/null || true" EXIT

# === 설정 ===
ID_GPU=${ID_GPU:-0}
SIDE_FILTER=${SIDE:-}
FILL_GAPS=${FILL_GAPS:-1}

DIR_DATA=/data/kevin_workspace/dataset_stereo
S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd
PRETRAINED=/data/kevin_workspace/pretrained_model
S2P_HD_IMG="s2p-hd-dl:latest"
DSM_INTERP_SO="${S2P_HD_REPO}/dsm_interpolation/dsm_interpolation.so"

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  Daegu (K3) FoundationStereo${NC}"
echo -e "${GREEN}    GPU       = ${ID_GPU}${NC}"
echo -e "${GREEN}    FILL_GAPS = ${FILL_GAPS}  (1: dabeeo interp 적용, 0: 안 함)${NC}"
[ -n "$SIDE_FILTER" ] && echo -e "${GREEN}    SIDE      = ${SIDE_FILTER}${NC}"
echo -e "${GREEN}=================================================${NC}"

if [ "$FILL_GAPS" = "1" ] && [ ! -f "$DSM_INTERP_SO" ]; then
    echo -e "${RED}  ERROR: dsm_interpolation.so 없음 — ${DSM_INTERP_SO}${NC}"
    echo -e "${RED}  cd ${S2P_HD_REPO}/dsm_interpolation && make${NC}"
    exit 1
fi

fill_dsm_gaps_host() {
    # dsm-filtered.tif → dsm-filled.tif (호스트 python에서 .so 호출)
    local in_tif=$1
    local out_tif=$2
    stdbuf -oL python3 -u - "$in_tif" "$out_tif" "$DSM_INTERP_SO" <<'PYEOF'
import sys, ctypes as ct
import numpy as np
import rasterio

in_tif, out_tif, so = sys.argv[1], sys.argv[2], sys.argv[3]
lib = ct.CDLL(so)
lib.interpolate_with_all_direction.argtypes = (
    ct.POINTER(ct.c_float), ct.c_int, ct.c_int,
    ct.POINTER(ct.c_longlong), ct.c_int, ct.c_int,
)
lib.interpolate_with_all_direction.restype = None

with rasterio.open(in_tif) as r:
    data = r.read(1).astype(np.float32)
    profile = r.profile.copy()
    nd = r.nodata

# NaN 또는 sentinel nodata 마스크
if nd is not None and not np.isnan(nd):
    invalid = (~np.isfinite(data)) | (data == nd)
else:
    invalid = ~np.isfinite(data)

data[invalid] = -9999.0  # in-place fill sentinel
nan_coords = np.argwhere(data == -9999.0).astype(np.int64)
n = int(nan_coords.shape[0])
print(f'  invalid pixels: {n}  ({100.0*n/data.size:.1f}%)', flush=True)

if n > 0:
    flat = np.ascontiguousarray(data.ravel())
    cflat = np.ascontiguousarray(nan_coords.ravel())
    lib.interpolate_with_all_direction(
        flat.ctypes.data_as(ct.POINTER(ct.c_float)),
        ct.c_int(data.shape[0]), ct.c_int(data.shape[1]),
        cflat.ctypes.data_as(ct.POINTER(ct.c_longlong)),
        ct.c_int(nan_coords.shape[0]), ct.c_int(nan_coords.shape[1]),
    )
    data = flat.reshape(data.shape)

profile.update(dtype='float32', nodata=-9999.0)
with rasterio.open(out_tif, 'w', **profile) as w:
    w.write(data, 1)
print(f'  wrote {out_tif}', flush=True)
PYEOF
}

run_side() {
    local side=$1
    local cfg_host="${DIR_DATA}/satellite/daegu/${side}/foundation_config.json"
    local cfg_docker="/data/satellite/daegu/${side}/foundation_config.json"
    local out_host="${DIR_DATA}/satellite/daegu/${side}/2_s2p-hd/output_foundation"

    echo ""
    echo -e "${YELLOW}========== [${side}] ==========${NC}"
    echo "  config: ${cfg_host}"
    echo "  out:    ${out_host}"

    if [ ! -f "$cfg_host" ]; then
        echo -e "${RED}  config missing — skip${NC}"
        return 1
    fi

    sudo rm -rf "$out_host"

    local S2P_INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
echo '=== s2p foundation (${side}) ==='
stdbuf -oL s2p ${cfg_docker}
echo '=== ${side} done ==='
"

    stdbuf -oL sudo docker run --rm \
        --user "$(id -u):$(id -g)" \
        --gpus "device=${ID_GPU}" \
        --shm-size=64g \
        --net=host \
        -e HOME=/tmp \
        -e HF_HOME=/tmp/hf_cache \
        -e XDG_CACHE_HOME=/tmp/.cache \
        -e TORCH_HOME=/tmp/torch_cache \
        -v "${DIR_DATA}":/data \
        -v "${S2P_HD_REPO}":/workspace \
        -v "${PRETRAINED}":/pretrained:ro \
        --entrypoint bash \
        "${S2P_HD_IMG}" \
        -c "${S2P_INNER}"
    local rc=$?

    if [ $rc -eq 0 ] && [ -f "${out_host}/dsm.tif" ]; then
        echo -e "${GREEN}  [${side}] s2p DONE  →  ${out_host}/dsm.tif  ($(du -h "${out_host}/dsm.tif" | cut -f1))${NC}"
    else
        echo -e "${RED}  [${side}] s2p FAILED  (exit=$rc)${NC}"
        return $rc
    fi

    # === dabeeo gap-fill (FILL_GAPS=1) ===
    if [ "$FILL_GAPS" = "1" ]; then
        local in_tif="${out_host}/dsm-filtered.tif"
        local out_tif="${out_host}/dsm-filled.tif"
        if [ ! -f "$in_tif" ]; then
            echo -e "${RED}  [${side}] dsm-filtered.tif 없음, dabeeo fill 건너뜀${NC}"
            return 1
        fi
        echo -e "${YELLOW}  [${side}] dabeeo dsm_interpolation (dsm-filtered → dsm-filled)...${NC}"
        if fill_dsm_gaps_host "$in_tif" "$out_tif"; then
            echo -e "${GREEN}  [${side}] fill DONE  →  ${out_tif}  ($(du -h "$out_tif" | cut -f1))${NC}"
        else
            echo -e "${RED}  [${side}] fill FAILED${NC}"
            return 1
        fi
    fi

    return 0
}

for side in south north; do
    if [ -n "$SIDE_FILTER" ] && [ "$side" != "$SIDE_FILTER" ]; then continue; fi
    run_side "$side"
done

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  All done  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}  Log: ${LOGFILE}${NC}"
echo -e "${GREEN}==================================================${NC}"
