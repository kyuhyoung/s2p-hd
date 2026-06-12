#!/bin/bash
# Daejeon (PNEO4 tri-stereo) FoundationStereo — upper + lower
# 호스트에서 실행. 각 part마다 단발 docker container로 s2p 실행.
#
# 전제: run_ba_daejeon.sh 완료 (0_PS/*.rpc 가 BA-adjusted 상태)
#
# 단계:
#   1) 0_PS(6-band) -> rgb/(3-band RGB) 추출 + BA rpc 복사  [호스트, 최초 1회]
#   2) s2p foundation stereo (dl_lr_threshold=1000)          [docker]
#   3) dabeeo dsm_interpolation (dsm-filtered -> dsm-filled) [호스트]
#
# Usage:
#   ./run_daejeon_foundation.sh                 # 양쪽 다 (upper -> lower)
#   PART=upper ./run_daejeon_foundation.sh      # upper만
#   PART=lower ./run_daejeon_foundation.sh      # lower만
#   ID_GPU=7 ./run_daejeon_foundation.sh        # GPU 선택 (default: 7)
#   ID_GPUS=4,5,6,7 ./run_daejeon_foundation.sh  # multi-GPU 타일 병렬 (stereo matching)
#   MAXPROC=8 ./run_daejeon_foundation.sh        # CPU 단계 병렬 워커 수
#   FILL_GAPS=0 ./run_daejeon_foundation.sh     # dabeeo gap-fill 건너뜀
#   FILL_MODE=ground ./run_daejeon_foundation.sh # 건물경계 gap을 ground쪽으로 채움 (halo 방지)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_daejeon_foundation.log"
: > "$LOGFILE"
exec > >(stdbuf -oL tee "$LOGFILE") 2>&1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# sudo keep-alive (명시적 PID 추적)
sudo -v
( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) &
SUDO_PID=$!
trap "kill $SUDO_PID 2>/dev/null || true" EXIT

# === 설정 ===
ID_GPU=${ID_GPU:-7}
ID_GPUS=${ID_GPUS:-}        # 예: "4,5,6,7" — 주어지면 multi-GPU 모드 (ID_GPU 무시)
MAXPROC=${MAXPROC:-}        # 예: 8 — 주어지면 config의 max_processes 오버라이드
PART_FILTER=${PART:-}
FILL_GAPS=${FILL_GAPS:-1}
FILL_MODE=${FILL_MODE:-avg}   # avg | ground (건물경계 gap을 ground쪽 최솟값으로)

DIR_DATA=/data/kevin_workspace/dataset_stereo
S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd
PRETRAINED=/data/kevin_workspace/pretrained_model
S2P_HD_IMG="s2p-hd-dl:latest"
DSM_INTERP_SO="${S2P_HD_REPO}/dsm_interpolation/dsm_interpolation.so"

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  Daejeon (PNEO4 tri-stereo) FoundationStereo${NC}"
if [ -n "$ID_GPUS" ]; then
    echo -e "${GREEN}    GPUs      = ${ID_GPUS}  (multi-GPU tile parallel)${NC}"
else
    echo -e "${GREEN}    GPU       = ${ID_GPU}${NC}"
fi
[ -n "$MAXPROC" ] && echo -e "${GREEN}    MAXPROC   = ${MAXPROC}${NC}"
echo -e "${GREEN}    FILL_GAPS = ${FILL_GAPS}  (FILL_MODE=${FILL_MODE})${NC}"
[ -n "$PART_FILTER" ] && echo -e "${GREEN}    PART      = ${PART_FILTER}${NC}"
echo -e "${GREEN}=================================================${NC}"

if [ "$FILL_GAPS" = "1" ] && [ ! -f "$DSM_INTERP_SO" ]; then
    echo -e "${RED}  ERROR: dsm_interpolation.so 없음 — ${DSM_INTERP_SO}${NC}"
    exit 1
fi

fill_dsm_gaps_host() {
    local in_tif=$1
    local out_tif=$2
    stdbuf -oL python3 -u - "$in_tif" "$out_tif" "$DSM_INTERP_SO" "$FILL_MODE" <<'PYEOF'
import sys, ctypes as ct
import numpy as np
import rasterio

in_tif, out_tif, so = sys.argv[1], sys.argv[2], sys.argv[3]
mode = 1 if (len(sys.argv) > 4 and sys.argv[4] == "ground") else 0
lib = ct.CDLL(so)
lib.interpolate_with_all_direction_mode.argtypes = (
    ct.POINTER(ct.c_float), ct.c_int, ct.c_int,
    ct.POINTER(ct.c_longlong), ct.c_int, ct.c_int, ct.c_int,
)
lib.interpolate_with_all_direction_mode.restype = None

with rasterio.open(in_tif) as r:
    data = r.read(1).astype(np.float32)
    profile = r.profile.copy()
    nd = r.nodata

if nd is not None and not np.isnan(nd):
    invalid = (~np.isfinite(data)) | (data == nd)
else:
    invalid = ~np.isfinite(data)

data[invalid] = -9999.0
nan_coords = np.argwhere(data == -9999.0).astype(np.int64)
n = int(nan_coords.shape[0])
print(f'  invalid pixels: {n}  ({100.0*n/data.size:.1f}%)', flush=True)

if n > 0:
    flat = np.ascontiguousarray(data.ravel())
    cflat = np.ascontiguousarray(nan_coords.ravel())
    lib.interpolate_with_all_direction_mode(
        flat.ctypes.data_as(ct.POINTER(ct.c_float)),
        ct.c_int(data.shape[0]), ct.c_int(data.shape[1]),
        cflat.ctypes.data_as(ct.POINTER(ct.c_longlong)),
        ct.c_int(nan_coords.shape[0]), ct.c_int(nan_coords.shape[1]),
        ct.c_int(mode),
    )
    data = flat.reshape(data.shape)

profile.update(dtype='float32', nodata=-9999.0)
with rasterio.open(out_tif, 'w', **profile) as w:
    w.write(data, 1)
print(f'  wrote {out_tif}', flush=True)
PYEOF
}

run_part() {
    local part=$1
    local base_host="${DIR_DATA}/satellite/daejeon/${part}"
    local cfg_host="${base_host}/foundation_config.json"
    local cfg_docker="/data/satellite/daejeon/${part}/foundation_config.json"
    local out_host="${base_host}/2_s2p-hd/output_foundation"
    local rgb_host="${base_host}/rgb"

    echo ""
    echo -e "${YELLOW}========== [${part}] ==========${NC}"
    echo "  config: ${cfg_host}"
    echo "  out:    ${out_host}"

    if [ ! -f "$cfg_host" ]; then
        echo -e "${RED}  config missing — skip${NC}"
        return 1
    fi

    # === 1) RGB 추출 (6-band PS -> 3-band) + BA rpc 복사 ===
    echo -e "${YELLOW}  [1/3] RGB 추출 (없으면 생성)...${NC}"
    mkdir -p "$rgb_host"
    local count=0
    for ps in "${base_host}"/0_PS/*_PS.tif; do
        [ -f "$ps" ] || continue
        local name=$(basename "$ps")
        local dst="${rgb_host}/${name}"
        if [ ! -f "$dst" ]; then
            echo "    extracting RGB: ${name}"
            gdal_translate -b 1 -b 2 -b 3 "$ps" "$dst" -co COMPRESS=LZW -co BIGTIFF=YES -q
        else
            echo "    exists: ${name}"
        fi
        # BA-adjusted rpc 복사 (0_PS는 run_ba_daejeon.sh가 이미 교체해 둠)
        cp -f "${ps%.tif}.rpc" "${rgb_host}/${name%.tif}.rpc"
        count=$((count + 1))
    done
    echo -e "${GREEN}  RGB ${count}장 준비 완료${NC}"

    # === GPU/병렬 설정을 config에 반영 ===
    python3 - "$cfg_host" "$ID_GPUS" "$MAXPROC" <<'PYCFG'
import json, sys
cfg_path, gpus, maxproc = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(cfg_path))
if gpus:
    n = len([g for g in gpus.split(',') if g.strip()])
    # 컨테이너에는 0..n-1로 재매핑되어 보임
    d['dl_stereo_device'] = ','.join(f'cuda:{i}' for i in range(n))
    d['max_processes_stereo_matching'] = n
else:
    d['dl_stereo_device'] = 'cuda:0'
    d['max_processes_stereo_matching'] = None
if maxproc:
    d['max_processes'] = int(maxproc)
json.dump(d, open(cfg_path, 'w'), indent=2)
print(f"  config: dl_stereo_device={d['dl_stereo_device']}, "
      f"max_processes_stereo_matching={d['max_processes_stereo_matching']}, "
      f"max_processes={d.get('max_processes')}")
PYCFG

    # === 2) s2p foundation ===
    echo -e "${YELLOW}  [2/3] s2p foundation stereo...${NC}"
    sudo rm -rf "$out_host"

    local S2P_INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
echo '=== s2p foundation (${part}) ==='
stdbuf -oL s2p ${cfg_docker}
echo '=== ${part} done ==='
"

    stdbuf -oL sudo docker run --rm \
        --user "$(id -u):$(id -g)" \
        --gpus "device=${ID_GPUS:-$ID_GPU}" \
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

    if [ $rc -ne 0 ] || [ ! -f "${out_host}/dsm.tif" ]; then
        echo -e "${RED}  [${part}] s2p FAILED (exit=$rc)${NC}"
        return $rc
    fi
    echo -e "${GREEN}  [${part}] s2p DONE  →  ${out_host}/dsm.tif  ($(du -h "${out_host}/dsm.tif" | cut -f1))${NC}"

    # === 3) dabeeo gap-fill ===
    if [ "$FILL_GAPS" = "1" ]; then
        local in_tif="${out_host}/dsm-filtered.tif"
        local out_tif="${out_host}/dsm-filled.tif"
        if [ ! -f "$in_tif" ]; then
            echo -e "${RED}  [${part}] dsm-filtered.tif 없음 — fill 건너뜀${NC}"
            return 1
        fi
        echo -e "${YELLOW}  [3/3] dabeeo dsm_interpolation...${NC}"
        if fill_dsm_gaps_host "$in_tif" "$out_tif"; then
            echo -e "${GREEN}  [${part}] fill DONE  →  ${out_tif}  ($(du -h "$out_tif" | cut -f1))${NC}"
        else
            echo -e "${RED}  [${part}] fill FAILED${NC}"
            return 1
        fi
    fi

    return 0
}

for part in upper lower; do
    if [ -n "$PART_FILTER" ] && [ "$part" != "$PART_FILTER" ]; then continue; fi
    run_part "$part"
done

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  All done  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}  Log: ${LOGFILE}${NC}"
echo -e "${GREEN}==================================================${NC}"
