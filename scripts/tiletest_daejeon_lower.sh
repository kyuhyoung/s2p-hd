#!/bin/bash
# 대전 lower 아파트 #1 단일 타일 재연 하네스 — dl_unipolarity_margin 스윕.
# 목적: "지면을 얼마나 깊게 앉혀야 pair_2가 타워를 잡는가"의 창을 실측.
#   회당 ~5-10분 (타일 1개, GPU 1개). 풀런(4시간) 없이 가설 검증.
#
# Usage:
#   ID_GPU=0 ./tiletest_daejeon_lower.sh              # 기본 스윕 {50,150,250,350,450}
#   MARGINS="100 200" ID_GPU=0 ./tiletest_daejeon_lower.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/tiletest_daejeon_lower.log"
: > "$LOGFILE"
exec > >(stdbuf -oL tee "$LOGFILE") 2>&1

ID_GPU=${ID_GPU:-0}
MARGINS=${MARGINS:-"50 150 250 350 450"}
DIR_DATA=/data/kevin_workspace/dataset_stereo
S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd
PRETRAINED=/data/kevin_workspace/pretrained_model
CFG_HOST=${DIR_DATA}/satellite/daejeon/lower/foundation_config_tiletest.json
CFG_DOCKER=/data/satellite/daejeon/lower/foundation_config_tiletest.json
OUT_HOST=${DIR_DATA}/satellite/daejeon/lower/2_s2p-hd/output_tiletest

echo "========== tiletest sweep $(date '+%F %T') | margins: ${MARGINS} =========="

for M in $MARGINS; do
    echo ""
    echo "===== dl_unipolarity_margin = ${M} ====="
    python3 - "$CFG_HOST" "$M" <<'PYCFG'
import json, sys
p, m = sys.argv[1], int(sys.argv[2])
d = json.load(open(p))
d['dl_unipolarity_margin'] = m
json.dump(d, open(p, 'w'), indent=2)
PYCFG
    rm -rf "$OUT_HOST"

    stdbuf -oL docker run --rm \
        --user "$(id -u):$(id -g)" \
        --gpus "\"device=${ID_GPU}\"" \
        --shm-size=16g --net=host \
        -e HOME=/tmp -e HF_HOME=/dl_cache/hf -e XDG_CACHE_HOME=/dl_cache/xdg -e TORCH_HOME=/dl_cache/torch \
        -v /data/kevin_workspace/pretrained_model/dl_cache:/dl_cache \
        -v "${DIR_DATA}":/data -v "${S2P_HD_REPO}":/workspace -v "${PRETRAINED}":/pretrained:ro \
        --entrypoint bash s2p-hd-dl:latest -c "
set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
stdbuf -oL s2p ${CFG_DOCKER} 2>&1 | grep -av 'RuntimeWarning\|warnings.warn' | tail -3
" || { echo \"margin ${M}: s2p 실패\"; continue; }

    # 측정: 아파트 지점의 pair별 최대높이 + pair_2 지면 시차
    python3 - "$M" <<'PYEOF'
import sys, glob, numpy as np, rasterio
m = sys.argv[1]
base='/data/kevin_workspace/dataset_stereo/satellite/daejeon/lower/2_s2p-hd/output_tiletest'
tiles=glob.glob(f'{base}/tiles/row_*/col_*')
if not tiles:
    print(f'[margin {m}] 타일 없음'); sys.exit()
t=tiles[0]
r,c=16435,15942
tr=int(t.split('row_')[1].split('_')[0]); tc=int(t.split('col_')[1].split('_')[0])
yy,xx=r-tr,c-tc
row=[f'[margin {m:>3s}]']
for pair in ['pair_1','pair_2']:
    try:
        hm=rasterio.open(f'{t}/{pair}/height_map.tif').read(1)
        sub=hm[max(0,yy-70):yy+70, max(0,xx-70):xx+70]
        row.append(f'{pair} max {np.nanmax(sub):5.1f}')
    except Exception as e:
        row.append(f'{pair} 없음')
try:
    d=rasterio.open(f'{t}/pair_2/rectified_disp.tif').read(1)
    dv=d[np.isfinite(d)]
    dmm=np.loadtxt(f'{t}/pair_2/disp_min_max.txt')
    row.append(f'| p2 지면 p50 {np.percentile(dv,50):.0f}  range[{dmm[0]:.0f},{dmm[1]:.0f}]')
except Exception: pass
try:
    hm=rasterio.open(f'{t}/height_map.tif').read(1)
    sub=hm[max(0,yy-70):yy+70, max(0,xx-70):xx+70]
    row.append(f'| fused max {np.nanmax(sub):5.1f} valid {100*np.isfinite(sub).mean():4.1f}%')
except Exception: pass
print('  '.join(row))
PYEOF
done

echo ""
echo "========== sweep 완료 $(date '+%F %T') =========="