#!/bin/bash
# San Fernando (WV03 PAN tri-view) FoundationStereo.
# 호스트에서 실행. 단발 docker.
#
# 전제: run_ba_san_fernando.sh 완료 (pan_aoi/*.rpc 가 BA-adjusted + 단위 strip)
# 입력: pan_aoi/{522,544,554}.tif (PAN 1밴드 직접 — RGB 추출 없음), foundation_6view_config.json
# 단계: s2p foundation (dl_lr_threshold=1000) -> dabeeo dsm_interpolation
#
# Usage:
#   ./run_san_fernando_foundation.sh
#   ID_GPU=7 ./run_san_fernando_foundation.sh
#   ID_GPUS=4,5,6,7 ./run_san_fernando_foundation.sh   # multi-GPU
#   FILL_GAPS=0 / FILL_MODE=ground 등 daejeon과 동일

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_sf_6view.log"
: > "$LOGFILE"
exec > >(stdbuf -oL tee "$LOGFILE") 2>&1
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

sudo -v
( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) &
SUDO_PID=$!; trap "kill $SUDO_PID 2>/dev/null || true" EXIT

ID_GPU=${ID_GPU:-7}
ID_GPUS=${ID_GPUS:-}
MAXPROC=${MAXPROC:-}
FILL_GAPS=${FILL_GAPS:-1}
FILL_MODE=${FILL_MODE:-avg}

DIR_DATA=/data/kevin_workspace/dataset_stereo
S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd
PRETRAINED=/data/kevin_workspace/pretrained_model
S2P_HD_IMG="s2p-hd-dl:latest"
DSM_INTERP_SO="${S2P_HD_REPO}/dsm_interpolation/dsm_interpolation.so"

REL=satellite/argentina/san_fernando
cfg_host="${DIR_DATA}/${REL}/foundation_6view_config.json"
cfg_docker="/data/${REL}/foundation_6view_config.json"
out_host="${DIR_DATA}/${REL}/2_s2p-hd/output_foundation_6view"

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  San Fernando (WV03 PAN tri-view) Foundation${NC}"
if [ -n "$ID_GPUS" ]; then echo -e "${GREEN}    GPUs=${ID_GPUS}  (multi-GPU)${NC}"; else echo -e "${GREEN}    GPU=${ID_GPU}${NC}"; fi
echo -e "${GREEN}    FILL_GAPS=${FILL_GAPS} (FILL_MODE=${FILL_MODE})${NC}"
echo -e "${GREEN}=================================================${NC}"

[ -f "$cfg_host" ] || { echo -e "${RED}config 없음${NC}"; exit 1; }
[ "$FILL_GAPS" = "1" ] && [ ! -f "$DSM_INTERP_SO" ] && { echo -e "${RED}.so 없음${NC}"; exit 1; }

fill_dsm_gaps_host() {
    stdbuf -oL python3 -u - "$1" "$2" "$DSM_INTERP_SO" "$FILL_MODE" <<'PYEOF'
import sys, ctypes as ct, numpy as np, rasterio
in_tif,out_tif,so=sys.argv[1],sys.argv[2],sys.argv[3]
mode=1 if (len(sys.argv)>4 and sys.argv[4]=="ground") else 0
lib=ct.CDLL(so)
lib.interpolate_with_all_direction_mode.argtypes=(ct.POINTER(ct.c_float),ct.c_int,ct.c_int,ct.POINTER(ct.c_longlong),ct.c_int,ct.c_int,ct.c_int)
lib.interpolate_with_all_direction_mode.restype=None
with rasterio.open(in_tif) as r:
    data=r.read(1).astype(np.float32); profile=r.profile.copy(); nd=r.nodata
invalid=(~np.isfinite(data))|((data==nd) if (nd is not None and not np.isnan(nd)) else False)
data[invalid]=-9999.0
nc=np.argwhere(data==-9999.0).astype(np.int64)
print(f'  invalid: {nc.shape[0]} ({100.0*nc.shape[0]/data.size:.1f}%)',flush=True)
if nc.shape[0]>0:
    flat=np.ascontiguousarray(data.ravel()); cf=np.ascontiguousarray(nc.ravel())
    lib.interpolate_with_all_direction_mode(flat.ctypes.data_as(ct.POINTER(ct.c_float)),ct.c_int(data.shape[0]),ct.c_int(data.shape[1]),cf.ctypes.data_as(ct.POINTER(ct.c_longlong)),ct.c_int(nc.shape[0]),ct.c_int(nc.shape[1]),ct.c_int(mode))
    data=flat.reshape(data.shape)
profile.update(dtype='float32',nodata=-9999.0)
with rasterio.open(out_tif,'w',**profile) as w: w.write(data,1)
print(f'  wrote {out_tif}',flush=True)
PYEOF
}

# GPU/병렬 config 반영
python3 - "$cfg_host" "$ID_GPUS" "$MAXPROC" <<'PYCFG'
import json,sys
p,gpus,mp=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p))
if gpus:
    n=len([g for g in gpus.split(',') if g.strip()])
    d['dl_stereo_device']=','.join(f'cuda:{i}' for i in range(n)); d['max_processes_stereo_matching']=n
else:
    d['dl_stereo_device']='cuda:0'; d['max_processes_stereo_matching']=1
if mp: d['max_processes']=int(mp)
json.dump(d,open(p,'w'),indent=2)
print(f"  config: device={d['dl_stereo_device']}, match_workers={d['max_processes_stereo_matching']}, max_processes={d.get('max_processes')}")
PYCFG

sudo rm -rf "$out_host"
S2P_INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
echo '=== warming DL model cache ==='
python3 - <<'PYWARM'
import json,sys; sys.path.insert(0,'/workspace')
c=json.load(open('${cfg_docker}')); c['dl_stereo_device']='cuda:0'
from s2p import dl_stereo; dl_stereo.load_model(c); print('warmed')
PYWARM
echo '=== s2p foundation (san_fernando) ==='
stdbuf -oL s2p ${cfg_docker}
echo '=== done ==='
"
echo -e "${YELLOW}[s2p foundation]${NC}"
stdbuf -oL sudo docker run --rm \
    --user "$(id -u):$(id -g)" \
    --gpus "\"device=${ID_GPUS:-$ID_GPU}\"" \
    --shm-size=64g --net=host \
    -e HOME=/tmp -e HF_HOME=/dl_cache/hf -e XDG_CACHE_HOME=/dl_cache/xdg -e TORCH_HOME=/dl_cache/torch \
    -v "${PRETRAINED}/dl_cache":/dl_cache \
    -v "${DIR_DATA}":/data -v "${S2P_HD_REPO}":/workspace -v "${PRETRAINED}":/pretrained:ro \
    --entrypoint bash "${S2P_HD_IMG}" -c "${S2P_INNER}"
rc=$?

if [ $rc -ne 0 ] || [ ! -f "${out_host}/dsm.tif" ]; then
    echo -e "${RED}  s2p FAILED (exit=$rc)${NC}"; exit $rc
fi
echo -e "${GREEN}  s2p DONE → ${out_host}/dsm.tif ($(du -h "${out_host}/dsm.tif"|cut -f1))${NC}"

if [ "$FILL_GAPS" = "1" ]; then
    in_tif="${out_host}/dsm-filtered.tif"; out_tif="${out_host}/dsm-filled.tif"
    [ -f "$in_tif" ] || { echo -e "${RED}dsm-filtered 없음${NC}"; exit 1; }
    echo -e "${YELLOW}[dabeeo fill]${NC}"
    fill_dsm_gaps_host "$in_tif" "$out_tif" && echo -e "${GREEN}  fill DONE → ${out_tif} ($(du -h "$out_tif"|cut -f1))${NC}"
fi

echo ""
echo -e "${GREEN}========== All done $(date '+%Y-%m-%d %H:%M:%S') | ${LOGFILE} ==========${NC}"
