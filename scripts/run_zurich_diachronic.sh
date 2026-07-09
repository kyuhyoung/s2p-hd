#!/bin/bash
# Zurich (PNEO 4뷰 cross-date) Diachronic(MonSter)Stereo DSM. 호스트 단발 docker.
# 전제: run_ba_zurich.sh 완료 (1_sat-ba/pan/*.rpc = BA보정).
# images는 1_sat-ba/pan 에서 런타임 자동 채움. use_srtm=true(alt_offset 함정), per-tile.
# ⚠️ cross-date라 단차 가능 → diachronic도 같이 비교 권장.
# Usage: ./run_zurich_foundation.sh   /   ID_GPUS=0,1 ./run_zurich_foundation.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_zurich_diachronic.log"; : > "$LOGFILE"
exec > >(stdbuf -oL tee "$LOGFILE") 2>&1
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
echo "========== $(date '+%Y-%m-%d %H:%M:%S') Zurich Diachronic =========="

if docker ps >/dev/null 2>&1; then SUDO=""; echo "docker: sudo 없이 실행"; else
  SUDO="sudo"; sudo -v; ( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) & trap "kill $! 2>/dev/null||true" EXIT; fi

ID_GPUS=${ID_GPUS:-0,1,2,3}; MAXPROC=${MAXPROC:-}; FILL_GAPS=${FILL_GAPS:-1}; FILL_MODE=${FILL_MODE:-avg}
DIR_DATA=/data/kevin_workspace/dataset_stereo; S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd; PRETRAINED=/data/kevin_workspace/pretrained_model
S2P_HD_IMG="s2p-hd-dl:latest"; DSM_INTERP_SO="${S2P_HD_REPO}/dsm_interpolation/dsm_interpolation.so"
REL=satellite/switzerland/zurich
cfg_host="${DIR_DATA}/${REL}/diachronic_config.json"; cfg_docker="/data/${REL}/diachronic_config.json"
out_host="${DIR_DATA}/${REL}/2_s2p-hd/output_diachronic"; pan_host="${DIR_DATA}/${REL}/1_sat-ba/pan"
MODEL_WARM=1   # dl_stereo 매처는 warming
EXCLUDE=${EXCLUDE:-PNEO4}   # DSM에서 제외할 뷰(파일명 부분문자열). PNEO4=여름 7개월차 → 20m bias라 제외. ""면 전체.

echo -e "${GREEN}  Zurich (PNEO 4뷰 cross-date) Diachronic(MonSter) | GPUs=${ID_GPUS}${NC}"
[ -f "$cfg_host" ] || { echo -e "${RED}config 없음${NC}"; exit 1; }
n_pan=$(ls "${pan_host}/"*_PAN.tif 2>/dev/null | wc -l)
[ "$n_pan" -lt 2 ] && { echo -e "${RED}BA보정 PAN 부족 (run_ba_zurich.sh 먼저)${NC}"; exit 1; }
echo "  BA보정 PAN: ${n_pan} 장"

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

# config: images 자동 채움(1_sat-ba/pan) + GPU device
python3 - "$cfg_host" "$ID_GPUS" "$MAXPROC" "$pan_host" "/data/${REL}/1_sat-ba/pan" "$EXCLUDE" <<'PYCFG'
import json,sys,glob,os
p,gpus,mp,panh,pand,excl=sys.argv[1:7]
d=json.load(open(p))
imgs=[]
for t in sorted(glob.glob(os.path.join(panh,'*_PAN.tif'))):
    b=os.path.basename(t)[:-4]
    if excl and excl in b:
        print(f"  제외: {b}"); continue
    imgs.append({"img":f"{pand}/{b}.tif","rpc":f"{pand}/{b}.rpc"})
d['images']=imgs
n=len([g for g in gpus.split(',') if g.strip()]) if gpus else 1
d['dl_stereo_device']=','.join(f'cuda:{i}' for i in range(n)); d['max_processes_stereo_matching']=n
if mp: d['max_processes']=int(mp)
json.dump(d,open(p,'w'),indent=2)
print(f"  config: {len(imgs)} imgs, device={d['dl_stereo_device']}, model={d.get('dl_stereo_model',d['matching_algorithm'])}")
PYCFG

$SUDO rm -rf "$out_host"
WARM=""
[ "$MODEL_WARM" = "1" ] && WARM="echo '=== warming ==='
python3 - <<'PYWARM'
import json,sys; sys.path.insert(0,'/workspace')
c=json.load(open('${cfg_docker}')); c['dl_stereo_device']='cuda:0'
from s2p import dl_stereo; dl_stereo.load_model(c); print('warmed')
PYWARM"
S2P_INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
${WARM}
echo '=== s2p diachronic (zurich) ==='
stdbuf -oL s2p ${cfg_docker}
echo '=== done ==='"
echo -e "${YELLOW}[s2p diachronic]${NC}"
stdbuf -oL $SUDO docker run --rm --user "$(id -u):$(id -g)" \
    --gpus "\"device=${ID_GPUS}\"" --shm-size=64g --net=host \
    -e HOME=/tmp -e HF_HOME=/dl_cache/hf -e XDG_CACHE_HOME=/dl_cache/xdg -e TORCH_HOME=/dl_cache/torch \
    -v "${PRETRAINED}/dl_cache":/dl_cache \
    -v "${DIR_DATA}":/data -v "${S2P_HD_REPO}":/workspace -v "${PRETRAINED}":/pretrained:ro \
    --entrypoint bash "${S2P_HD_IMG}" -c "${S2P_INNER}"
rc=$?
[ $rc -eq 0 ] && [ -f "${out_host}/dsm.tif" ] || { echo -e "${RED}  s2p FAILED (exit=$rc)${NC}"; exit $rc; }
echo -e "${GREEN}  s2p DONE → ${out_host}/dsm.tif ($(du -h "${out_host}/dsm.tif"|cut -f1))${NC}"
if [ "$FILL_GAPS" = "1" ]; then
  in_tif="${out_host}/dsm-filtered.tif"; out_tif="${out_host}/dsm-filled.tif"
  [ -f "$in_tif" ] && { echo -e "${YELLOW}[dabeeo fill]${NC}"; fill_dsm_gaps_host "$in_tif" "$out_tif" && echo -e "${GREEN}  fill DONE → ${out_tif}${NC}"; }
fi
echo -e "${GREEN}========== All done $(date '+%Y-%m-%d %H:%M:%S') | ${LOGFILE} ==========${NC}"
