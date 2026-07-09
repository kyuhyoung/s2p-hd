#!/bin/bash
# Marseille (PNEO3 PAN tri-stereo) classic SGM (stereosgm_gpu) DSM.
# 호스트에서 단발 docker. BA보정 PAN + use_srtm=true (sgm_config.json).
# foundation/diachronic 과 동일 입력·AOI, 매처만 stereosgm_gpu.
#
# Usage: ./run_marseille_sgm.sh   /  ID_GPUS=0,1 ./run_marseille_sgm.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_marseille_sgm.log"; : > "$LOGFILE"
exec > >(stdbuf -oL tee "$LOGFILE") 2>&1
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
echo "========== $(date '+%Y-%m-%d %H:%M:%S')  Marseille SGM =========="

if docker ps >/dev/null 2>&1; then SUDO=""; echo "docker: sudo 없이 실행"; else
  SUDO="sudo"; sudo -v; ( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) & trap "kill $! 2>/dev/null||true" EXIT; fi

ID_GPUS=${ID_GPUS:-0,1,2,3}
FILL_GAPS=${FILL_GAPS:-1}; FILL_MODE=${FILL_MODE:-avg}
DIR_DATA=/data/kevin_workspace/dataset_stereo
S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd
PRETRAINED=/data/kevin_workspace/pretrained_model
S2P_HD_IMG="s2p-hd-dl:latest"
DSM_INTERP_SO="${S2P_HD_REPO}/dsm_interpolation/dsm_interpolation.so"

REL=satellite/france/marseille
cfg_host="${DIR_DATA}/${REL}/sgm_config.json"
cfg_docker="/data/${REL}/sgm_config.json"
out_host="${DIR_DATA}/${REL}/2_s2p-hd/output_sgm"

echo -e "${GREEN}  Marseille (PNEO3 PAN tri-stereo) SGM | GPUs=${ID_GPUS}${NC}"
[ -f "$cfg_host" ] || { echo -e "${RED}config 없음${NC}"; exit 1; }
n_pan=$(ls "${DIR_DATA}/${REL}/1_sat-ba/pan/"*_PAN.tif 2>/dev/null | wc -l)
[ "$n_pan" -lt 2 ] && { echo -e "${RED}BA보정 PAN 부족${NC}"; exit 1; }

$SUDO rm -rf "$out_host"
S2P_INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
echo '=== s2p stereosgm_gpu (marseille) ==='
stdbuf -oL s2p ${cfg_docker}
echo '=== done ==='"
echo -e "${YELLOW}[s2p sgm]${NC}"
stdbuf -oL $SUDO docker run --rm --user "$(id -u):$(id -g)" \
    --gpus "\"device=${ID_GPUS}\"" --shm-size=64g --net=host -e HOME=/tmp \
    -v "${DIR_DATA}":/data -v "${S2P_HD_REPO}":/workspace \
    --entrypoint bash "${S2P_HD_IMG}" -c "${S2P_INNER}"
rc=$?
[ $rc -eq 0 ] && [ -f "${out_host}/dsm.tif" ] || { echo -e "${RED}  SGM FAILED (exit=$rc)${NC}"; exit $rc; }
echo -e "${GREEN}  SGM DONE → ${out_host}/dsm.tif ($(du -h "${out_host}/dsm.tif"|cut -f1))${NC}"

if [ "$FILL_GAPS" = "1" ] && [ -f "$DSM_INTERP_SO" ]; then
  in_tif="${out_host}/dsm-filtered.tif"; out_tif="${out_host}/dsm-filled.tif"
  [ -f "$in_tif" ] && {
    echo -e "${YELLOW}[dabeeo fill]${NC}"
    stdbuf -oL python3 -u - "$in_tif" "$out_tif" "$DSM_INTERP_SO" "$FILL_MODE" <<'PYEOF'
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
fi
echo -e "${GREEN}========== All done $(date '+%Y-%m-%d %H:%M:%S') | ${LOGFILE} ==========${NC}"
