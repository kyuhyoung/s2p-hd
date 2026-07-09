#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_sf_sgm3.log"; : > "$LOGFILE"; exec > >(stdbuf -oL tee "$LOGFILE") 2>&1
echo "===== $(date '+%H:%M:%S') ====="
sudo -v; ( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) & trap "kill $! 2>/dev/null||true" EXIT
DIR=/data/kevin_workspace/dataset_stereo; REPO=/data/kevin_workspace/etc/s2p-hd
cfg=/data/satellite/argentina/san_fernando/sgm3_config.json
out=$DIR/satellite/argentina/san_fernando/2_s2p-hd/output_sgm3
ID_GPU=${ID_GPU:-7}
sudo rm -rf "$out"
INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1|tail -1
echo '=== s2p stereosgm_gpu (PAN 3) ==='
stdbuf -oL s2p ${cfg}
echo '=== done ==='"
stdbuf -oL sudo docker run --rm --user "$(id -u):$(id -g)" --gpus "device=${ID_GPU}" --shm-size=64g --net=host -e HOME=/tmp \
  -v "${DIR}":/data -v "${REPO}":/workspace --entrypoint bash s2p-hd-dl:latest -c "${INNER}"
rc=$?
if [ ! -f "${out}/dsm.tif" ]; then echo "SGM3 FAILED (exit=$rc)"; echo "Log: ${LOGFILE}"; exit $rc; fi
python3 -c "
import rasterio,numpy as np
with rasterio.open('${out}/dsm.tif') as r:
 a=r.read(1);nd=r.nodata;v=np.isfinite(a)&(a!=nd if nd else True)
 print(f'SGM3 DSM: {r.width}x{r.height}, valid {100*v.sum()/a.size:.1f}%, z {np.nanpercentile(a[v],[1,50,99]).round(1) if v.sum() else None}')
"

# === dabeeo dsm_interpolation 빈칸 메움 (foundation 스크립트와 동일) ===
FILL_MODE=${FILL_MODE:-avg}
DSM_INTERP_SO="${REPO}/dsm_interpolation/dsm_interpolation.so"
in_tif="${out}/dsm-filtered.tif"; out_tif="${out}/dsm-filled.tif"
[ -f "$DSM_INTERP_SO" ] || { echo ".so 없음: $DSM_INTERP_SO"; exit 1; }
[ -f "$in_tif" ] || { echo "dsm-filtered 없음"; exit 1; }
echo "=== dabeeo fill (mode=${FILL_MODE}) ==="
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
echo "Log: ${LOGFILE}"
