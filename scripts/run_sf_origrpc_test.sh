#!/bin/bash
# San Fernando MGM 테스트 — foundation 매처 격리용 (classic MGM stereo)
# 같은 3장/RPC로 MGM 돌려 정상 DSM 나오는지 확인.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_sf_origrpc_test.log"
: > "$LOGFILE"; exec > >(stdbuf -oL tee "$LOGFILE") 2>&1
echo "========== $(date '+%H:%M:%S') =========="
sudo -v
( while true; do sudo -n true 2>/dev/null; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) &
trap "kill $! 2>/dev/null || true" EXIT

DIR_DATA=/data/kevin_workspace/dataset_stereo
S2P_HD_REPO=/data/kevin_workspace/etc/s2p-hd
cfg_docker="/data/satellite/argentina/san_fernando/origrpc_test_config.json"
out_host="${DIR_DATA}/satellite/argentina/san_fernando/2_s2p-hd/output_origrpc_test"
ID_GPU=${ID_GPU:-7}

sudo rm -rf "$out_host"
INNER="set -e
pip3 install --quiet --root-user-action=ignore -e /workspace 2>&1 | tail -1
echo '=== s2p foundation(origRPC) ==='
stdbuf -oL s2p ${cfg_docker}
echo '=== done ==='
"
stdbuf -oL sudo docker run --rm --user "$(id -u):$(id -g)" \
    --gpus "device=${ID_GPU}" --shm-size=64g --net=host -e HOME=/tmp -e HF_HOME=/dl_cache/hf -e XDG_CACHE_HOME=/dl_cache/xdg -e TORCH_HOME=/dl_cache/torch \
    -v /data/kevin_workspace/pretrained_model/dl_cache:/dl_cache -v /data/kevin_workspace/pretrained_model:/pretrained:ro \
    -v "${DIR_DATA}":/data -v "${S2P_HD_REPO}":/workspace \
    --entrypoint bash s2p-hd-dl:latest -c "${INNER}"
rc=$?
echo ""
if [ -f "${out_host}/dsm.tif" ]; then
    echo "ORIGRPC DSM done:"
    python3 -c "
import rasterio,numpy as np
with rasterio.open('${out_host}/dsm.tif') as r:
    a=r.read(1); nd=r.nodata; v=np.isfinite(a)&(a!=nd if nd else True)
    print(f'  {r.width}x{r.height}, valid {100*v.sum()/a.size:.1f}%, z {np.nanpercentile(a[v],[1,50,99]).round(1) if v.sum() else None}')
    print(f'  bounds N: {r.bounds.bottom:.0f}-{r.bounds.top:.0f} (GT: 6184859-6189242)')
"
else
    echo "ORIGRPC FAILED (exit=$rc)"
fi
echo "Log: ${LOGFILE}"
