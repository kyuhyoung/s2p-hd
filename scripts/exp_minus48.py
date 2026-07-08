# -48 실패 원인 분리 실험: 크기(0 근처) vs 부호 혼재 vs 방향(거울)
# sec 이미지를 수평 이동(Δ)하면 모든 s2p-시차가 +Δ만큼 균일 이동한다는 성질 이용.
#
# 입력물: daejeon/lower/2_s2p-hd/tiletest_AFTER (정준화 성공 런), tiletest_BAD (invert 강제 런)
#   재생성법: scripts/tiletest_daejeon_lower.sh (AFTER는 그대로, BAD는
#   dl_orientation_canonical에 임시 invert 분기 필요 — real_before_after.png 참조)
# 실행 (docker, GPU 1개):
#   docker run --rm --user "$(id -u):$(id -g)" --gpus '"device=0"' --shm-size=16g \
#     -e HOME=/tmp -e HF_HOME=/dl_cache/hf -e XDG_CACHE_HOME=/dl_cache/xdg -e TORCH_HOME=/dl_cache/torch \
#     -v /data/kevin_workspace/pretrained_model/dl_cache:/dl_cache \
#     -v /data/kevin_workspace/dataset_stereo:/data -v /data/kevin_workspace/etc/s2p-hd:/workspace \
#     -v /data/kevin_workspace/pretrained_model:/pretrained:ro \
#     --entrypoint bash s2p-hd-dl:latest -c \
#     "pip3 install -q -e /workspace && python3 /workspace/scripts/exp_minus48.py"
# 결과(2026-07-08 실측): 방향 좋으면 -48/-10도 정확히 발견, 나쁘면 -48/-300 모두 지면값 스냅
#   -> 실패 원인 = 방향(거울 세계). s2p-양수 시차는 표현불가(0 클램프)도 확인.
import os, sys, json, warnings
import numpy as np
import rasterio
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace')
from s2p import dl_stereo
from s2p.config import get_default_config

SCRATCH = '/data/satellite/daejeon/lower/2_s2p-hd'  # (docker 내 경로; 호스트=/data/kevin_workspace/dataset_stereo/...)
CASES = {
    'good': dict(tile=f'{SCRATCH}/tiletest_AFTER/tiles/row_0016140_height_1614/col_0014913_width_1657',
                 tower=(1020, 1001), ground=(1013, 1201)),
    'bad':  dict(tile=f'{SCRATCH}/tiletest_BAD/tiles/row_0016140_height_1614/col_0014913_width_1657',
                 tower=(1879, 1001), ground=(1866, 1195)),
}
# (케이스, Δpx, 라벨)
EXPS = [
    ('good',    0, '1) GOOD 원본          (기대: 지면 -50, 옥상 -165)'),
    ('good',  +40, '2) GOOD sec+40        (기대: 지면 -10, 옥상 -125) — 0근처+좋은방향'),
    ('good', +117, '3) GOOD sec+117       (기대: 지면 +67, 옥상 -48) — 옥상을 -48에, 방향 좋음'),
    ('bad',     0, '4) BAD 원본           (기대: 지면 -179, 옥상 -48) — 나쁜 방향'),
    ('bad',  -252, '5) BAD sec-252        (기대: 지면 -431, 옥상 -300) — 나쁜 방향+깊은 앵커'),
]

cfg = get_default_config()
user = json.load(open('/data/satellite/daejeon/lower/foundation_config_tiletest.json'))
for k in ('dl_stereo_model', 'dl_stereo_ckpt'):
    cfg[k] = user[k]
cfg['dl_stereo_device'] = 'cuda:0'

model = dl_stereo.load_model(cfg)

def shift_write(src_path, delta, out_path):
    with rasterio.open(src_path) as ds:
        a = ds.read()
        prof = ds.profile.copy()
    out = np.zeros_like(a)
    if delta == 0:
        out = a
    elif delta > 0:
        out[:, :, delta:] = a[:, :, :-delta]
    else:
        out[:, :, :delta] = a[:, :, -delta:]
    with rasterio.open(out_path, 'w', **prof) as ds:
        ds.write(out)

def probe(disp, x, y, r=3):
    w = disp[y-r:y+r+1, x-r:x+r+1]
    v = w[np.isfinite(w)]
    return (np.median(v) if v.size else float('nan')), v.size

print('=' * 88)
for case, delta, label in EXPS:
    C = CASES[case]
    ref = f"{C['tile']}/pair_2/rectified_ref.tif"
    sec = f"{C['tile']}/pair_2/rectified_sec.tif"
    sec_use = sec
    if delta != 0:
        sec_use = f'/tmp/sec_shift_{case}_{delta:+d}.tif'
        shift_write(sec, delta, sec_use)
    disp = dl_stereo.predict_disparity(cfg, model, ref, sec_use)
    (tx, ty), (gx, gy) = C['tower'], C['ground']
    dt, nt = probe(disp, tx, ty)
    dg, ng = probe(disp, gx, gy)
    exp_t = {'good': -165, 'bad': -48}[case] + delta
    exp_g = {'good': -50, 'bad': -179}[case] + delta
    ok_t = 'O' if np.isfinite(dt) and abs(dt - exp_t) < 15 else 'X'
    ok_g = 'O' if np.isfinite(dg) and abs(dg - exp_g) < 15 else 'X'
    print(label)
    print(f'   옥상: 기대 {exp_t:+.0f}  실측 {dt:+.1f}  {ok_t}    |    지면: 기대 {exp_g:+.0f}  실측 {dg:+.1f}  {ok_g}')
    print('-' * 88)
