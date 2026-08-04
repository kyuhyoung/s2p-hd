#!/usr/bin/env python3
"""
Find the maximum tile size each DL stereo model can process before CUDA OOM.

Method: exponential bracketing -> binary search on the OOM threshold ->
repeat-confirmation at the winner.

Why binary search and not a fit-and-extrapolate: the predicate we care about
("does this OOM?") depends on reserved + fragmented + workspace memory, not on
torch.cuda.max_memory_allocated(), and the attention path (FoundationStereo,
DepthAnythingV2) is not guaranteed linear in pixel count. So we measure the
predicate directly.

Search variable is the side N of a square NxN input, i.e. area N^2 — area is
the physical invariant (cost volume ~ (H/4)(W/4)D with D fixed, attention
tokens ~ HW/patch^2), not the s2p `tile_size` scalar.

Each trial runs in a fresh subprocess: an OOM leaves the caching allocator
fragmented and would break the monotonicity the bisection relies on.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

REPO = '/workspace'
PROBE = os.path.join(REPO, 'tools', 'probe_tile_trial.py')
ALLOC_CONF = os.environ.get('PROBE_ALLOC_CONF', '').strip()
TAG = '_expandable' if 'expandable' in ALLOC_CONF else ''
TRIAL_LOG_DIR = os.path.join(REPO, 'max_tile_logs' + TAG)

SEED_DIR = ('/data/satellite/jax/jax_214_all_ba_including_config/s2p_out_dl_foundation'
            '/tiles/row_0000000_height_1000/col_0000000_width_1000/pair_1')
SEED_REF = os.path.join(SEED_DIR, 'rectified_ref.tif')
SEED_SEC = os.path.join(SEED_DIR, 'rectified_sec.tif')

DAV2 = '/pretrained/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth'

# The 4 models exactly as configured in configs/jax_214/
MODELS = [
    {'label': 'dl_stereo (diachronic-monster)', 'model': 'monster',
     'ckpt': '/pretrained/diachronic-stereo/final.pth', 'dav2': DAV2},
    {'label': 'monster (mix_all)', 'model': 'monster',
     'ckpt': '/pretrained/monster/mix_all.pth', 'dav2': DAV2},
    {'label': 'foundationstereo', 'model': 'foundationstereo',
     'ckpt': '/pretrained/foundationstereo/23-51-11/model_best_bp2.pth', 'dav2': None},
    {'label': 'stereoanywhere', 'model': 'stereoanywhere',
     'ckpt': '/pretrained/stereoanywhere/stereoanywhere_sceneflow.pth', 'dav2': DAV2},
]

GRANULARITY = 64      # stop bisecting below this — the OOM boundary is stochastic
CONFIRM_RUNS = 3      # repeats at the winner to reject a lucky pass
MAX_STEPDOWNS = 3     # how far to back off if confirmation fails
START = 512
CEILING = 8192

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def run_trial(m, size, gpu):
    """One trial in a fresh process. Returns (ok, result_dict)."""
    cmd = [sys.executable, '-u', PROBE,
           '--model', m['model'], '--label', m['label'], '--ckpt', m['ckpt'],
           '--size', str(size), '--device', 'cuda:0',
           '--seed-ref', SEED_REF, '--seed-sec', SEED_SEC]
    if m['dav2']:
        cmd += ['--dav2', m['dav2']]

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Keep stderr: when a trial dies of something other than plain OOM we need
    # the traceback to tell a capacity limit from a real bug.
    os.makedirs(TRIAL_LOG_DIR, exist_ok=True)
    slug = m['label'].split()[0].replace('/', '_')
    errlog = os.path.join(TRIAL_LOG_DIR, f'{slug}_{size}.stderr.log')

    t0 = time.time()
    with open(errlog, 'w') as ef:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=ef,
                                env=env, text=True, bufsize=1)
        result = None
        for line in proc.stdout:
            if line.startswith('RESULT:'):
                result = json.loads(line[len('RESULT:'):])
        proc.wait()
    elapsed = time.time() - t0

    if result is None:
        result = {'status': 'error', 'msg': f'no result (exit={proc.returncode})',
                  'peak_alloc_mb': None, 'peak_reserved_mb': None}
    ok = result['status'] == 'ok'

    peak = result.get('peak_alloc_mb')
    resv = result.get('peak_reserved_mb')
    detail = f"alloc={peak}MB resv={resv}MB" if peak is not None else ''
    if result['status'] not in ('ok', 'oom'):
        detail = (detail + '  ' + result.get('msg', '')).strip()
    log(f"  [{m['label']:<30s} gpu{gpu}] {size:>5d}x{size:<5d} "
        f"{result['status']:<5s} {detail}  ({elapsed:.0f}s)")
    return ok, result


def search_model(m, gpu):
    """Bracket -> bisect -> confirm. Returns a summary dict."""
    log(f"=== {m['label']} (gpu{gpu}) ===")
    trials = 0
    peaks = {}
    last_fail = {}

    # --- 1. exponential bracketing ---
    lo, hi = 0, None
    size = START
    while size <= CEILING:
        ok, r = run_trial(m, size, gpu)
        trials += 1
        if r['status'] == 'error':
            return {'label': m['label'], 'max_side': None, 'trials': trials,
                    'note': f"trial error: {r.get('msg', '')}"}
        if ok:
            lo = size
            peaks[size] = r
            size *= 2
        else:
            hi = size
            last_fail = r
            break

    if lo == 0:
        return {'label': m['label'], 'max_side': None, 'trials': trials,
                'note': f'OOM even at {START}x{START}'}
    if hi is None:
        return {'label': m['label'], 'max_side': lo, 'trials': trials,
                'peak': peaks.get(lo), 'gpu': gpu,
                'note': f'no OOM up to ceiling {CEILING}'}

    # --- 2. binary search on the threshold ---
    log(f"  bracket: pass={lo}, fail={hi} -> bisecting to {GRANULARITY}px")
    while hi - lo > GRANULARITY:
        mid = (lo + hi) // 2
        mid = (mid // GRANULARITY) * GRANULARITY   # keep multiple-of-32 friendly
        if mid <= lo:
            break
        ok, r = run_trial(m, mid, gpu)
        trials += 1
        if r['status'] == 'error':
            break
        if ok:
            lo = mid
            peaks[mid] = r
        else:
            hi = mid
            last_fail = r

    # --- 3. confirmation: the boundary is stochastic, so repeat ---
    stepdowns = 0
    while stepdowns <= MAX_STEPDOWNS and lo > 0:
        log(f"  confirming {lo}x{lo} ({CONFIRM_RUNS - 1} extra runs)")
        stable = True
        for _ in range(CONFIRM_RUNS - 1):
            ok, r = run_trial(m, lo, gpu)
            trials += 1
            if not ok:
                stable = False
                break
            peaks[lo] = r
        if stable:
            break
        lo -= GRANULARITY
        stepdowns += 1
        log(f"  unstable -> stepping down to {lo}")

    mode = last_fail.get('status', '')
    note = f"boundary={mode}"
    if mode == 'fail':
        note += f" ({last_fail.get('msg', '')[:60]})"
    return {'label': m['label'], 'max_side': lo, 'trials': trials,
            'peak': peaks.get(lo), 'gpu': gpu, 'note': note,
            'fail_size': hi, 'fail_detail': last_fail}


def main():
    # Report the seed tile's rectified size so N can be mapped back to an
    # s2p `tile_size` for this dataset.
    try:
        sys.path.insert(0, REPO)
        import rasterio
        with rasterio.open(SEED_REF) as f:
            log(f"seed rectified pair: {f.width}x{f.height} "
                f"(from s2p tile_size=1000, jax_214)")
    except Exception as e:
        log(f"seed info unavailable: {e}")

    # PROBE_MODELS=foundationstereo re-runs a single model (labels matched as
    # substrings) without redoing the ones already measured.
    sel = os.environ.get('PROBE_MODELS', '').strip()
    models = MODELS
    if sel:
        keys = [s.strip() for s in sel.split(',') if s.strip()]
        models = [m for m in MODELS if any(k in m['label'] for k in keys)]
        if not models:
            log(f'no model matches PROBE_MODELS={sel}')
            return

    gpus = [int(g) for g in os.environ.get('PROBE_GPUS', '0,1,2').split(',')]
    log(f"GPUs: {gpus}   models: {len(models)}")
    log(f"granularity={GRANULARITY}px  confirm_runs={CONFIRM_RUNS}  "
        f"bracket={START}..{CEILING}")
    log(f"allocator: PYTORCH_CUDA_ALLOC_CONF={ALLOC_CONF or '(default)'}")
    log('')

    gpu_pool = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)

    results = [None] * len(models)

    def worker(i, m):
        gpu = gpu_pool.get()          # one model per GPU at a time — sharing a
        try:                          # GPU would corrupt the measurement
            results[i] = search_model(m, gpu)
        finally:
            gpu_pool.put(gpu)

    threads = [threading.Thread(target=worker, args=(i, m))
               for i, m in enumerate(models)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log('')
    log('=' * 78)
    log(f'MAX TILE SIZE (square, 24GB Quadro RTX 6000, 1 proc/GPU, alloc={ALLOC_CONF or "default"})')
    log('=' * 78)
    log(f"{'model':<32s} {'max NxN':>12s} {'Mpx':>7s} {'alloc':>8s} {'resv':>8s}  note")
    log('-' * 78)
    for r in results:
        if r is None:
            continue
        if r['max_side']:
            n = r['max_side']
            pk = r.get('peak') or {}
            log(f"{r['label']:<32s} {n:>5d}x{n:<6d} {n * n / 1e6:>7.1f} "
                f"{str(pk.get('peak_alloc_mb')) + 'MB':>8s} "
                f"{str(pk.get('peak_reserved_mb')) + 'MB':>8s}  {r['note']}")
        else:
            log(f"{r['label']:<32s} {'FAILED':>12s} {'':>7s} {'':>8s} {'':>8s}  {r['note']}")
    log('-' * 78)
    log('Recommended production value: ~90% of max side (~80% of area) — the OOM')
    log('boundary moves with allocator fragmentation between runs.')
    log('Divide the area budget by max_processes if several s2p workers share a GPU.')

    # Merge into any previous run's results so a single-model re-run does not
    # discard the models already measured.
    out = os.path.join(REPO, f'max_tile_results{TAG}.json')
    merged = {}
    if os.path.exists(out):
        try:
            for r in json.load(open(out)):
                if r:
                    merged[r['label']] = r
        except Exception:
            pass
    for r in results:
        if r:
            merged[r['label']] = r
    with open(out, 'w') as f:
        json.dump([merged[m['label']] for m in MODELS if m['label'] in merged], f, indent=2)
    log('raw results -> max_tile_results.json')


if __name__ == '__main__':
    main()
