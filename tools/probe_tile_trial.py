#!/usr/bin/env python3
"""
Single VRAM trial for one DL stereo model at one tile size.

Runs exactly one forward pass on an NxN input and reports peak VRAM.
Designed to be spawned as a FRESH PROCESS per trial: an OOM leaves the CUDA
caching allocator fragmented, which would break the monotonicity assumption
that the binary search in max_tile_search.py depends on.

Exit codes:
    0  success
    2  CUDA OOM
    3  other error

Prints exactly one JSON line to stdout (prefixed RESULT:) plus free-form
progress to stderr.
"""
import argparse
import json
import os
import sys
import time

# Default: keep allocator behaviour identical to a production s2p run — we want
# the threshold that actually applies, not an idealised one. PROBE_ALLOC_CONF
# overrides it (e.g. 'expandable_segments:True') to measure the same models
# under a different allocator strategy.
_alloc_conf = os.environ.get('PROBE_ALLOC_CONF', '').strip()
if _alloc_conf:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = _alloc_conf
else:
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)


def build_input(seed_ref, seed_sec, size, device):
    """Return two [1, 3, size, size] tensors in [0, 1] built from a real
    rectified pair (tiled to fill the requested size).

    VRAM depends on shape, not content, but using real texture keeps the
    inference path realistic (no degenerate all-constant shortcuts)."""
    import torch
    sys.path.insert(0, '/workspace')
    from s2p.dl_stereo import _read_rectified_image

    L = _read_rectified_image(seed_ref)  # [1, 3, h, w]
    R = _read_rectified_image(seed_sec)

    h, w = L.shape[-2], L.shape[-1]
    reps_h = -(-size // h)
    reps_w = -(-size // w)
    L = L.repeat(1, 1, reps_h, reps_w)[:, :, :size, :size].contiguous()
    R = R.repeat(1, 1, reps_h, reps_w)[:, :, :size, :size].contiguous()
    return L, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['monster', 'stereoanywhere', 'foundationstereo'])
    ap.add_argument('--label', default=None, help='display name (e.g. dl_stereo)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dav2', default=None)
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed-ref', required=True)
    ap.add_argument('--seed-sec', required=True)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--save-disp', default=None,
                    help='dump the disparity map to this .npy for kernel comparison')
    args = ap.parse_args()

    label = args.label or args.model
    result = {
        'model': args.model,
        'label': label,
        'size': args.size,
        'status': 'error',
        'peak_alloc_mb': None,
        'peak_reserved_mb': None,
        'nvsmi_peak_mb': None,
        'sec': None,
        'msg': '',
    }

    def emit(code):
        print('RESULT:' + json.dumps(result), flush=True)
        sys.exit(code)

    try:
        sys.path.insert(0, '/workspace')
        import torch
        from s2p import dl_stereo

        cfg = {
            'dl_stereo_model': args.model,
            'dl_stereo_ckpt': args.ckpt,
            'dl_stereo_device': args.device,
            'dl_depth_anything_v2_path': args.dav2,
            'dl_deterministic': args.deterministic,
        }

        t0 = time.time()
        model = dl_stereo.load_model(cfg)
        imgL, imgR = build_input(args.seed_ref, args.seed_sec, args.size, args.device)

        torch.cuda.reset_peak_memory_stats(args.device)
        t1 = time.time()
        disp = dl_stereo._run_model(cfg, model, imgL, imgR)
        torch.cuda.synchronize(args.device)

        result['sec'] = round(time.time() - t1, 1)
        result['load_sec'] = round(t1 - t0, 1)
        result['peak_alloc_mb'] = round(torch.cuda.max_memory_allocated(args.device) / 2**20)
        result['peak_reserved_mb'] = round(torch.cuda.max_memory_reserved(args.device) / 2**20)
        result['out_shape'] = list(disp.shape)
        result['status'] = 'ok'
        if args.save_disp:
            import numpy as np
            np.save(args.save_disp, disp)
            result['saved'] = args.save_disp
        emit(0)

    except Exception as e:  # noqa: BLE001 — we classify below
        import traceback
        msg = f'{type(e).__name__}: {e}'
        result['msg'] = msg.split('\n')[0][:300]
        low = str(e).lower()
        is_oom = ('OutOfMemoryError' in type(e).__name__
                  or 'out of memory' in low)
        # cuDNN/cuBLAS report a generic failure when they cannot secure their
        # workspace, and cuDNN also refuses tensors past its own size limits.
        # Both mean "this tile size is not usable" for our purposes, but they
        # are NOT plain OOM — keep them as a separate 'fail' status so the
        # failure mode stays visible in the report.
        is_capacity = ('cudnn_status_not_supported' in low
                       or 'cudnn_status_alloc_failed' in low
                       or 'cublas_status_alloc_failed' in low
                       or 'cudnn error' in low
                       or 'cublas' in low)
        try:
            import torch
            result['peak_alloc_mb'] = round(torch.cuda.max_memory_allocated(args.device) / 2**20)
            result['peak_reserved_mb'] = round(torch.cuda.max_memory_reserved(args.device) / 2**20)
        except Exception:
            pass
        if is_oom:
            result['status'] = 'oom'
            traceback.print_exc(file=sys.stderr)
            emit(2)
        if is_capacity:
            result['status'] = 'fail'
            traceback.print_exc(file=sys.stderr)
            emit(2)
        result['status'] = 'error'
        traceback.print_exc(file=sys.stderr)
        emit(3)


if __name__ == '__main__':
    main()
