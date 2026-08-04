#!/usr/bin/env python3
"""
Compare the two bilinear-interpolation implementations at the operation level.

Instead of measuring the final disparity (which mixes the kernel difference with
32 iterations of refinement), this wraps FoundationStereo's bilinear_sampler and
runs BOTH implementations on the SAME input tensor at every call:

    out_cudnn  = F.grid_sample(...)                       # cuDNN kernel
    out_native = F.grid_sample(...) with cuDNN disabled   # PyTorch CUDA kernel

and reports statistics of |out_cudnn - out_native| per call and in aggregate.

It also splits the statistics by whether the sample coordinate falls inside the
volume, since out-of-bounds handling is a plausible source of disagreement.
"""
import argparse
import os
import sys

sys.path.insert(0, '/workspace')

import numpy as np
import torch


STATS = []          # one dict per bilinear_sampler call
SUBSAMPLE = []      # pooled |diff| sample for percentiles


def install_comparer(max_subsample=2_000_000):
    import s2p.dl_stereo as ds

    targets = [m for name, m in list(sys.modules.items())
               if m is not None and name.split('.')[-1] == 'geometry'
               and getattr(m, 'bilinear_sampler', None) is not None]
    if not targets:
        raise RuntimeError('geometry module not found')

    for mod in targets:
        orig = mod.bilinear_sampler
        # unwrap the production fallback wrapper if present
        orig = getattr(orig, '__wrapped_orig__', orig)

        def cmp_wrapper(img, coords, *a, _orig=orig, **kw):
            with torch.backends.cudnn.flags(enabled=True):
                out_c = _orig(img, coords, *a, **kw)
            with torch.backends.cudnn.flags(enabled=False):
                out_n = _orig(img, coords, *a, **kw)

            c = out_c[0] if isinstance(out_c, tuple) else out_c
            n = out_n[0] if isinstance(out_n, tuple) else out_n

            d = (c.float() - n.float()).abs()
            ref = c.float().abs()

            # normalised x coordinate, same formula bilinear_sampler uses
            W = img.shape[-1]
            xg = coords[..., 0:1]
            xg = 2 * xg / (W - 1) - 1
            oob = ((xg < -1) | (xg > 1))
            # broadcast the per-sample oob flag over the channel dim of the output
            oob_frac = float(oob.float().mean())

            rec = {
                'call': len(STATS),
                'in_shape': tuple(img.shape),
                'out_shape': tuple(c.shape),
                'dtype': str(c.dtype),
                'n': int(d.numel()),
                'max': float(d.max()),
                'mean': float(d.mean()),
                'signed_mean': float((c.float() - n.float()).mean()),
                'nonzero_frac': float((d > 0).float().mean()),
                'val_absmean': float(ref.mean()),
                'val_absmax': float(ref.max()),
                'oob_frac': oob_frac,
            }
            STATS.append(rec)

            if sum(x.size for x in SUBSAMPLE) < max_subsample:
                flat = d.flatten()
                k = min(200_000, flat.numel())
                idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
                SUBSAMPLE.append(flat[idx].cpu().numpy())

            return out_c

        cmp_wrapper.__wrapped_orig__ = orig
        mod.bilinear_sampler = cmp_wrapper
    return len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=1536)
    ap.add_argument('--ckpt', default='/pretrained/foundationstereo/23-51-11/model_best_bp2.pth')
    ap.add_argument('--seed-ref', required=True)
    ap.add_argument('--seed-sec', required=True)
    args = ap.parse_args()

    from s2p import dl_stereo
    from tools.probe_tile_trial import build_input

    cfg = {'dl_stereo_model': 'foundationstereo', 'dl_stereo_ckpt': args.ckpt,
           'dl_stereo_device': 'cuda:0', 'dl_depth_anything_v2_path': None,
           'dl_deterministic': False}
    model = dl_stereo.load_model(cfg)
    n_mod = install_comparer()
    print(f'comparer installed on {n_mod} module(s); size={args.size}', flush=True)

    imgL, imgR = build_input(args.seed_ref, args.seed_sec, args.size, 'cuda:0')
    dl_stereo._run_model(cfg, model, imgL, imgR)

    if not STATS:
        print('bilinear_sampler was never called'); return

    print(f'\nbilinear_sampler calls: {len(STATS)}')
    print(f"tensor dtype: {STATS[0]['dtype']}")
    print()
    hdr = (f"{'call':>4} {'input shape':>26} {'elems':>12} {'|val| mean':>11} "
           f"{'max|d|':>10} {'mean|d|':>11} {'rel mean':>10} {'differ%':>8} {'oob%':>7}")
    print(hdr)
    print('-' * len(hdr))
    for r in STATS[:12]:
        rel = r['mean'] / r['val_absmean'] if r['val_absmean'] else float('nan')
        print(f"{r['call']:>4} {str(r['in_shape']):>26} {r['n']:>12} {r['val_absmean']:>11.4f} "
              f"{r['max']:>10.5f} {r['mean']:>11.3e} {rel:>10.2e} "
              f"{100*r['nonzero_frac']:>7.2f}% {100*r['oob_frac']:>6.2f}%")
    if len(STATS) > 12:
        print(f'... ({len(STATS) - 12} more calls)')

    tot_n = sum(r['n'] for r in STATS)
    w_mean = sum(r['mean'] * r['n'] for r in STATS) / tot_n
    w_signed = sum(r['signed_mean'] * r['n'] for r in STATS) / tot_n
    w_val = sum(r['val_absmean'] * r['n'] for r in STATS) / tot_n
    w_nz = sum(r['nonzero_frac'] * r['n'] for r in STATS) / tot_n
    g_max = max(r['max'] for r in STATS)

    s = np.concatenate(SUBSAMPLE) if SUBSAMPLE else np.zeros(1)
    print()
    print('=== aggregate over every interpolated value ===')
    print(f'values compared      : {tot_n:,}')
    print(f'typical value |v|    : {w_val:.4f}   (max {max(r["val_absmax"] for r in STATS):.4f})')
    print(f'max |diff|           : {g_max:.6f}')
    print(f'mean |diff|          : {w_mean:.3e}   ({100*w_mean/w_val:.4f}% of typical value)')
    print(f'signed mean diff     : {w_signed:.3e}   (bias check: ~0 means symmetric)')
    print(f'values that differ   : {100*w_nz:.3f}%')
    for p in (50, 90, 99, 99.9, 100):
        print(f'  p{p:<5}|diff|       : {np.percentile(s, p):.6e}')
    print(f'out-of-bounds samples: {100*sum(r["oob_frac"]*r["n"] for r in STATS)/tot_n:.2f}%')


if __name__ == '__main__':
    main()
