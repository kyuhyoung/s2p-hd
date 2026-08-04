#!/usr/bin/env python3
"""
Does last-bit rounding at the interpolation step explain the final disparity gap?

compare_grid_sample.py showed the cuDNN and PyTorch-native grid_sample kernels
agree to ~1 ULP, yet the final disparity maps differ by mean 0.205 px. The claim
that the refinement loop amplifies ULP noise into that gap was an inference, not
a measurement. This tests it directly.

Instead of switching kernels, keep cuDNN and inject an artificial perturbation
with the SAME profile as the measured kernel discrepancy: move ~11% of the
interpolated values by one ULP in a random direction. If the resulting final
disparity diverges by a comparable amount, amplification is confirmed.

Runs in one process so the model is loaded once:
    clean_a, clean_b   -> in-process run-to-run noise floor
    pert_a, pert_b     -> two perturbed runs (different seeds)
"""
import argparse
import sys

sys.path.insert(0, '/workspace')

import numpy as np
import torch

PERTURB = {'on': False, 'frac': 0.11, 'seed': 0, 'calls': 0, 'changed': 0, 'total': 0}


def _perturb(x):
    """Move a random fraction of entries by exactly one ULP, random direction."""
    g = torch.Generator(device=x.device)
    g.manual_seed(PERTURB['seed'] * 100003 + PERTURB['calls'])
    mask = torch.rand(x.shape, device=x.device, generator=g) < PERTURB['frac']
    up = torch.rand(x.shape, device=x.device, generator=g) < 0.5
    direction = torch.where(up,
                            torch.full_like(x, float('inf')),
                            torch.full_like(x, float('-inf')))
    out = torch.where(mask, torch.nextafter(x, direction), x)
    PERTURB['calls'] += 1
    PERTURB['changed'] += int((out != x).sum())
    PERTURB['total'] += x.numel()
    return out


def install(mod_filter='geometry'):
    targets = [m for name, m in list(sys.modules.items())
               if m is not None and name.split('.')[-1] == mod_filter
               and getattr(m, 'bilinear_sampler', None) is not None]
    if not targets:
        raise RuntimeError('geometry module not found')
    for mod in targets:
        orig = getattr(mod.bilinear_sampler, '__wrapped_orig__', mod.bilinear_sampler)

        def wrapper(*a, _orig=orig, **kw):
            out = _orig(*a, **kw)
            if not PERTURB['on']:
                return out
            if isinstance(out, tuple):
                return (_perturb(out[0]),) + tuple(out[1:])
            return _perturb(out)

        wrapper.__wrapped_orig__ = orig
        mod.bilinear_sampler = wrapper
    return len(targets)


def stats(a, b):
    d = np.abs(a - b)
    return {
        'mean': d.mean(), 'rms': float(np.sqrt(((a - b) ** 2).mean())),
        'p50': np.percentile(d, 50), 'p99': np.percentile(d, 99),
        'p99.9': np.percentile(d, 99.9), 'max': d.max(),
        'f01': 100 * (d > 0.1).mean(), 'f1': 100 * (d > 1).mean(),
        'f5': 100 * (d > 5).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=1536)
    ap.add_argument('--frac', type=float, default=0.11,
                    help='fraction of values to nudge (measured kernel value: 0.11)')
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
    n = install()
    PERTURB['frac'] = args.frac
    print(f'perturber installed on {n} module(s); size={args.size} frac={args.frac}', flush=True)

    imgL, imgR = build_input(args.seed_ref, args.seed_sec, args.size, 'cuda:0')

    def run(perturb_on, seed):
        PERTURB['on'] = perturb_on
        PERTURB['seed'] = seed
        PERTURB['calls'] = 0
        return dl_stereo._run_model(cfg, model, imgL, imgR)

    print('run 1/4: clean a', flush=True);  clean_a = run(False, 0)
    print('run 2/4: clean b', flush=True);  clean_b = run(False, 0)
    print('run 3/4: perturbed a', flush=True); pert_a = run(True, 1)
    ch, tot = PERTURB['changed'], PERTURB['total']
    print('run 4/4: perturbed b', flush=True); pert_b = run(True, 2)

    print(f'\nactually nudged: {100 * ch / tot:.2f}% of {tot:,} interpolated values '
          f'(target {100 * args.frac:.0f}%)')

    cols = {
        'ULP주입 vs clean': stats(clean_a, pert_a),
        'ULP주입 b vs clean': stats(clean_a, pert_b),
        'clean vs clean (노이즈)': stats(clean_a, clean_b),
    }
    try:
        cols['cuDNN vs native (참조)'] = stats(np.load('/workspace/fs_verify/safe_cudnn.npy'),
                                             np.load('/workspace/fs_verify/safe_native.npy'))
    except Exception:
        pass

    rows = [('mean |d|', 'mean', '{:.6f}'), ('RMS', 'rms', '{:.6f}'),
            ('p50', 'p50', '{:.6f}'), ('p99', 'p99', '{:.6f}'),
            ('p99.9', 'p99.9', '{:.4f}'), ('max', 'max', '{:.4f}'),
            ('>0.1px %', 'f01', '{:.3f}'), ('>1px %', 'f1', '{:.4f}'),
            ('>5px %', 'f5', '{:.4f}')]
    w = 24
    print()
    print(f"{'':<12}" + ''.join(f'{k:>{w}}' for k in cols))
    print('-' * (12 + w * len(cols)))
    for label, key, fmt in rows:
        print(f'{label:<12}' + ''.join(f'{fmt.format(c[key]):>{w}}' for c in cols.values()))


if __name__ == '__main__':
    main()
