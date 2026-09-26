"""Run pinned official Minimax/MGD3/IMS3 entrypoints with optional UAG injection.

The base sampler, mode guidance, fine-tuning and subgroup selection remain upstream.
Only the CFG noise prediction is wrapped; learned variance is unchanged.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path
from .common import ROOT, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['minimax', 'mgd3', 'ims3'], required=True)
    p.add_argument('--stage', choices=['sample', 'finetune', 'evaluate'], default='sample')
    p.add_argument('--config', required=True)
    p.add_argument('--scale', type=float, default=0)
    p.add_argument('--cutoff', type=int, default=795)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = load_config(a.config)
    repo = ROOT / 'vendor' / {'minimax': 'MinimaxDiffusion', 'mgd3': 'mode_guidance', 'ims3': 'IMS3'}[a.method]
    extra = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    if a.scale and a.stage != 'sample':
        p.error('UAG is sampling-only')
    if a.stage == 'finetune' and a.method == 'mgd3':
        p.error('MGD3 has no fine-tuning stage')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    # Fix a genuine upstream subset-label bug: global DiT label != subset ordinal.
    import data
    canonical = (repo / 'misc/class_indices.txt').read_text().split()
    data.ImageFolder.find_original_classes = lambda self: [canonical.index(c) for c in self.classes]
    if a.scale:
        import torch
        import models
        from diffusion import create_diffusion
        from diffusers import AutoencoderKL
        from .networks import reference
        from .guidance import UAGPrediction
        original = models.DiT.forward_with_cfg
        cached = {}

        def wrapped(self, x, t, y, cfg_scale):
            if id(self) not in cached:
                # Lazy loading must not change the base method's random trajectory.
                with torch.random.fork_rng(devices=[x.device.index or 0] if x.is_cuda else []):
                    ref = reference(cfg['reference'], cfg['classes'], x.device)
                    vae_kind = 'ema' if a.method == 'ims3' else 'mse'
                    if '--vae' in extra:
                        vae_kind = extra[extra.index('--vae')+1]
                    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-'+vae_kind,
                        local_files_only=cfg.get('offline', True)).to(x.device).eval().requires_grad_(False)
                    if '--vae-ckpt' in extra:
                        state = torch.load(extra[extra.index('--vae-ckpt')+1], map_location='cpu')
                        vae.load_state_dict(state['model'], strict=True)
                    self.eval().requires_grad_(False)
                    cached[id(self)] = UAGPrediction(lambda z, ts, **kw: original(self, z, ts, **kw),
                        vae, ref, create_diffusion('').alphas_cumprod, a.scale, a.cutoff)
            return cached[id(self)](x, t, y=y, cfg_scale=cfg_scale)
        models.DiT.forward_with_cfg = wrapped
    if a.stage == 'finetune':
        entry = 'train_dit.py'
    elif a.stage == 'evaluate':
        entry = 'train.py'
    else:
        entry = {'minimax': 'sample.py', 'mgd3': 'sample_mode_guidance.py', 'ims3': 'centroid.py'}[a.method]
    sys.argv = [str(repo / entry)] + extra
    if a.stage == 'evaluate':
        from .common import write_json
        output = None
        if '--save-dir' in sys.argv:
            i = sys.argv.index('--save-dir')
            output = sys.argv[i+1]
            del sys.argv[i:i+2]
        from argument import args
        if output:
            args.save_dir = output
        import train
        from misc.utils import Logger
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        results = []
        original_train = train.train
        def capture(*pos, **kw):
            best, last = original_train(*pos, **kw)
            results.append(dict(best_top1=float(best), final_top1=float(last)))
            return best, last
        train.train = capture
        train.main(args, Logger(args.save_dir), args.repeat)
        write_json(Path(args.save_dir)/'result.json', dict(method=a.method, seed=args.seed,
                   arch=args.net_type, ipc=args.ipc, repeats=results,
                   final_top1=sum(v['final_top1'] for v in results)/len(results)))
        return
    runpy.run_path(str(repo / entry), run_name='__main__')


if __name__ == '__main__':
    main()
