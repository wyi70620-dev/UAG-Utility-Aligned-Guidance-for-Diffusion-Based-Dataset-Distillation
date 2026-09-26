import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from torchvision.utils import save_image
from .common import ROOT, load_config, lock_config, seed_all, fingerprint, write_json
from .guidance import UAGPrediction
from .networks import reference


def main():
    p = argparse.ArgumentParser(description='DiT / UAG DDPM generation; executes only when invoked')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ipc', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--scale', type=float, default=10)
    p.add_argument('--cutoff', type=int, default=795)
    p.add_argument('--checkpoint')
    p.add_argument('--reference')
    p.add_argument('--device', default='cuda')
    p.add_argument('--deterministic', action='store_true')
    a = p.parse_args()
    if a.ipc < 1 or not 0 <= a.cutoff <= 999:
        p.error('IPC must be positive; cutoff must be 0..999')
    cfg = load_config(a.config)
    ckpt_path = a.checkpoint or cfg['dit_checkpoint']
    ref_path = a.reference or cfg['reference']
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(ckpt_path)
    if a.deterministic:
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'vendor/MinimaxDiffusion'))
    from models import DiT_models
    from diffusion import create_diffusion
    from diffusers import AutoencoderKL
    dit = DiT_models['DiT-XL/2'](input_size=32, num_classes=1000).to(a.device)
    state = torch.load(ckpt_path, map_location='cpu')
    if 'ema' in state:
        state = state['ema']
    elif 'model' in state:
        state = state['model']
    incompatible = dit.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not k.endswith(('.gamma1', '.gamma2')) for k in incompatible.missing_keys):
        raise RuntimeError(str(incompatible))
    parameters = dict(dit.named_parameters())
    if any(not torch.all(parameters[k] == 1) for k in incompatible.missing_keys):
        raise RuntimeError('Missing DiffFit scales must be initialized to identity')
    vae = AutoencoderKL.from_pretrained(cfg['vae'], local_files_only=cfg.get('offline', True)).to(a.device)
    ref = reference(ref_path, cfg['classes'], a.device) if a.scale else None
    for model in (dit, vae, ref):
        if model is not None:
            model.eval().requires_grad_(False)
    diffusion = create_diffusion('50')
    guided = UAGPrediction(dit.forward_with_cfg, vae, ref, create_diffusion('').alphas_cumprod,
                           a.scale, a.cutoff, resize_on_cpu=a.deterministic)
    out = Path(a.out)
    metadata = dict(config=cfg, ipc=a.ipc, seed=a.seed, scale=a.scale, cutoff=a.cutoff,
                    checkpoint=ckpt_path, checkpoint_sha256=fingerprint(ckpt_path),
                    reference=ref_path if a.scale else None, deterministic=a.deterministic,
                    reference_sha256=fingerprint(ref_path) if a.scale and Path(ref_path).is_file() else None,
                    timesteps=diffusion.timestep_map, cfg=4., cfg_channels=3,
                    missing_identity_parameters=list(incompatible.missing_keys))
    lock_config(out, metadata)
    started = time.perf_counter()
    for c, imagenet_id in zip(cfg['classes'], cfg['imagenet_ids']):
        folder = out / c
        folder.mkdir(exist_ok=True)
        for i in range(a.ipc):
            path = folder / ('%05d.png' % i)
            record_path = folder / ('%05d.json' % i)
            if path.exists() and record_path.exists():
                continue
            # ImageNet ID keeps the same sample seed across subset order and IPC changes.
            seed = a.seed * 10000000 + imagenet_id * 10000 + i
            seed_all(seed)
            z = torch.randn(1, 4, 32, 32, device=a.device)
            noise_hash = hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()
            z = torch.cat([z, z])
            y = torch.tensor([imagenet_id, 1000], device=a.device)
            guided.trace = []
            tic = time.perf_counter()
            with torch.no_grad():
                latents = diffusion.p_sample_loop(guided, z.shape, z, clip_denoised=False,
                    model_kwargs=dict(y=y, cfg_scale=4.), device=a.device, progress=False)
                image = vae.decode(latents[:1] / .18215).sample
            if not torch.isfinite(image).all():
                raise FloatingPointError('Nonfinite image')
            tmp = path.with_suffix('.tmp.png')
            save_image(image, tmp, normalize=True, value_range=(-1, 1))
            tmp.replace(path)
            rng = torch.cuda.get_rng_state(a.device) if str(a.device).startswith('cuda') else torch.get_rng_state()
            record = dict(wnid=c, index=i, seed=seed, initial_noise_sha256=noise_hash,
                          final_rng_sha256=hashlib.sha256(rng.cpu().numpy().tobytes()).hexdigest(),
                          seconds=time.perf_counter()-tic, trace=guided.trace)
            write_json(record_path, record)
            print(json.dumps({k: v for k, v in record.items() if k != 'trace'}), flush=True)
    write_json(out / 'complete.json', dict(images=len(cfg['classes'])*a.ipc,
               invocation_seconds=time.perf_counter()-started, includes_resumed_work=False))


if __name__ == '__main__':
    main()
