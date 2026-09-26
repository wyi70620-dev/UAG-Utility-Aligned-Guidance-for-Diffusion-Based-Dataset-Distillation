"""Equations 17-21. No denoiser gradients, clipping, or entropy thresholds."""
import torch
from torch.nn import functional as F


def entropy(logits):
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def epsilon_correction(x, epsilon, alpha, utility, scale):
    """Sum independent per-image entropy to avoid batch-size-dependent strength."""
    with torch.enable_grad():
        leaf = x.detach().requires_grad_(True)
        clean = (leaf - (1 - alpha).sqrt() * epsilon.detach()) / alpha.sqrt()
        values = utility(clean)
        gradient, = torch.autograd.grad(values.sum(), leaf)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite utility gradient; no silent gradient clipping')
    return epsilon.detach() - scale * (1 - alpha).sqrt() * gradient.detach(), values.detach()


class UAGPrediction:
    """Wrap a DiT CFG callable; input t MUST already be original 0..999 time."""
    def __init__(self, predictor, vae, classifier, alphas, scale=10.0, cutoff=795,
                 latent_scale=0.18215, duplicate_cfg=True, resize_on_cpu=False):
        self.predictor, self.vae, self.classifier = predictor, vae, classifier
        self.alphas = torch.as_tensor(alphas, dtype=torch.float32)
        self.scale, self.cutoff = scale, cutoff
        self.latent_scale = latent_scale
        self.duplicate_cfg = duplicate_cfg
        self.resize_on_cpu = resize_on_cpu
        self.trace = []

    def utility(self, clean):
        # Keep this entire path differentiable, including VAE and normalization.
        rgb = (self.vae.decode(clean / self.latent_scale).sample + 1) / 2
        device = rgb.device
        if self.resize_on_cpu:
            rgb = rgb.cpu()
        rgb = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False).to(device)
        mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
        std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        return entropy(self.classifier((rgb - mean) / std))

    def __call__(self, x, t, **kwargs):
        with torch.no_grad():
            prediction = self.predictor(x, t, **kwargs).detach()
        if self.scale == 0:
            return prediction
        if (t < 0).any() or (t >= len(self.alphas)).any():
            raise ValueError('Expected original diffusion timesteps')
        n = x.shape[0] // 2 if self.duplicate_cfg else x.shape[0]
        if self.duplicate_cfg and x.shape[0] != 2 * n:
            raise ValueError('CFG batch must have conditional/unconditional halves')
        mask = t[:n] <= self.cutoff
        if not mask.any():
            return prediction
        channels = x.shape[1]
        alpha = self.alphas.to(x.device)[t[:n][mask].long()].view(-1, 1, 1, 1)
        eps = prediction[:n, :channels][mask]
        corrected, values = epsilon_correction(x[:n][mask], eps, alpha, self.utility, self.scale)
        delta = torch.zeros_like(prediction[:n, :channels])
        delta[mask] = corrected - eps
        result = prediction.clone()
        result[:n, :channels] += delta
        if self.duplicate_cfg:
            result[n:, :channels] += delta
        # Learned variance channels remain exactly unchanged.
        self.trace.append({'t': int(t[0]), 'entropy': float(values.mean()),
                           'correction_l2': float(delta.flatten(1).norm(dim=1).mean())})
        return result
