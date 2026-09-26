"""Section 4.3: frozen real partitions, exact removal diagnostic, joint permutation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from .common import load_config, write_json
from .data import ClassFolders, transform
from .networks import build, reference
from .guidance import entropy


def features(a):
    cfg = load_config(a.config)
    ds = ClassFolders(a.images, cfg['classes'], transform('eval'), a.ipc)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(a.device).eval().requires_grad_(False)
    ref = reference(cfg['reference'], cfg['classes'], a.device)
    xs, hs, ys = [], [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64, num_workers=a.workers):
            x = x.to(a.device)
            xs.append(F.normalize(model(x), dim=1).cpu().numpy())
            hs.append(entropy(ref(x)).cpu().numpy())
            ys.append(y.numpy())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, features=np.concatenate(xs), entropy=np.concatenate(hs),
                        labels=np.concatenate(ys), paths=np.array([p for p, _ in ds.samples]),
                        classes=np.array(cfg['classes']))


def focal_removal(model, images, labels):
    """Exact max_j |g_j dot(g_i - mean(g))/(B-1)|; focal i=0.

    Uses O(P) gradient storage rather than a B x P matrix. No optimizer steps.
    Eval mode makes each sample loss independent of other batch members.
    """
    n = len(labels)
    if n < 2:
        raise ValueError('Removal utility requires at least two samples')
    params = tuple(p for p in model.parameters() if p.requires_grad)
    mean_grad = torch.autograd.grad(F.cross_entropy(model(images), labels), params)
    focal_grad = torch.autograd.grad(F.cross_entropy(model(images[:1]), labels[:1]), params)
    difference = tuple((gi-gm).detach()/(n-1) for gi, gm in zip(focal_grad, mean_grad))
    del focal_grad, mean_grad
    maximum = images.new_zeros(())
    for j in range(1, n):
        peer = torch.autograd.grad(F.cross_entropy(model(images[j:j+1]), labels[j:j+1]), params)
        dot = sum((g*d).sum() for g, d in zip(peer, difference)).abs()
        maximum = torch.maximum(maximum, dot.detach())
    return float(maximum)


def utility(a):
    cfg = load_config(a.config)
    ds = ClassFolders(a.images, cfg['classes'], transform('eval'), a.ipc)
    if len(ds) < a.batch_size:
        raise ValueError('Dataset smaller than diagnostic batch size; reduce explicitly')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .common import lock_config, fingerprint
    lock_config(out, dict(config=cfg, images=a.images, ipc=a.ipc, batch=a.batch_size,
                         contexts=a.contexts, seed=a.seed,
                         checkpoint_hashes=[fingerprint(p) for p in a.checkpoints]))
    all_scores = []
    for ci, checkpoint in enumerate(a.checkpoints):
        ck = torch.load(checkpoint, map_location='cpu')
        if ck['classes'] != cfg['classes']:
            raise ValueError('Checkpoint class order mismatch')
        model = build(ck['arch'], len(cfg['classes']), ck['role']).to(a.device).eval()
        model.load_state_dict(ck['model'])
        scores = np.full((len(ds), a.contexts), np.nan, dtype=np.float64)
        cache = out / ('checkpoint_%d.npy' % ci)
        if cache.exists():
            scores = np.load(cache)
        for i in range(len(ds)):
            peers = np.delete(np.arange(len(ds)), i)
            for q in range(a.contexts):
                if np.isfinite(scores[i, q]):
                    continue
                # Paired contexts remain fixed across checkpoint and sampling method.
                rng = np.random.default_rng(a.seed*10000000+i*a.contexts+q)
                idx = [i] + rng.choice(peers, a.batch_size-1, replace=False).tolist()
                batch = [ds[j] for j in idx]
                x = torch.stack([v[0] for v in batch]).to(a.device)
                y = torch.tensor([v[1] for v in batch], device=a.device)
                scores[i, q] = focal_removal(model, x, y)
            if i % 10 == 0 or i == len(ds)-1:
                temporary = out / 'progress.tmp.npy'
                np.save(temporary, scores)
                temporary.replace(cache)
                print(json.dumps(dict(checkpoint=ci, sample=i, samples=len(ds))), flush=True)
        all_scores.append(scores)
        del model
    values = np.stack(all_scores).mean(axis=(0, 2))
    np.savez_compressed(out / 'utility.npz', utility=values,
                        paths=np.array([p for p, _ in ds.samples]), labels=np.array(ds.targets))


def spearman(x, y):
    from scipy.stats import spearmanr
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or len(np.unique(x[good])) < 2 or len(np.unique(y[good])) < 2:
        return np.nan
    return float(spearmanr(x[good], y[good]).statistic)


def region_values(features, labels, centroids):
    assignment = np.empty(len(labels), dtype=np.int64)
    for c, centers in enumerate(centroids):
        idx = np.flatnonzero(labels == c)
        # Features are 512D; cap temporary distance matrix by looping batches.
        for start in range(0, len(idx), 1024):
            chosen = idx[start:start+1024]
            dist = ((features[chosen, None] - centers[None])**2).sum(-1)
            assignment[chosen] = dist.argmin(1)
    return assignment


def region_stats(labels, assignments, values, classes, k):
    freq = np.zeros((classes, k))
    mean = np.full((classes, k), np.nan)
    for c in range(classes):
        denom = (labels == c).sum()
        if not denom:
            raise ValueError('Empty class in feature set')
        for r in range(k):
            mask = (labels == c) & (assignments == r)
            freq[c, r] = mask.sum()/denom
            if mask.any():
                mean[c, r] = values[mask].mean()
    return freq, mean


def joint_permutation(pairs, repeats, seed):
    """Independent within-class permutations at each K; two-sided across-K mean."""
    observed = np.array([spearman(x.ravel(), y.ravel()) for x, y in pairs])
    if not np.isfinite(observed).all():
        return dict(rho=None, p=None, reason='Undefined correlation for at least one K')
    statistic = observed.mean()
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repeats):
        permuted = []
        for x, y in pairs:
            # Restrict permutation to valid pairs, preserving empty-region masks.
            shuffled = y.copy()
            for c in range(len(y)):
                valid = np.isfinite(x[c]) & np.isfinite(y[c])
                shuffled[c, valid] = rng.permutation(y[c, valid])
            permuted.append(spearman(x.ravel(), shuffled.ravel()))
        if abs(np.mean(permuted)) >= abs(statistic):
            extreme += 1
    return dict(rho=float(statistic), p=(extreme+1)/(repeats+1), per_k=observed.tolist())


def regions(a):
    from sklearn.cluster import KMeans
    real, base, guided = [np.load(p) for p in [a.real, a.base, a.guided]]
    ub, ug = np.load(a.base_utility), np.load(a.guided_utility)
    if not np.array_equal(base['paths'], ub['paths']) or not np.array_equal(guided['paths'], ug['paths']):
        raise ValueError('Utility path order differs from feature order')
    if not np.array_equal(real['classes'], base['classes']) or not np.array_equal(real['classes'], guided['classes']):
        raise ValueError('Feature class lists differ')
    nc = len(real['classes'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = {key: [] for key in ['D_Fbase', 'D_Ubase', 'H_Ubase', 'Fbase_Ubase', 'Fuag_Uuag', 'Fuag_Ubase']}
    coverage = {}
    for k in [7, 10, 15, 20]:
        centers = []
        for c in range(nc):
            x = real['features'][real['labels'] == c]
            if len(x) < k:
                raise ValueError('Too few real images for K=%d class=%d' % (k, c))
            centers.append(KMeans(k, random_state=a.seed, n_init=10).fit(x).cluster_centers_)
        np.save(out / ('centroids_k%d.npy' % k), np.array(centers))
        ar, ab, ag = [region_values(v['features'], v['labels'], centers) for v in [real, base, guided]]
        density, h = region_stats(real['labels'], ar, real['entropy'], nc, k)
        fb, ubm = region_stats(base['labels'], ab, ub['utility'], nc, k)
        fg, ugm = region_stats(guided['labels'], ag, ug['utility'], nc, k)
        for key, x, y in [('D_Fbase', density, fb), ('D_Ubase', density, ubm), ('H_Ubase', h, ubm),
                           ('Fbase_Ubase', fb, ubm), ('Fuag_Uuag', fg, ugm), ('Fuag_Ubase', fg, ubm)]:
            pairs[key].append((x, y))
        coverage[str(k)] = dict(base_occupied=int(np.isfinite(ubm).sum()), uag_occupied=int(np.isfinite(ugm).sum()), total=nc*k)
        np.savez_compressed(out / ('regions_k%d.npz' % k), density=density, entropy=h,
                            base_frequency=fb, uag_frequency=fg, base_utility=ubm, uag_utility=ugm)
    report = {key: joint_permutation(value, a.permutations, a.seed) for key, value in pairs.items()}
    report.update(mean_base_utility=float(ub['utility'].mean()), mean_uag_utility=float(ug['utility'].mean()),
                  coverage=coverage, permutations=a.permutations,
                  empty_regions='Excluded pairwise; never filled with zero utility')
    write_json(out / 'report.json', report)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    for name, fn in [('features', features), ('utility', utility)]:
        q = sub.add_parser(name)
        q.set_defaults(func=fn)
        q.add_argument('--config', required=True)
        q.add_argument('--images', required=True)
        q.add_argument('--out', required=True)
        q.add_argument('--ipc', type=int)
        q.add_argument('--device', default='cuda')
        q.add_argument('--workers', type=int, default=4)
        if name == 'utility':
            q.add_argument('--checkpoints', nargs=3, required=True)
            q.add_argument('--contexts', type=int, default=5)
            q.add_argument('--batch-size', type=int, default=64)
            q.add_argument('--seed', type=int, default=0)
    q = sub.add_parser('regions')
    q.set_defaults(func=regions)
    for arg in ['real', 'base', 'guided', 'base-utility', 'guided-utility', 'out']:
        q.add_argument('--'+arg, required=True)
    q.add_argument('--permutations', type=int, default=10000)
    q.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
