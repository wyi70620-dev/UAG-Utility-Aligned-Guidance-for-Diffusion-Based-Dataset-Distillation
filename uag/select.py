"""Random, K-center and herding real-data baselines; optional cached encoder features."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from .common import load_config, lock_config, write_json
from .data import ClassFolders


def choose(x, ipc, method, rng):
    if method == 'random':
        return rng.choice(len(x), ipc, replace=False).tolist()
    chosen = []
    if method == 'herding':
        mean = x.mean(0)
        total = np.zeros_like(mean)
        for i in range(ipc):
            distance = ((mean[None] - (x+total[None])/(i+1))**2).sum(1)
            distance[chosen] = np.inf
            index = int(distance.argmin())
            chosen.append(index)
            total += x[index]
    elif method == 'kcenter':
        chosen = [int(rng.integers(len(x)))]
        distance = ((x - x[chosen[0]])**2).sum(1)
        for _ in range(1, ipc):
            distance[chosen] = -np.inf
            index = int(distance.argmax())
            chosen.append(index)
            distance = np.minimum(distance, ((x-x[index])**2).sum(1))
    else:
        raise ValueError(method)
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ipc', type=int, required=True)
    p.add_argument('--method', choices=['random', 'kcenter', 'herding'], required=True)
    p.add_argument('--features', help='NPZ from diagnostics features, required except random')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cfg = load_config(a.config)
    ds = ClassFolders(Path(cfg['data_root']) / 'train', cfg['classes'])
    features = np.zeros((len(ds), 1))
    if a.method != 'random':
        if not a.features:
            p.error('Feature file required')
        archive = np.load(a.features)
        if not np.array_equal(archive['paths'], [p for p, _ in ds.samples]):
            raise ValueError('Feature sample ordering mismatch')
        features = archive['features']
    out = Path(a.out)
    lock_config(out, dict(args=vars(a), config=cfg))
    rng = np.random.default_rng(a.seed)
    rows = []
    for c, wnid in enumerate(cfg['classes']):
        ix = np.flatnonzero(np.array(ds.targets) == c)
        if len(ix) < a.ipc:
            raise ValueError('IPC exceeds real class size')
        selected = ix[choose(features[ix], a.ipc, a.method, rng)]
        (out / wnid).mkdir(exist_ok=True)
        for j, index in enumerate(selected):
            src = Path(ds.samples[index][0])
            target = out / wnid / ('%05d%s' % (j, src.suffix.lower()))
            shutil.copy2(src, target)
            rows.append(dict(source=str(src), target=str(target)))
    write_json(out / 'selection.json', rows)


if __name__ == '__main__':
    main()
