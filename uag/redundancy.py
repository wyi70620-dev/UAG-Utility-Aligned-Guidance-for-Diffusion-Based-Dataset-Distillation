"""Section 4.4, five disjoint stratified subsets and all 25 cross accuracies."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .common import ROOT, load_config, seed_all, write_json, lock_config
from .data import ClassFolders, transform
from .networks import build
from .train import accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--syn', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ipc', type=int, default=50)
    p.add_argument('--arch', default='resnet18')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--execute', action='store_true', help='Without this flag, writes only split and command plan')
    a = p.parse_args()
    cfg = load_config(a.config)
    seed_all(a.seed)
    ds = ClassFolders(a.syn, cfg['classes'], transform('eval'), a.ipc)
    if a.ipc % 5:
        p.error('IPC must be divisible by 5')
    out = Path(a.out).resolve()
    lock_config(out, dict(config=cfg, syn=str(Path(a.syn).resolve()), ipc=a.ipc, arch=a.arch, seed=a.seed))
    rng = np.random.default_rng(a.seed)
    splits = [[] for _ in range(5)]
    for c in range(len(cfg['classes'])):
        indices = rng.permutation(np.flatnonzero(np.array(ds.targets) == c))
        for j, values in enumerate(np.split(indices, 5)):
            splits[j].extend(values.tolist())
    commands = []
    for j, indices in enumerate(splits):
        split_path = out / ('split%d.json' % j)
        write_json(split_path, indices)
        # Each subset has IPC/5 images; train scheduling reflects that budget.
        from .common import paper_epochs
        commands.append([sys.executable, '-m', 'uag.train', '--config', str(Path(a.config).resolve()),
            '--syn', str(Path(a.syn).resolve()), '--out', str(out / ('student%d' % j)),
            '--arch', a.arch, '--ipc', str(a.ipc), '--indices', str(split_path),
            '--epochs', str(paper_epochs(a.ipc//5)), '--seed', str(a.seed*10+j), '--device', a.device])
    write_json(out / 'commands.json', commands)
    if not a.execute:
        print('Split/command plan written; no training started.')
        return
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)
    matrix = np.zeros((5, 5))
    for i in range(5):
        ck = torch.load(out / ('student%d/final.pt' % i), map_location='cpu')
        model = build(a.arch, len(cfg['classes'])).to(a.device)
        model.load_state_dict(ck['model'])
        for j in range(5):
            matrix[i, j] = accuracy(model, DataLoader(Subset(ds, splits[j]), batch_size=64), a.device)
        del model
    diagonal = float(np.diag(matrix).mean())
    off_diagonal = float(matrix[~np.eye(5, dtype=bool)].mean())
    write_json(out / 'report.json', dict(matrix=matrix.tolist(), Rself=diagonal,
               Rcross=off_diagonal, Gcross=diagonal-off_diagonal))


if __name__ == '__main__':
    main()
