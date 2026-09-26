"""Explicit region partition + ResNet18 labels; no hidden external relabeler."""
import argparse
from pathlib import Path
import torch
from .common import load_config, lock_config, write_json
from .data import ClassFolders, transform
from .networks import reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--syn', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ipc', type=int, required=True)
    p.add_argument('--grid', type=int, default=2, help='Manuscript omits region layout; declared default 2x2')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.grid < 1:
        p.error('--grid must be positive')
    cfg = load_config(a.config)
    teacher = reference(cfg['reference'], cfg['classes'], a.device)
    ds = ClassFolders(a.syn, cfg['classes'], ipc=a.ipc)
    out = Path(a.out)
    lock_config(out, dict(args=vars(a), config=cfg))
    (out / 'regions').mkdir(exist_ok=True)
    preprocess = transform('eval')
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            image, label = ds[i]
            w, h = image.size
            for gy in range(a.grid):
                for gx in range(a.grid):
                    box = (gx*w//a.grid, gy*h//a.grid, (gx+1)*w//a.grid, (gy+1)*h//a.grid)
                    crop = image.crop(box)
                    relative = 'regions/%08d_%d_%d.png' % (i, gy, gx)
                    crop.save(out / relative)
                    probs = teacher(preprocess(crop).unsqueeze(0).to(a.device)).softmax(-1)[0].cpu().tolist()
                    rows.append(dict(image=relative, source=ds.samples[i][0], box=box,
                                     hard_label=label, probabilities=probs))
    write_json(out / 'manifest.json', dict(classes=cfg['classes'], grid=a.grid, rows=rows))


if __name__ == '__main__':
    main()
