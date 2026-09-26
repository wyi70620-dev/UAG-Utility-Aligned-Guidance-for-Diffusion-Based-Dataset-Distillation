"""Reference, hard-label and soft-label students; checkpointed diagnostic stages."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .common import (load_config, lock_config, seed_all, rng_state, restore_rng,
                     save_checkpoint, write_json, paper_epochs, read_json, fingerprint)
from .data import ClassFolders, RegionTargets, transform
from .networks import build, reference


def cutmix(x, y):
    lam = np.random.beta(1, 1)
    order = torch.randperm(len(y), device=x.device)
    h, w = x.shape[-2:]
    cx, cy = np.random.randint(w), np.random.randint(h)
    cw, ch = int(w*np.sqrt(1-lam)), int(h*np.sqrt(1-lam))
    x1, x2 = max(0, cx-cw//2), min(w, cx+cw//2)
    y1, y2 = max(0, cy-ch//2), min(h, cy+ch//2)
    x[:, :, y1:y2, x1:x2] = x[order, :, y1:y2, x1:x2]
    return x, y, y[order], 1 - (x2-x1)*(y2-y1)/(w*h)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, count = 0, 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
        count += len(y)
    if count == 0:
        raise ValueError('Empty evaluation set')
    return 100 * correct / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--role', choices=['reference', 'student', 'soft'], default='student')
    p.add_argument('--arch', choices=['convnet6', 'resnetap10', 'resnet18', 'wrn28_10', 'vit_tiny16'], default='resnet18')
    p.add_argument('--syn')
    p.add_argument('--regions')
    p.add_argument('--indices', help='JSON list of synthetic indices, used by redundancy experiment')
    p.add_argument('--ipc', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--teacher', help='For online relabeling of the exact augmented region')
    p.add_argument('--fixed-soft-targets', action='store_true', help='Use saved pre-augmentation targets; explicit alternative protocol')
    a = p.parse_args()
    cfg = load_config(a.config)
    classes = cfg['classes']
    role = a.role
    epochs = a.epochs or (200 if role == 'reference' else 300 if role == 'soft' else paper_epochs(a.ipc))
    batch = a.batch_size or (256 if role in ('reference', 'soft') else 64)
    if epochs < 1 or batch < 1:
        raise ValueError('Epochs/batch size must be positive')
    seed_all(a.seed)
    torch.set_num_threads(4)
    indices = read_json(a.indices) if a.indices else None
    if role == 'reference':
        tr = ClassFolders(Path(cfg['data_root']) / 'train', classes, transform('reference'))
    elif role == 'soft':
        if not a.regions or a.arch != 'resnet18':
            p.error('Soft protocol requires --regions and --arch resnet18')
        tr = RegionTargets(a.regions, transform('soft'))
        if tr.meta['classes'] != classes:
            raise ValueError('Region target class order mismatch')
    else:
        if not a.syn:
            p.error('Student requires --syn')
        tr = ClassFolders(a.syn, classes, transform('hard'), a.ipc, indices)
    va = ClassFolders(Path(cfg['data_root']) / 'val', classes, transform('eval'))
    train_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=a.workers, pin_memory=True)
    val_loader = DataLoader(va, batch_size=batch, num_workers=a.workers, pin_memory=True)
    # For ImageNet soft protocol use standard BN ResNet-18, hard protocol uses IN.
    model_role = 'reference' if role in ('reference', 'soft') else 'student'
    model = build(a.arch, len(classes), model_role).to(a.device)
    teacher = None
    if role == 'soft' and not a.fixed_soft_targets:
        teacher = reference(a.teacher or cfg['reference'], classes, a.device)
    if role == 'reference':
        opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [2*epochs//3, 5*epochs//6], gamma=.2)
    out = Path(a.out)
    run = dict(args=vars(a), config=cfg, actual_epochs=epochs, actual_batch_size=batch,
               training_count=len(tr), validation_count=len(va),
               indices_sha256=fingerprint(a.indices) if a.indices else None,
               regions_sha256=fingerprint(a.regions) if a.regions else None)
    lock_config(out, run)
    if (out / 'result.json').exists():
        print('Already completed: ' + str(out))
        return
    start, best = 0, 0.
    if (out / 'resume.pt').exists():
        ck = torch.load(out / 'resume.pt', map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        # Move loaded optimizer tensors explicitly for torch 1.13.
        for value in opt.state.values():
            for key, tensor in value.items():
                if torch.is_tensor(tensor):
                    value[key] = tensor.to(a.device)
        scheduler.load_state_dict(ck['scheduler'])
        restore_rng(ck['rng'])
        start, best = ck['epoch'], ck['best']
    diagnostic_epochs = {max(1, math.ceil(epochs*f)): name for f, name in [(0.25, '25'), (.5, '50'), (.75, '75')]}
    for epoch in range(start + 1, epochs + 1):
        tick = time.perf_counter()
        model.train()
        total = 0.
        for x, y in train_loader:
            x, y = x.to(a.device), y.to(a.device)
            opt.zero_grad(set_to_none=True)
            if role == 'student':
                x, y1, y2, lam = cutmix(x, y)
                logits = model(x)
                loss = lam*F.cross_entropy(logits, y1)+(1-lam)*F.cross_entropy(logits, y2)
            elif role == 'soft':
                with torch.no_grad():
                    target = teacher(x).softmax(-1) if teacher is not None else y
                loss = -(target * F.log_softmax(model(x), -1)).sum(-1).mean()
            else:
                loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
        scheduler.step()
        acc = None
        if epoch % 10 == 0 or epoch in (1, epochs):
            acc = accuracy(model, val_loader, a.device)
            best = max(best, acc)
        row = dict(epoch=epoch, loss=total/len(tr), top1=acc, lr=opt.param_groups[0]['lr'], seconds=time.perf_counter()-tick)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        ck = dict(model=model.state_dict(), classes=classes, arch=a.arch, role=model_role, epoch=epoch)
        if epoch in diagnostic_epochs:
            save_checkpoint(out / ('checkpoint_' + diagnostic_epochs[epoch] + '.pt'), ck)
        if epoch % 50 == 0 or epoch in diagnostic_epochs or epoch == epochs:
            save_checkpoint(out / 'resume.pt', dict(ck, optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(), best=best))
        if epoch == epochs:
            save_checkpoint(out / 'final.pt', ck)
            write_json(out / 'result.json', dict(arch=a.arch, role=role, ipc=a.ipc, seed=a.seed,
                       epochs=epochs, final_top1=acc, best_top1=best, training_count=len(tr)))


if __name__ == '__main__':
    main()
