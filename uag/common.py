import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    torch.save(value, tmp)
    os.replace(tmp, path)


def lock_config(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'config.json'
    if path.exists() and read_json(path) != config:
        raise ValueError('Output configuration differs: use a new directory: ' + str(directory))
    write_json(path, config)


def load_config(path):
    cfg = read_json(path)
    base = Path(path).resolve().parent
    if 'classes_file' in cfg:
        cf = Path(cfg['classes_file'])
        if not cf.is_absolute():
            cf = base / cf
        cfg['classes'] = sorted(cf.read_text().split())
    for key in ['data_root', 'dit_checkpoint', 'reference', 'minimax_checkpoint', 'ims3_checkpoint']:
        value = cfg.get(key)
        if value and not value.startswith('torchvision:') and not Path(value).is_absolute():
            cfg[key] = str(ROOT / value)
    classes = cfg['classes']
    if len(classes) != len(set(classes)) or not classes:
        raise ValueError('Class list must be nonempty and unique')
    all_classes = (ROOT / 'vendor/MinimaxDiffusion/misc/class_indices.txt').read_text().split()
    if not set(classes) <= set(all_classes):
        raise ValueError('Classes must be ImageNet WNIDs')
    cfg['imagenet_ids'] = [all_classes.index(c) for c in classes]
    return cfg


def paper_epochs(ipc):
    return 2000 if ipc <= 10 else 1500 if ipc <= 50 else 1000
