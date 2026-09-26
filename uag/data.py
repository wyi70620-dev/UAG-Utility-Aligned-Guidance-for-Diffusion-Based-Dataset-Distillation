from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def transform(kind='eval', size=224):
    if kind == 'reference':
        ops = [T.RandomResizedCrop(size), T.RandomHorizontalFlip()]
    elif kind == 'hard':
        # Explicit engineering choice where the manuscript only says random-crop.
        ops = [T.Resize(size), T.CenterCrop(size),
               T.RandomResizedCrop(size, scale=(0.5, 1.0)), T.RandomHorizontalFlip()]
    elif kind == 'soft':
        ops = [T.Resize((size, size)), T.RandAugment(num_ops=2, magnitude=9),
               T.RandomHorizontalFlip()]
    else:
        ops = [T.Resize(round(size * 256 / 224)), T.CenterCrop(size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class ClassFolders(Dataset):
    """Explicit ordered WNIDs prevent classifier/generator label misalignment."""
    def __init__(self, root, classes, transform=None, ipc=None, indices=None):
        self.root = Path(root)
        self.classes = list(classes)
        self.transform = transform
        samples = []
        for y, c in enumerate(classes):
            folder = self.root / c
            if not folder.is_dir():
                raise FileNotFoundError(str(folder))
            rows = sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS)
            if not rows or (ipc is not None and len(rows) < ipc):
                raise ValueError('Insufficient images for ' + c)
            samples.extend((str(p), y) for p in (rows[:ipc] if ipc else rows))
        self.samples = samples if indices is None else [samples[i] for i in indices]
        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as f:
            image = f.convert('RGB')
        return (self.transform(image) if self.transform else image), label


class RegionTargets(Dataset):
    """Targets describe exact saved crops. Teacher sees the SAME augmented crop."""
    def __init__(self, manifest, augmentation=None):
        from .common import read_json
        self.manifest = Path(manifest)
        self.meta = read_json(manifest)
        self.rows = self.meta['rows']
        self.transform = augmentation or transform('eval')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.manifest.parent / row['image']) as f:
            image = self.transform(f.convert('RGB'))
        return image, torch.tensor(row['probabilities'], dtype=torch.float32)
