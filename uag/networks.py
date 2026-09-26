import torch
from torch import nn
from torchvision import models
from .common import ROOT

class WideBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        y = torch.relu(self.bn1(x))
        residual = self.shortcut(x if isinstance(self.shortcut, nn.Identity) else y)
        y = self.conv1(y)
        return residual + self.conv2(torch.relu(self.bn2(y)))


class WideResNet28(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        layers = []
        cin = 16
        for cout, stride in [(160, 1), (320, 2), (640, 2)]:
            for i in range(4):
                layers.append(WideBlock(cin, cout, stride if i == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(640)
        self.fc = nn.Linear(640, classes)

    def forward(self, x):
        x = torch.relu(self.bn(self.blocks(self.stem(x))))
        return self.fc(x.mean((2, 3)))


def build(arch, classes, role='student'):
    if arch == 'resnet18' and role in ('reference', 'teacher'):
        return models.resnet18(weights=None, num_classes=classes)
    if arch == 'convnet6':
        from vendor.MinimaxDiffusion.train_models.convnet import ConvNet
        return ConvNet(classes, net_depth=6, net_width=128, net_norm='instance', im_size=(224, 224))
    if arch == 'resnetap10':
        from vendor.MinimaxDiffusion.train_models.resnet_ap import ResNetAP
        return ResNetAP('imagenet', 10, classes, norm_type='instance', size=224)
    if arch == 'resnet18':
        from vendor.MinimaxDiffusion.train_models.resnet import ResNet
        return ResNet('imagenet', 18, classes, norm_type='instance', size=224)
    if arch == 'wrn28_10':
        return WideResNet28(classes)
    if arch == 'vit_tiny16':
        import timm
        return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=classes)
    raise ValueError(arch)


def reference(path, classes, device):
    if path == 'torchvision:resnet18':
        if len(classes) != 1000:
            raise ValueError('ImageNet pretrained reference requires all 1000 classes')
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Explicit map even if a caller supplies a noncanonical class ordering.
        all_classes = (ROOT / 'vendor/MinimaxDiffusion/misc/class_indices.txt').read_text().split()
        order = torch.tensor([all_classes.index(c) for c in classes])
        with torch.no_grad():
            model.fc.weight.copy_(model.fc.weight[order].clone())
            model.fc.bias.copy_(model.fc.bias[order].clone())
    else:
        ck = torch.load(path, map_location='cpu')
        if ck['classes'] != classes:
            raise ValueError('Reference class ordering differs from config')
        model = build(ck.get('arch', 'resnet18'), len(classes), ck.get('role', 'reference'))
        model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)
