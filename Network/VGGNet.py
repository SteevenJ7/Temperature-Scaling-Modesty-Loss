import math
import torch.nn as nn


class VGGNet(nn.Module):

    def __init__(self, num_classes=100):
        super(VGGNet, self).__init__()
        self.features = make_layers(cfg['A'], batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Dropout().cuda(),
            nn.Linear(512, 512).cuda(),
            nn.ReLU(True).cuda(),
            nn.Dropout().cuda(),
            nn.Linear(512, 512).cuda(),
            nn.ReLU(True).cuda(),
            nn.Linear(512, num_classes).cuda(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2).cuda()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1).cuda()
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v).cuda(), nn.ReLU(inplace=True).cuda()]
            else:
                layers += [conv2d, nn.ReLU(inplace=True).cuda()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}
