import torch.nn as nn


""" Optional conv block """
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels)
    ) 

""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_1 = conv_block(3, 64, kernel_size=1, padding=0)
        self.layer1 = conv_block(3, 64)
        self.layer2 = conv_block(64, 64)
        self.layer3 = conv_block(64, 64)

        self.layer4_1 = conv_block(64, 96, kernel_size=1, padding=0)
        self.layer4 = conv_block(64, 96)
        self.layer5 = conv_block(96, 96)
        self.layer6 = conv_block(96, 96)

        self.layer7_1 = conv_block(96, 128, kernel_size=1, padding=0)
        self.layer7 = conv_block(96, 128)
        self.layer8 = conv_block(128, 128)
        self.layer9 = conv_block(128, 128)

        self.layer10_1 = conv_block(128, 256, kernel_size=1, padding=0)
        self.layer10 = conv_block(128, 256)
        self.layer11 = conv_block(256, 256)
        self.layer12 = conv_block(256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        identity = self.layer1_1(x)
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.layer3(out) + identity
        out = self.relu(out)
        out = self.maxpool(out)

        identity = self.layer4_1(out)
        out = self.relu(self.layer4(out))
        out = self.relu(self.layer5(out))
        out = self.layer6(out) + identity
        out = self.relu(out)
        out = self.maxpool(out)

        identity = self.layer7_1(out)
        out = self.relu(self.layer7(out))
        out = self.relu(self.layer8(out))
        out = self.layer9(out) + identity
        out = self.relu(out)
        out = self.maxpool(out)

        identity = self.layer10_1(out)
        out = self.relu(self.layer10(out))
        out = self.relu(self.layer11(out))
        out = self.layer12(out) + identity
        out = self.relu(out)

        return out.view(out.size(0),-1)
