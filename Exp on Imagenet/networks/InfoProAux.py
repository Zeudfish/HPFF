import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, inplanes):
        super(Decoder, self).__init__()
        self.decoder1 = nn.Sequential(
            nn.Conv2d(inplanes, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=[112, 112], mode='bilinear', align_corners=True),
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(inplanes, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=[56, 56], mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[112, 112], mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(inplanes, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=[112, 112], mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=[56, 56], mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[112, 112], mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(inplanes, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(size=[14, 14], mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=[28, 28], mode='bilinear', align_corners=True),
            nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[56, 56], mode='bilinear', align_corners=True),
            nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Upsample(size=[112, 112], mode='bilinear', align_corners=True),
            nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True),
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, features):
        if features.size(3) == 56:
            return self.decoder1(features)
        elif features.size(3) == 28:
            return self.decoder2(features)
        elif features.size(3) == 14:
            return self.decoder3(features)
        elif features.size(4) == 7:
            return self.decoder4(features)





class AuxClassifier(nn.Module):
    def __init__(self, inplanes):
        super(AuxClassifier, self).__init__()
        if inplanes == 256:
            self.head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 1000)
            )
        elif inplanes == 512:
            self.head = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 1000)
            )
        elif inplanes == 1024:
            self.head = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 1000)
            )
        elif inplanes == 2048:
            self.head = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 1000)
            )

    def forward(self,features):
        return self.head(features)
