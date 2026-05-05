"""
ResNet Encoder
Pretrained ResNet-18/50 kullanarak çok ölçekli özellik çıkarır.
Skip connection için 5 farklı seviyede feature map döner.

Referans: He et al., "Deep Residual Learning for Image Recognition", 2015
"""

import numpy as np
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet tabanlı encoder ağı.
    
    Girdi görüntüsünden 5 farklı çözünürlükte özellik haritası çıkarır.
    Her seviye bir öncekinin yarısı çözünürlüktedir.
    
    Args:
        num_layers (int): ResNet katman sayısı (18 veya 50)
        pretrained (bool): ImageNet pretrained ağırlıklar kullanılsın mı
    """

    def __init__(self, num_layers=18, pretrained=True):
        super().__init__()

        # Desteklenen ResNet varyantları
        assert num_layers in [18, 50], \
            f"ResNet-{num_layers} desteklenmiyor. 18 veya 50 kullanın."

        # Her seviyedeki kanal sayıları
        if num_layers == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        else:  # 50
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.num_ch_enc = np.array([64, 256, 512, 1024, 2048])

        # ResNet katmanlarını ayır
        self.conv1 = resnet.conv1        # 3 → 64, stride 2 (1/2)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool     # stride 2 (1/4)
        self.layer1 = resnet.layer1       # 1/4  çözünürlük
        self.layer2 = resnet.layer2       # 1/8  çözünürlük
        self.layer3 = resnet.layer3       # 1/16 çözünürlük
        self.layer4 = resnet.layer4       # 1/32 çözünürlük

        if pretrained:
            print(f"[Encoder] ResNet-{num_layers} (ImageNet pretrained) yüklendi")
        else:
            print(f"[Encoder] ResNet-{num_layers} (sıfırdan) oluşturuldu")

    def forward(self, x):
        """
        Args:
            x (Tensor): RGB görüntü [B, 3, H, W]
        
        Returns:
            list[Tensor]: 5 seviye özellik haritası
                features[0]: [B, 64,   H/2,  W/2 ]
                features[1]: [B, 64,   H/4,  W/4 ]
                features[2]: [B, 128,  H/8,  W/8 ]
                features[3]: [B, 256,  H/16, W/16]
                features[4]: [B, 512,  H/32, W/32]
        """
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)                         # Seviye 0: 1/2

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)                         # Seviye 1: 1/4

        x = self.layer2(x)
        features.append(x)                         # Seviye 2: 1/8

        x = self.layer3(x)
        features.append(x)                         # Seviye 3: 1/16

        x = self.layer4(x)
        features.append(x)                         # Seviye 4: 1/32

        return features
