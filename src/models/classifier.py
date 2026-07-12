"""PyTorch classifier module for AISlop detection."""

import torch
import torch.nn as nn
import timm


class AISlopClassifier(nn.Module):
    def __init__(self, num_classes=2, backbone_name="efficientnet_b3", pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        in_features = self._detect_in_features()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def _detect_in_features(self):
        dummy = torch.randn(1, 3, 224, 224)
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(dummy)
        return features.shape[1]

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def extract_features(self, x):
        return self.backbone(x)
