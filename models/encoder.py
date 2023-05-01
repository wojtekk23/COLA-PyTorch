from enum import Enum
import torch
import torch.nn as nn
import torchvision.models


class EncoderBackend(Enum):
    EFFICIENTNET_B0 = 0


class Encoder(nn.Module):
    def __init__(self, backend_type: EncoderBackend, output_size: int):
        super(Encoder, self).__init__()
        self.backend = Encoder.get_backend(backend_type)
        self.fc = nn.Identity() if output_size == 1280 else nn.Linear(1280, output_size)

    def forward(self, x: torch.Tensor):
        x = self.backend(x)
        # x, _ = torch.max(x, (1, 2))
        x = self.fc(x)
        return x

    def get_backend(backend_type: EncoderBackend, pretrained: bool = False):
        if backend_type == EncoderBackend.EFFICIENTNET_B0:
            return torchvision.models.efficientnet_b0(pretrained=pretrained, num_classes=1280)
        # TODO
