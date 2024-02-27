import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1. dropout
2. normalization
3. learning rate decay
4. residual connection
5. network depth
"""

"""
网络结构拟采用 resnet
网络深度改为 resnet 相应的块数
"""


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channel: int,
        dropout: bool = False,
        normalize: bool = False,
        has_residual: bool = False,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel) if normalize else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel) if normalize else nn.Identity(),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()
        self.has_residual = has_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.has_residual:
            # res = output + x  #! 为什么 output += x 会报错
            output = output + x
        output = F.relu(self.dropout(output))
        return output
        # return F.relu(self.dropout(res))


class MyModule(nn.Module):
    def __init__(
        self,
        depth: int,
        dropout: bool = False,
        normalize: bool = False,
        has_residual: bool = False,
    ):
        super().__init__()
        in_channel = 3
        self.channel = 64
        self.expansion = 2
        """
        64*64*3
        32*32*64
        16*16*64
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, self.channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.channel) if normalize else nn.Identity(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        for _ in range(depth):
            self.model.append(
                ResidualBlock(self.channel, dropout, normalize, has_residual)
            )
            self.model.append(
                nn.Conv2d(self.channel, self.channel * self.expansion, kernel_size=1)
            )
            self.model.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=1))
            self.channel *= self.expansion
        self.model.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(self.channel, 200))
        self.model.append(nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    import torchvision
    import cv2
    import numpy as np

    x0 = cv2.imread("../data/tiny-imagenet-200/test/images/test_0.JPEG")
    x0 = torchvision.transforms.ToTensor()(x0)
    x1 = cv2.imread("../data/tiny-imagenet-200/test/images/test_1.JPEG")
    x1 = torchvision.transforms.ToTensor()(x1)
    x = torch.stack((x0, x1))
    model = torchvision.models.resnet34()
    y = model(x)
    print("finish")
