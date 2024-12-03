import torch
import torch.nn as nn
from blocks import ResidualBlock, BottleneckBlock


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 초기 Conv Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Adaptive Pooling & Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * (4 if block == BottleneckBlock else 1), num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # Downsample 레이어 추가 (stride나 채널 수가 맞지 않을 경우)
        if stride != 1 or self.in_channels != out_channels * (4 if block == BottleneckBlock else 1):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * (4 if block == BottleneckBlock else 1),
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * (4 if block == BottleneckBlock else 1)),
            )

        layers = []
        # 첫 번째 블록
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * (4 if block == BottleneckBlock else 1)

        # 나머지 블록
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 초기 처리
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling 및 Fully Connected
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ResNet 모델 생성 함수
def ResNet18(num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes=num_classes)


# # 테스트 코드
# if __name__ == "__main__":
#     # ResNet50 모델 생성 및 출력 확인
#     model = ResNet50(num_classes=10)
#     print(model)

#     # 임의의 입력 데이터 테스트
#     x = torch.randn(1, 3, 224, 224)  # 배치 크기 1, 채널 3, 크기 224x224
#     output = model(x)
#     print("Output shape:", output.shape)  # 예상 출력: [1, 10]
