import torch.nn as nn
import torch.nn.functional as F
import torch

"""
ResidualBlock은 ResNet18, ResNet34에 사용된다.
ResNet의 가장 기본적인 구성 요소로, 입력을 직접 다음 layer로 더해주는 "shortcut" 연결을 사용하여 기울기 소실 문제를 해결
[Residual Learning의 이점]
기울기 소실 문제 해결.
빠르고 효율적으로 수렴.
중요한 정보가 소실되지 않고 효과적으로 전달.
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # downsample: input, output의 차원이 다를 때 사용하는 downsample layer
        # 주 목적은 input, output 채널을 맞추기 위해 사용된다.
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    # forward 함수는 input을 받아서 block을 통과시키고 output을 생성하는 과정.
    def forward(self, x):
        identity = x
        #역할: input x를 identity에 저장, 입력값을 Residual에 더하는것이므로,
        # 입력을 따로 저장해두는 것이 필요.

        if self.downsample is not None:
            identity = self.downsample(x)
        # 이 조건문은 downsample이 정의된 경우
        # self.""is not None에 입력 텐서를 적절히 변환

        out = self.conv1(x)  # kernel_size 3x3
        out = self.bn1(out)  # 학습안정성 up, 기울기 소실/폭주 문제 down
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        # Shortcut 연결
        # Residual learning의 핵심이며, 입력 텐서 x를 연산 결과에 더해줌으로써 잔차를 학습
        out = F.relu(out)
        return out

"""
[Bottleneck]
병목 구조 사용
계산 효율성 및 성능 
residual learning 구현
"""


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.expansion = 4 # Bottleneck 확장 비율
        
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Downsample Layer
        if downsample is None and (stride != 1 or in_channels != out_channels * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = F.relu(out)

        return out


if __name__ == "__main__":
    # ResidualBlock 테스트
    block = ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128)
    ))
    x = torch.randn(1, 64, 56, 56)  # 입력 텐서
    out = block(x)
    print("ResidualBlock output shape:", out.shape)  # 예상 출력: [1, 128, 28, 28]

    # BottleneckBlock 테스트
    bottleneck = BottleneckBlock(in_channels=256, out_channels=64, stride=2)
    x = torch.randn(1, 256, 56, 56)  # 입력 텐서
    out = bottleneck(x)
    print("BottleneckBlock output shape:", out.shape)  # 예상 출력: [1, 256, 28, 28]