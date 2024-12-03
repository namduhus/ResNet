import torch
import torch.nn as nn
from blocks import ResidualBlock, BottleneckBlock 
# models.blocks의 정의한 블록 불러오기


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResidualBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * (4 if block == BottleneckBlock else 1), num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * (4 if block == BottleneckBlock else 1):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * (4 if block == BottleneckBlock else 1), kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * (4 if block == BottleneckBlock else 1)),
            )
        
        
        layers = []
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * (4 if block == BottleneckBlock else 1)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
      
# ResNet 모델 생성
def ResNet18(num_classes=100):
    return ResNet(ResidualBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=100):
    return ResNet(ResidualBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=100):
    return ResNet(BottleneckBlock, [3,4,6,3], num_classes=num_classes)
  
def ResNet101(num_classes = 100):
    return ResNet(BottleneckBlock, [3,4,23,3],num_classes=num_classes)

def ResNet152(num_classes = 100):
    return ResNet(BottleneckBlock, [3,8,36,3], num_classes=num_classes)
  

# ## Test
# if __name__ == "__main__":
#     model = ResNet34(num_classes=10)
#     print(model)
    
#     # 임의의 입력 데이터 test
#     x = torch.randn(1, 3, 224, 224) # 배치크기 1, channel 3, 224x224
#     output = model(x)
#     print("Output-shape", output.shape)
    