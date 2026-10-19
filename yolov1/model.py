import torch
from torch import nn
"""
(kernel_size, outputfeatures, stride, padding)
'M' represents MaxPoolLayer
padding = (Kernel_size -1)/2
"""
architecture_config = [
    (7, 64, 2, 3),
    'M',
    (3, 192,1, 1),
    'M',
    (1,128, 1,0),
    (3, 256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    'M',
    [(1,256,1,0), (3,512,1,1), 4],
    (1,512,1,0),
    (3,1024,1,1),
    'M',
    [(1,512,1,0), (3,1024,1,1), 2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1)
]
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.CNN = nn.Conv2d(self.in_channels, out_channels, bias= False, **kwargs)
    def forward(self, x):
        x = self.leaky_relu(self.batch_norm(self.CNN(x)))
        return x

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture_config = architecture_config
        self.in_channels = in_channels
        self.darknet = self._conv_layers(self.architecture_config)
        self.fcs = self._conn_layers(**kwargs)
        
    def _conv_layers(self, architecture):
        in_channels = self.in_channels
        layers = []
        for x in architecture:
            if type(x) == tuple:
                layers +=[
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size = x[0],
                        padding = x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers +=[
                    nn.MaxPool2d(2)
                ]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                repeat_num = x[2]
                for _ in range(repeat_num):
                    layers +=[
                        CNNBlock(
                        in_channels,
                        conv1[1],
                        kernel_size = conv1[0],
                        padding = conv1[3]
                    )
                    ]
                    in_channels = conv1[1]
                    layers += [
                        CNNBlock(
                        in_channels,
                        conv2[1],
                        kernel_size = conv2[0],
                        padding = conv2[3]
                    )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    def _conn_layers(self, split_size=7, class_size=20, num_boxes=2):
        S = split_size
        B = num_boxes
        C = class_size
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C + (B*5)))
        )
    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(x)
        return x
