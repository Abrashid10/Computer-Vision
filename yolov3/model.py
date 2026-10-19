import torch
from torch import nn
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,bn_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn = bn_act
    def forward(self, x):
        if self.use_bn:
            x = self.leaky(self.bn(self.conv(x)))
        else:
            x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual = True, num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_repeats = num_repeats
        for _ in range(self.num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size = 1),
                    CNNBlock(channels //2 , channels, kernel_size = 3, padding = 1)
                )
            ]
        self.use_residual = use_residual
    def forward(self, x):
        for layers in self.layers:
            if self.use_residual:
                x = x + layers(x)
            else:
                x = layers(x)
        return x

class ScalePredictions(nn.Module):
    def __init__(self,in_channels, num_classes = 80):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2, kernel_size = 3, padding = 1),
            CNNBlock(in_channels * 2, (num_classes + 5) * 3, bn_act = False, kernel_size = 1)
        )
        self.num_classes = num_classes
    def forward(self, x):
        x = self.pred(x).reshape(x.shape[0], 3, (self.num_classes+5), x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        return x
#(filters, kernel_size, stride) 
class YOLOv3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 80):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self._conv_layers()

    def forward(self,x):
        output = []
        root_connections = []
        for layer in self.conv_layers:
            if isinstance(layer, ScalePredictions):
                output.append(layer(x))
                continue
                
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                root_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, root_connections[-1]], dim=1)
                root_connections.pop()
        return output
            
    def _conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in self.config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 1 if kernel_size == 3 else 0)
                )
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeat = module[1]
                layers.append(
                    ResidualBlock(in_channels, num_repeats = num_repeat)
                )
            elif isinstance(module, str):
                if module == 'S':
                    layers +=[
                        ResidualBlock(in_channels, use_residual = False, num_repeats = 1),
                        CNNBlock(in_channels, in_channels // 2, bn_act = False, kernel_size = 1),
                        ScalePredictions(in_channels//2, num_classes = self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == 'U':
                    layers.append(
                        nn.Upsample(scale_factor = 2)
                    )
                    in_channels = in_channels * 3 # in_channels = channels_from_upsample + channels_from_route
        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
