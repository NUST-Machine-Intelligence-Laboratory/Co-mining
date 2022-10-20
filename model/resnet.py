import torch.nn as nn
import torchvision
import torch.nn.functional as F

# class ResNet(nn.Module):
#     def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
#         super().__init__()
#         assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
#         resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
#         self.backbone = nn.Sequential(
#             resnet.conv1,
#             resnet.bn1,
#             resnet.relu,
#             resnet.maxpool,
#             resnet.layer1,
#             resnet.layer2,
#             resnet.layer3,
#             resnet.layer4,
#         )
#         self.feat_dim = resnet.fc.in_features
#         self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         if classifier == 'linear':
#             self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
#             init_weights(self.fc, init_method='He')
#         else:
#             raise AssertionError(f'{classifier} classifier is not supported.')
#         init_weights(self.project, init_method='He')

#     def forward(self, x):
#         N = x.size(0)
#         x = self.backbone(x)
#         x = self.neck(x).view(N, -1)
#         logits = self.fc(x)
#         return logits


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                  norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.bn3 = norm_layer(planes)

    def forward(self, x):
        # print(input[0].shape, )
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = 0.5 * out + 0.5 * identity
        out = self.relu(out)

        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, input):
        x, cut = input
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = 0.5*out + 0.5*shortcut
        return (out,cut)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, input):
        x, cut = input
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = 0.5*out + 0.5*self.shortcut(x)
        out = F.relu(out)
        return (out,cut)


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, input):
        x, cut = input
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = 0.5*out + 0.5*shortcut
        return (out,cut)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.block = block
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feat_dim = self.fc.in_features
        init_weights(self.fc, init_method='He')
    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1,):
        norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                             norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        # print(nn.Sequential(*layers))
        return nn.Sequential(*layers)
        # return layers[0]
    
    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.layer1( out)
        out = self.layer2( out)
        out = self.layer3( out)
        out = self.layer4( out)

        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        representation = out.view(out.size(0), -1)
        out = self.fc(representation)
        return out, representation


def PreActResNet18(num_classes):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes)

def ResNet18(num_classes):
    model =  ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    model_dict = model.state_dict()
    pretrained_dict =  torchvision.models.__dict__['resnet18'](pretrained=True).state_dict()
    
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.shape==model_dict[k].shape}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

    
class ResNetfeat(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, classifier='linear',):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.fc, init_method='He')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        feat = self.neck(x).view(N, -1)
        logits = self.fc(feat)
        return logits, feat

class ResNetfeatproj(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True,  classifier='linear', project_dim=128):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.project_dim = project_dim
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.fc, init_method='He')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.project = nn.Linear(in_features=self.feat_dim, out_features= self.project_dim)
        init_weights(self.project, init_method='He')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.fc(x)
        feat = self.project(x)
        return logits, feat
    
    
def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)