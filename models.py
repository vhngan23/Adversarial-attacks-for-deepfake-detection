from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict
import torch 
from math import ceil, exp
import math


class conv_bn(nn.Module):
  def __init__(self, in_channels, out_channels,ks = 3, **kwargs):
    super(conv_bn,self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,ks,padding=int((ks-1)/2),**kwargs)
    self.batch_norm = nn.BatchNorm2d(out_channels)
  
  def forward(self,x):
    return self.batch_norm(self.conv(x))


class separable_conv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(separable_conv,self).__init__()
    self.point_wise_conv = nn.Conv2d(in_channels,out_channels,1)
    self.depth_wise_conv = nn.Conv2d(out_channels,out_channels,3, padding=1,groups=out_channels)
    self.batch_norm = nn.BatchNorm2d(out_channels)

  def forward(self,x):  
    return self.batch_norm(self.depth_wise_conv(self.point_wise_conv(x)))

class entry_flow(nn.Module):
  def __init__(self, in_channels=3):
    super(entry_flow, self).__init__()
    self.conv1 = conv_bn(in_channels,32,stride = 2  )
    self.relu = nn.ReLU()
    self.conv2 = conv_bn(32, 64)

    self.sepConv1 = separable_conv(64,128)
    self.sepConv2 = separable_conv(128,128)
    self.pool1 = nn.MaxPool2d(3,2,padding=1)
    self.skip1 = conv_bn(64,128,1,stride=2)

    self.sepConv3 = separable_conv(128,256)
    self.sepConv4 = separable_conv(256,256)
    self.pool2 = nn.MaxPool2d(3,2,padding=1)
    self.skip2 = conv_bn(128,256,1,stride=2)

    self.sepConv5 = separable_conv(256,728)
    self.sepConv6 = separable_conv(728,728)
    self.pool3 = nn.MaxPool2d(3,2,padding=1)
    self.skip3 = conv_bn(256,728,1,stride=2)

  def forward(self,x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    
    skip = self.skip1(x)
    x = self.sepConv1(x)
    x = self.relu(x)
    x = self.sepConv2(x)
    x = self.pool1(x)
    x = x+skip

    skip = self.skip2(skip)
    x = self.sepConv3(x)
    x = self.relu(x)
    x = self.sepConv4(x)
    x = self.pool2(x)
    x = x+skip

    skip = self.skip3(skip)
    x = self.sepConv5(x)
    x = self.relu(x)
    x = self.sepConv6(x)
    x = self.pool3(x)
    x = x+skip
    return x

class middle_block(nn.Module):
  def __init__(self):
    super(middle_block,self).__init__()
    self.relu = nn.ReLU()
    self.sepConv1 = separable_conv(728,728)
    self.sepConv2 = separable_conv(728,728)
    self.sepConv3 = separable_conv(728,728)

  def forward(self,x):
    skip = x
    x = self.relu(x)
    x = self.sepConv1(x)
    x = self.relu(x)
    x = self.sepConv2(x)    
    x = self.relu(x)
    x = self.sepConv3(x)
    return x+skip

class exit_flow(nn.Module):
  def __init__(self):
    super(exit_flow,self).__init__()
    self.relu = nn.ReLU()
    self.sepConv1 = separable_conv(728,728)
    self.sepConv2 = separable_conv(728,1024)
    self.pool = nn.MaxPool2d(3,2,padding=1)
    self.skip = conv_bn(728,1024,1,stride=2)
    
    self.sepConv3 = separable_conv(1024,1536)
    self.sepConv4 = separable_conv(1536,2048)
    self.globPool = nn.MaxPool2d(7)
    self.flat = nn.Flatten()
    self.classify = nn.Linear(2048,1)
    
  def forward(self,x):
    skip = self.skip(x)
    x = self.relu(x)
    x = self.sepConv1(x)
    x = self.relu(x)
    x = self.sepConv2(x)
    x = self.pool(x)
    x = x+skip

    x = self.sepConv3(x)
    x = self.relu(x)
    x = self.sepConv4(x)
    x = self.relu(x)
    x = self.globPool(x)
    x = self.flat(x)
    x = self.classify(x)
    return x

class Xception(nn.Module):
  def __init__(self):
    super(Xception,self).__init__()
    self.entry = entry_flow()

    self.middle1 = middle_block()
    self.middle2 = middle_block()
    self.middle3 = middle_block()
    self.middle4 = middle_block()
    self.middle5 = middle_block()
    self.middle6 = middle_block()
    self.middle7 = middle_block()
    self.middle8 = middle_block()

    self.out = exit_flow()

  def forward(self, x):
    x = self.entry(x)
    x = self.middle1(x)
    x = self.middle2(x)
    x = self.middle3(x)
    x = self.middle4(x)
    x = self.middle5(x)
    x = self.middle6(x)
    x = self.middle7(x)
    x = self.middle8(x)
    x = self.out(x)
    return x



class SimpleConvModel(nn.Module):
  def __init__(self):
    super(SimpleConvModel, self).__init__()
    self.features = nn.Sequential(OrderedDict([
            ('conv0', conv_bn(3,8)),
            ('relu0', nn.ReLU()),
            ('conv1', conv_bn(8,16)),
            ('relu1', nn.ReLU()),
            ('pool0', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv2', conv_bn(16,32)),
            ('relu2', nn.ReLU()),
            ('pool1', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv3', conv_bn(32,64)),
            ('relu3', nn.ReLU()),
            ]))
    self.dropout = nn.Dropout(0.5)
    self.classify = nn.Linear(64,1)
  def forward(self,x):
    x =  self.features(x)
    x = F.relu(x, inplace=True)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.classify(x)
    return x



#EfficientNet
base_model  = [
  # expand_ratio, channels, repeats, stride, kernel_size
  [1,16,1,1,3],
  [6, 24, 2, 2, 3],
  [6, 40, 2, 2, 5],
  [6, 80, 3, 2, 3],
  [6, 112, 3, 1, 5],
  [6, 192, 4, 2, 5],
  [6, 320, 1, 1, 3],
]
phi_values = {
  # phi, resolution, drop_rate
  "b0":(0,224,0.2),
  "b1":(0.5,240,0.2),
  "b2":(1,260,0.3),
  "b3":(2,300,0.3),
}

class CNNBLock(nn.Module):
  def __init__(self, in_channels,out_channels, ks, stride, padding, groups=1):
    super(CNNBLock,self).__init__()
    self.cnn = nn.Conv2d(
      in_channels,
      out_channels,
      ks,
      stride,
      padding,
      groups=groups,
      bias=False
    )
    self.bn = nn.BatchNorm2d(out_channels)
    self.silu = nn.SiLU()

  def forward(self, x):
    return self.silu(self.bn(self.cnn(x)))

  

class SqueezeExcitation(nn.Module):
  def __init__(self,in_channels, reduced_dim):
      super(SqueezeExcitation,self).__init__()
      self.se = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),#C x H x W = C x 1 x 1
        nn.Conv2d(in_channels,reduced_dim,1),
        nn.SiLU(),
        nn.Conv2d(reduced_dim,in_channels,1),
        nn.Sigmoid(),
      )

  def forward(self,x ):
    return x * self.se(x)

class InvertedResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, ks, stride,  padding, expand_ratio, reduction=4,survival_prob=0.8):
      super(InvertedResidualBlock,self).__init__()
      self.survival_prob = survival_prob
      self.use_residual = in_channels == out_channels and stride == 1
      hidden_dim = in_channels*expand_ratio
      self.expand = in_channels != hidden_dim
      reduced_dim = int(in_channels/reduction)
      if self.expand:
        self.expand_conv = CNNBLock(
          in_channels,hidden_dim,ks=3, stride =1, padding = 1
        )
      self.conv = nn.Sequential(
        CNNBLock(
          hidden_dim,hidden_dim,ks, stride,padding, groups=hidden_dim
        ),
        SqueezeExcitation(hidden_dim,reduced_dim),
        nn.Conv2d(hidden_dim,out_channels,1,bias=False),
        nn.BatchNorm2d(out_channels),

      ) 

  def stochastic_depth(self,x):
    if not self.training:
      return x
    binary_tensor = torch.rand(x.shape[0],1,1,1,device=x.device )< self.survival_prob
    return torch.div(x,self.survival_prob) * binary_tensor 
  
  def forward(self,inputs ):
    x = self.expand_conv(inputs) if self.expand else inputs
    if self.use_residual:
      return self.stochastic_depth(self.conv(x))+ inputs
    else:
      return self.conv(x)


class EfficientNet(nn.Module):
  def __init__(self, version, num_classes=1):
    super(EfficientNet,self).__init__()
    width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
    last_channels = ceil(1280*width_factor)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.features = self.create_features(width_factor,depth_factor,last_channels)
    self.classifier = nn.Sequential(
      nn.Dropout(dropout_rate),
      nn.Linear(last_channels,num_classes)
    )

  def create_features(self,width_factor, depth_factor, last_channels):
    channels = int(32*width_factor)
    features = [CNNBLock(3,channels, 3, stride=2,padding=1)]
    in_channels = channels
    for expand_ratio,channels, repeats, stride,ks in base_model:
      out_channels = 4*ceil(int(channels*width_factor)/4)
      layers_repeats = ceil(repeats*depth_factor)
      for layer in range(layers_repeats):
        features.append(
          InvertedResidualBlock(
            in_channels,
            out_channels,
            expand_ratio=expand_ratio,
            stride = stride if layer == 0 else 1,
            ks = ks,
            padding=ks//2
          )
        )
        in_channels = out_channels
    features.append(
      CNNBLock(in_channels,last_channels,ks=1,stride = 1, padding =0)

    )
    return nn.Sequential(*features)


  def calculate_factors(self,version, alpha =1.2, beta=1.1):
    phi, res, drop_rate = phi_values[version]
    depth_factor = alpha **phi 
    width_factor = beta**phi 
    return width_factor, depth_factor, drop_rate
  
  def forward(self,x):
    x = self.pool(self.features(x))
    return self.classifier(x.view(x.shape[0],-1))

# https://github.com/d-li14/efficientnetv2.pytorch


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
