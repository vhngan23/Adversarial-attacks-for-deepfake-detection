from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict
import torch 


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

class Conv_bn(nn.Module):
  def __init__(self, in_channels, out_channels,ks = 3, **kwargs):
    super(Conv_bn,self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,ks,padding=int((ks-1)/2),**kwargs)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
  def forward(self,x):
    return self.relu(self.batch_norm(self.conv(x)))

class SimpleConvModel(nn.Module):
  def __init__(self):
    super(SimpleConvModel, self).__init__()
    self.features = nn.Sequential(OrderedDict([
            ('conv0', Conv_bn(3,8)),
            ('conv1', Conv_bn(8,16)),
            ('pool0', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv_bn(16,32)),
            ('pool1', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv3', Conv_bn(32,64)),
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
