import torch.nn as nn
from feat.networks import *


class ProtoNet(nn.Module):

    def __init__(self):
        super().__init__()
        if args.model_type == 'ConvNet':
            self.encoder = ConvNet()
        elif args.model_type == 'ResNet':
            self.encoder = ResNet()
        else:
        	raise ValueError('')

    def forward(self, x):
        x = self.encoder(x)
        return x