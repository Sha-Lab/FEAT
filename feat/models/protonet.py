import torch.nn as nn
from feat.utils import euclidean_metric

class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')

    def forward(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature
        return logits