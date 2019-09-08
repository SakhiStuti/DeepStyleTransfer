import torch.nn as nn
import torch


class decoder(nn.Module):
    def __init__(self, path = None):
        super(decoder, self).__init__()
        self.network = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(512, 256, (3, 3)),
                      nn.ReLU(),
                      nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(256, 256, (3, 3)),
                      nn.ReLU(),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(256, 256, (3, 3)),
                      nn.ReLU(),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(256, 256, (3, 3)),
                      nn.ReLU(),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(256, 128, (3, 3)),
                      nn.ReLU(),
                      nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(128, 128, (3, 3)),
                      nn.ReLU(),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(128, 64, (3, 3)),
                      nn.ReLU(),
                      nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(64, 64, (3, 3)),
                      nn.ReLU(),
                      nn.ReflectionPad2d((1, 1, 1, 1)),
                      nn.Conv2d(64, 3, (3, 3))
                      )
        if path:
            self.network.load_state_dict(torch.load(path))
    def forward(self, features):
         return self.network(features)