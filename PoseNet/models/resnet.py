import torch
import torch.nn as nn
import torchvision

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        resnet = torchvision.models.resnet152(
                pretrained=True) 

        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        #self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(2048, 1024), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(1024, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(2048, 1024), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(1024, 4) )

    def forward(self, x):
        
        x = self.backbone(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot