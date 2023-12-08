import torch
import torch.nn as nn
import torchvision

class EachNet(torch.nn.Module): # tr, rot backbones are different
    def __init__(self):
        super(EachNet, self).__init__()

        mobilenet = torchvision.models.mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights.DEFAULT) 

        self.backbone = torch.nn.Sequential(*list(mobilenet.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(1280 * 7 * 7, 1024), 
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(1280 * 7 * 7, 1024), 
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 4) )

    def forward(self, x):
        
        tr = self.backbone(x)
        tr = self.avgpool(tr)
        tr = tr.view(tr.size(0), -1)
        tr = self.trans_head(tr)

        rot = self.backbone(x)
        rot = self.avgpool(rot)
        rot = rot.view(rot.size(0), -1)
        rot = self.rot_head(rot)

        return  tr, rot