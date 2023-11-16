import torch
import torch.nn as nn
import torchvision

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        vggnet = torchvision.models.vgg16(
                pretrained=True) 

        self.backbone = torch.nn.Sequential(*list(vggnet.children())[:-1])
        #self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(25088, 4096), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(4096, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(25088, 4096), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Linear(4096, 4) )

    def forward(self, x):
        
        x = self.backbone(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot