import torch
import torch.nn as nn
import torchvision


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        #############################################
        #############################################
        self.feat11 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),)
        self.feat12 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat21 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feat22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat31 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feat32 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat41 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feat42 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #############################################
        #############################################
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(128 * 7 * 7, 1024), 
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(128 * 7 * 7, 1024), 
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 4) )

    def forward(self, x):
        
        x = self.feat11(x)
        x = self.feat12(x)
        x = self.pool1(x)

        x = self.feat21(x)
        x = self.feat22(x)
        x = self.pool2(x)

        x = self.feat31(x)
        x = self.feat32(x)
        x = self.pool3(x)

        x = self.feat41(x)
        x = self.feat42(x)
        x = self.pool4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot