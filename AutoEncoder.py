import torch
from torch import nn

class FCEDN38(nn.Module):
    def __init__(self):
        super(FCEDN38, self).__init__()
        # Encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )

        # Decoder
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer11 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer14 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer15 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer16 = nn.Sequential(
            nn.ConvTranspose2d(30, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        h5 = self.layer5(h4)
        h6 = self.layer6(h5)
        h7 = self.layer7(h6)
        h8 = self.layer8(h7)
        h9 = self.layer9(h8)
        h10 = self.layer10(h9)
        h11 = self.layer11(h10)
        h12 = self.layer12(h11)
        h13 = self.layer13(h12)
        h14 = self.layer14(h13)
        h15 = self.layer15(h14)
        output = self.layer16(h15)
        return output