import torch
import os
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

os.system("gdown --id 1qO2OLR7skDibo1LaMKD3CiOl_jaCTZ0h")

class JHARMNet(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.premodel = pretrained_model
        self.premodel.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.premodel.fc.weight)

    def forward(self, x):
        out = self.premodel(x)
        return out

class HiddenLayer(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.premodel = pretrained_model
        self.new_layer = nn.Sequential(
                nn.Linear(1000, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 10)
                )

    def forward(self, x):
        out = self.premodel(x)
        out_new_layer = self.new_layer(out)
        return out_new_layer


resnet = models.resnet50(pretrained=True)

x = torch.randn((2, 3, 32, 32))

print(x.shape)
model_check = HiddenLayer(resnet)
model_check.load_state_dict(torch.load("CIFAR_end_hll.pt"))
model_check.eval()
print(model_check.forward(x).shape)
