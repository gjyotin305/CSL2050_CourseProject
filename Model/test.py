import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models


class JHARMNet(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.premodel = pretrained_model
        self.premodel.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.premodel.fc.weight)

    def forward(self, x):
        out = self.premodel(x)
        return out

resnet = models.resnet50(pretrained=True)

x = torch.randn((2, 3, 32, 32))

print(x.shape)
model_check = JHARMNet(resnet, 10)
model_check.load_state_dict(torch.load("CIFAR.pt"))
model_check.eval()
print(model_check.forward(x))
