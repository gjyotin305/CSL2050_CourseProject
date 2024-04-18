import torchvision.models as models
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as v2
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


mps_device = torch.device("cuda")

IMAGE_SIZE = 224
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

transform_test = v2.Compose(
    [v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
     v2.ToTensor(),
     v2.Normalize(mean, std)])

transform_train = v2.Compose(
    [v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.RandomRotation(20),
        v2.RandomHorizontalFlip(0.1),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        v2.ToTensor(),
        v2.Normalize(mean, std),
        v2.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)])

cifar_trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train)
cifar_testset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test)

batch_size = 64

trainLoader = DataLoader(
    cifar_trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)
testLoader = DataLoader(
    cifar_testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(next(iter(trainLoader))[0].shape)


resnet = models.resnet50(pretrained=True)
# print(resnet)
# print(resnet.fc)


class JHARMNet(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.premodel = pretrained_model
        self.premodel.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.premodel.fc.weight)

    def forward(self, x):
        out = self.premodel(x)
        return out


model_check = JHARMNet(resnet, 10)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1

optimizer = torch.optim.SGD(
    model_check.parameters(),
    lr=learning_rate,
    momentum=0.2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


def train_model(
        model,
        train_loader,
        validation_loader,
        optimizer,
        n_epochs=20):

    # Global variable
    N_test = len(cifar_testset)
    accuracy_list = []
    train_loss_list = []
    model = model.to(mps_device)
    train_cost_list = []
    val_cost_list = []

    for epoch in range(n_epochs):
        train_COST = 0
        print(f"Training Epoch: {epoch}")
        for x, y in tqdm(train_loader):
            x = x.to(mps_device)
            y = y.to(mps_device)
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            train_COST += loss.item()

        train_COST = train_COST / len(train_loader)
        train_cost_list.append(train_COST)
        correct = 0
        print(f"Validation Loop")
        # Perform the prediction on the validation data
        val_COST = 0
        for x_test, y_test in tqdm(validation_loader):
            model.eval()
            x_test = x_test.to(mps_device)
            y_test = y_test.to(mps_device)
            z = model(x_test)
            val_loss = criterion(z, y_test)
            scheduler.step(val_loss)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            val_COST += val_loss.item()

        val_COST = val_COST / len(validation_loader)
        val_cost_list.append(val_COST)
        accuracy = correct / N_test
        accuracy_list.append(accuracy)

        print("--> Epoch Number : {}".format(epoch + 1),
              " | Training Loss : {}".format(round(train_COST, 4)),
              " | Validation Loss : {}".format(round(val_COST, 4)),
              " | Validation Accuracy : {}%".format(round(accuracy * 100, 2)))

    return accuracy_list, train_cost_list, val_cost_list, model


accuracy_list_normalv5, train_cost_listv5, val_cost_listv5, model_to_save = train_model(
    model=model_check, n_epochs=100, train_loader=trainLoader, validation_loader=testLoader, optimizer=optimizer)

torch.save(model_to_save.state_dict(), "CIFAR.pt")
model = torch.load("CIFAR.pt")
model.eval()

pred = model(next(iter(testLoader))[0])
print(pred[0][0], next(iter(testLoader))[1][0])
