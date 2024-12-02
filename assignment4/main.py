from torch.utils.data import DataLoader
from torch import nn
from torch import Tensor
import torch
from torchvision.datasets import MNIST
from tqdm import tqdm
import pandas as pd

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.layer_2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.layer_4 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor):
        x = self.layer_1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)

        x = self.layer_2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)

        # x = self.dropout(x)

        x = self.layer_4(x)
        return x


model = MyModel(input_size=784, hidden_size1=256, hidden_size2= 126, output_size=10)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = get_device()
print(device)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)

criterion = nn.CrossEntropyLoss()

def train(model, train_dataloader, criterion, optimizer, device):
    model.train()

    mean_loss = 0.0

    for data, labels in train_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        loss.backward()

        if False:
            last_layer_w_grad = model.layer_2.weight.grad
            last_layer_b_grad = model.layer_2.bias.grad
            print(f"Last layer gradient: {last_layer_w_grad.shape}")
            print(f"Last layer gradient: {last_layer_b_grad.shape}")

        optimizer.step()
        optimizer.zero_grad()

        mean_loss += loss.item()

    mean_loss /= len(train_dataloader)
    return mean_loss


def val(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(dataloader)
    accuracy = correct / total
    return val_loss, accuracy


def main(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs):
    with tqdm(tuple(range(epochs))) as tbar:
        for epoch in tbar:
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_accuracy = val(model, val_dataloader, criterion, device)
            scheduler.step(val_loss)
            tbar.set_description(f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Val test accuracy: {val_accuracy*100:.3f}")

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transforms)
val_dataset = MNIST(root='./data', train=False, download=True, transform=val_transforms)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=10,
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=500,
    shuffle=False,
    num_workers=10,
    pin_memory=True
)


main(model, train_dataloader, val_dataloader, criterion, optimizer, device, 200)

results = {
    "ID": [],
    "target": []
}

model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(val_dataloader):
        images = images.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        batch_size = labels.size(0)
        results["ID"].extend(range(i * batch_size, i * batch_size + batch_size))
        results["target"].extend(predictions.cpu().tolist())

df = pd.DataFrame(results)
df.to_csv("submission.csv", index=False)