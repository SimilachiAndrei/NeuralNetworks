from torch.utils.data import DataLoader
from torch import nn
from torch import Tensor
import torch
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm
from torchvision import transforms

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size1)
        self.layer_2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer_3 = nn.Linear(hidden_size2, hidden_size3)
        self.layer_4 = nn.Linear(hidden_size3, output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor):
        x = self.layer_1(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)

        x = self.layer_2(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)

        x = self.layer_3(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)

        x = self.layer_4(x)

        return x


model = MyModel(input_size=784, hidden_size1=256, hidden_size2= 126, hidden_size3=64, output_size=10)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
        nn.init.zeros_(m.bias)  # Initialize biases to 0
model.apply(init_weights)



def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = get_device()
print(device)

model = model.to(device)

# Optimizers apply the gradients calculated by the Autograd engine to the weights, using their own optimization technique
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.001)  # SGD with Nesterov momentum and weight decay
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 'min' for metrics like validation loss, 'max' for accuracy
    factor=0.5,  # Factor by which the learning rate is reduced
    patience=5,  # Number of epochs to wait before reducing the learning rate
)

criterion = nn.CrossEntropyLoss()


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()  # We need to activate the dropout & batch norm layers

    mean_loss = 0.0

    for data, labels in train_dataloader:
        data = data.to(device)  # We move the data to device. Bonus: we can do this in an async manner using non_blocking and pin_memory
        labels = labels.to(device)

        outputs = model(data)  # the forward pass
        loss = criterion(outputs, labels)  # we calculate the loss

        loss.backward()  # we backpropagate the loss

        if False:
            # After loss.backward(), the gradients for each weight and bias are calculated and assigned to layer.weight.grad and layer.bias.grad
            last_layer_w_grad = model.layer_2.weight.grad
            last_layer_b_grad = model.layer_2.bias.grad
            print(f"Last layer gradient: {last_layer_w_grad.shape}")
            print(f"Last layer gradient: {last_layer_b_grad.shape}")

        optimizer.step()  # we update the weights
        optimizer.zero_grad()  # we reset the gradients

        mean_loss += loss.item()

    mean_loss /= len(train_dataloader)
    return mean_loss


def val(model, dataloader, criterion, device):
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total number of samples

    val_loss /= len(dataloader)  # Average loss
    accuracy = correct / total  # Accuracy as a percentage
    return val_loss, accuracy


def main(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs):
    with tqdm(tuple(range(epochs))) as tbar:
        for epoch in tbar:
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_accuracy = val(model, val_dataloader, criterion, device)
            scheduler.step(val_loss)
            tbar.set_description(f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Val test accuracy: {val_accuracy*100:.3f}")

def transforms():
    return lambda x: torch.from_numpy(np.array(x, dtype=np.float32).flatten() / 255)


train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms())
val_dataset = MNIST(root='./data', train=False, download=True, transform=transforms())
train_dataloader = DataLoader(
    train_dataset,
    batch_size=128, #64 seems to work better
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


main(model, train_dataloader, val_dataloader, criterion, optimizer, device, 50)
