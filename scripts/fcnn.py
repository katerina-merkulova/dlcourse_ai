import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


BATCH_SIZE = 64


def load_data():
    data_train = dset.SVHN('./data/', split='train',
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.43,0.44,0.47],
                                                    std=[0.20,0.20,0.20])
                           ])
                           )
    data_test = dset.SVHN('./data/', split='test',
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.43,0.44,0.47],
                                                   std=[0.20,0.20,0.20])
                          ]))
    return data_train, data_test


def split_data(data_train):
    data_size = data_train.data.shape[0]
    validation_split = .2
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                             sampler=val_sampler)
    return train_loader, val_loader


class Flattener(nn.Module):
    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)


def build_model():
    nn_model = nn.Sequential(
        Flattener(),
        nn.Linear(3*32*32, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),

        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),

        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(128),

        nn.Linear(128, 10),
    )

    optimizer = optim.Adam(nn_model.parameters(), lr=1e-2)
    loss = nn.CrossEntropyLoss().type(torch.FloatTensor)

    return nn_model, optimizer, loss


def train_model(data_train, num_epochs):
    train_loader, val_loader = split_data(data_train)
    model, optimizer, loss = build_model()

    loss_history = []
    train_history = []
    val_history = []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    for epoch in range(num_epochs):
        model.train()    # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y)
            total_samples += y.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        scheduler.step()

        print(f'Epoch {epoch}, '
              f'Average loss: {ave_loss:.4f}, '
              f'Train accuracy: {train_accuracy:.4f}, '
              f'Val accuracy: {val_accuracy}:.4f')

    return model, loss_history, train_history, val_history


def compute_accuracy(model, loader):
    model.eval()    # Evaluation mode
    acc_history = []

    for i_step, (x, y) in enumerate(loader):
        pred_prob = model(x)
        pred_labels = pred_prob.argmax(dim=1)
        accuracy = sum([1 for pred, test in zip(pred_labels, y) if pred == test]) / BATCH_SIZE
        acc_history.append(accuracy)

    return np.mean(acc_history)


if __name__ == '__main__':
    data_train, data_test = load_data()
    nn_model, loss_history, train_history, val_history = train_model(data_train, num_epochs=8)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE)
    test_accuracy = compute_accuracy(nn_model, test_loader)
    print(f'Test accuracy: {test_accuracy:.4f}')