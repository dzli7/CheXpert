import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from validation_data import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as M
import torch.hub
from collections import OrderedDict
import time
import math
from unet_model import UNet

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def densenet(out_classes):
    model = M.densenet121(memory_efficient=True, pretrained=True)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Sequential(
            nn.Linear(1024, out_classes),
            nn.Tanh()
        )
    return model

# individual dense models
models = []
optimizers = []
for label in label_columns:
    model = densenet(1)
    models.append(model)
    optimizers.append(optim.Adam(model.parameters(),lr=0.1))

criterion = nn.MSELoss()

# one model dense
dense = densenet(14)
dense_optim = optim.Adam(model.parameters(),lr=0.1)

# one model unet
unet = UNet(14)
unet_optim = optim.Adam(unet.parameters(),lr=0.1)


def train(model, trainloader, validloader, optimizer, n_epochs=100):
    model.cuda()
    training_loss_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])
    for epoch in range(n_epochs):
        # train
        model.train()
        for i, data in enumerate(trainloader):
            image, labels = data
            image, labels = image.cuda(), labels.cuda()
            optimizer.zero_grad()
            # forward pass
            output = model(image)
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()

            training_loss_history[epoch] += loss.item()
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(validloader):
                image, labels = data
                image, labels = image.cuda(), labels.cuda()
                output = model(image)
                loss = criterion(output, labels)
                
                valid_loss_history[epoch] += loss.item()

        training_loss_history[epoch] /= len(trainloader)
        valid_loss_history[epoch] /= len(validloader)

    return training_loss_history, valid_loss_history

def plot(train, valid, label=None, method=None):
    plt.figure()
    plt.plot(range(1,train.shape[0]+1),train,label='Training Loss')
    plt.plot(range(1,train.shape[0]+1),valid,label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='upper right')
    if label is not None:
        plt.title(f'{label} Training and Validation MSE Curves')
        plt.savefig(f'/home/ding/cs156b/validation/{label}.png')
    elif method is not None:
        plt.title(f'{method} Training and Validation MSE Curves')
        plt.savefig(f'/home/ding/cs156b/validation/{method}.png')

def main():
    start = time.time()
    trainloaders = []
    validloaders = []
    for label in label_columns:
        trainloader, validloader = get_train_data14(label, method_dict[label])
        trainloaders.append(trainloader)
        validloaders.append(validloader)
        
    trainloader, validloader = get_train_data()
    
    n_epochs = 50
    for i in range(14):
        train_loss, valid_loss = train(models[i], trainloaders[i], validloaders[i], optimizers[i], n_epochs=n_epochs)
        plot(train_loss, valid_loss, label=label_columns[i])
    
    dense_train, dense_valid = train(dense, trainloader, validloader, dense_optim, n_epochs=n_epochs)
    unet_train, unet_valid = train(unet, trainloader, validloader, unet_optim, n_epochs=n_epochs)
    plot(dense_train, dense_valid, method='DenseNet121')
    plot(unet_train, unet_valid, method='UNet')

if __name__ == '__main__':
    main()
