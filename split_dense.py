import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from data import *
from split_data import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.hub
from collections import OrderedDict
import time
import math


heart_model = models.densenet121(memory_efficient=True, pretrained=True)
heart_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
heart_model.classifier = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Tanh()
        )
heart_criterion = nn.MSELoss()
heart_optimizer = optim.Adam(heart_model.parameters())

lung_model = models.densenet121(memory_efficient=True, pretrained=True)
lung_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
lung_model.classifier = nn.Sequential(
            nn.Linear(1024, 8),
            nn.Tanh()
        )
lung_criterion = nn.MSELoss()
lung_optimizer = optim.Adam(lung_model.parameters())

other_model = models.densenet121(memory_efficient=True, pretrained=True)
other_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
other_model.classifier = nn.Sequential(
            nn.Linear(1024, 4),
            nn.Tanh()
        )
other_criterion = nn.MSELoss()
other_optimizer = optim.Adam(other_model.parameters())


def train(trainloader, n_epochs):
    heart_model.train()
    heart_training_loss_history = np.zeros([n_epochs, 1])
    
    lung_model.train()
    lung_training_loss_history = np.zeros([n_epochs, 1])
    
    other_model.train()
    other_training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}: ', end='')
        # train
        for i, data in enumerate(trainloader):
            image, labels = data
            heart_labels = labels[:,[1,2]]
            lung_labels = labels[:,[3,4,6,7,8,9,10,11]]
            other_labels = labels[:,[0,5,12,13]]

            heart_optimizer.zero_grad()
            lung_optimizer.zero_grad()
            other_optimizer.zero_grad()
            
            # forward pass
            heart_output = heart_model(image)
            lung_output = lung_model(image)
            other_output = other_model(image)
            
            # calculate categorical cross entropy loss
            heart_loss = heart_criterion(heart_labels, heart_output)
            lung_loss = lung_criterion(lung_output, lung_labels)
            other_loss = other_criterion(other_output, other_labels)
            # backward pass
            heart_loss.backward()
            heart_optimizer.step()
            
            lung_loss.backward()
            lung_optimizer.step()
            
            other_loss.backward()
            other_optimizer.step()
            
            # track training loss
            heart_training_loss_history[epoch] += heart_loss.item()
            lung_training_loss_history[epoch] += lung_loss.item()
            other_training_loss_history[epoch] += other_loss.item()

        heart_training_loss_history[epoch] /= len(trainloader)
        lung_training_loss_history[epoch] /= len(trainloader)
        other_training_loss_history[epoch] /= len(trainloader)
        print(str((heart_loss.item() + lung_loss.item() + other_loss.item())/3))

    return (heart_training_loss_history, lung_training_loss_history, other_training_loss_history)


def predict(model, testloader, label_len):
    preds = np.zeros((len(testloader),label_len))
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader):
            # forward pass
            output = model(data)
            preds[i,:] = output.data.cpu().numpy()
    return preds

def main():
    train_loader = dia_train_data()
    testloader = get_test_data()
    
    n_epochs = 10
    (heart_training_loss_history, lung_training_loss_history, other_training_loss_history) = train(train_loader, n_epochs)
    heart_preds = predict(heart_model, testloader, 2)
    lung_preds = predict(lung_model, testloader, 8)
    other_preds = predict(other_model, testloader, 4)
    
    preds = np.zeros((len(testloader),14))
    preds[:,0] = other_preds[:,0]
    preds[:,1] = heart_preds[:,0]
    preds[:,2] = heart_preds[:,1]
    preds[:,3] = lung_preds[:,0]
    preds[:,4] = lung_preds[:,1]
    preds[:,5] = other_preds[:,1]
    preds[:,6] = lung_preds[:,2]
    preds[:,7] = lung_preds[:,3]
    preds[:,8] = lung_preds[:,4]
    preds[:,9] = lung_preds[:,5]
    preds[:,10] = lung_preds[:,6]
    preds[:,11] = lung_preds[:,7]
    preds[:,12] = other_preds[:,2]
    preds[:,13] = other_preds[:,3]
    
    testdf = pd.read_csv(test_path)
    output = pd.DataFrame(np.array(preds), columns=label_columns)
    output.insert(0, 'Id', testdf['Id'])
    output.to_csv('/home/dzli/densepreds_multiple2.csv')
    
    plt.figure()
    plt.plot(range(1,n_epochs+1),heart_training_loss_history)
    plt.plot(range(1,n_epochs+1),lung_training_loss_history)
    plt.plot(range(1,n_epochs+1),other_training_loss_history)
    plt.savefig('/home/dzli/testdense_multiple2.png')

if __name__ == '__main__':
    main()