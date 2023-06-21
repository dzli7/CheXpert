import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from data import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.hub
from torchvision.models import resnet18
from collections import OrderedDict


model = models.resnet18(pretrained=True)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

def train(trainloader, n_epochs=10):
    model.cuda()
    model.train()
    training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}:', end='')
        # train
        for i, data in enumerate(trainloader):
            image, labels = data
            image, labels = image.cuda(), labels.cuda()
            optimizer.zero_grad()
            # forward pass
            output = model(image)
            # calculate categorical cross entropy loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            
            # track training loss
            training_loss_history[epoch] += loss.item()

        training_loss_history[epoch] /= len(trainloader)
        print('\n\tloss: ' + str(training_loss_history[epoch,0]))

    return training_loss_history

def predict(model, testloader):
    preds = np.zeros((len(testloader),14))
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader):
            image = data.cuda()
            # forward pass
            output = model(image)
            preds[i,:] = output.data.cpu().numpy()
    return preds

def main():
    X, Y = get_train_data()
    _, train_loader = trainload(X, Y)
    #testloader = get_test_data()
    
    n_epochs = 100
    losses = train(train_loader, n_epochs=n_epochs)
    #preds = predict(model, testloader)
    
    #testdf = pd.read_csv(test_path)
    #output = pd.DataFrame(preds, columns=label_columns)
    #output.insert(0, 'Id', testdf['Id'])
    #output.to_csv('/home/ding/densepreds.csv')
    
    plt.figure()
    plt.plot(range(1,n_epochs+1),losses)
    plt.savefig('/home/jp4rk/testdense.png')

if __name__ == '__main__':
    main()
