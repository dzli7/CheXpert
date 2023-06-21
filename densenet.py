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
import torchvision.models as models
import torch.hub
from collections import OrderedDict
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

model = models.densenet121(memory_efficient=True, pretrained=True)
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.classifier = nn.Sequential(
            nn.Linear(1024, 14),
            nn.Tanh()
        )
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

def train(trainloader, n_epochs=10):
    model.train()
    model.cuda()
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
        model.cuda()
        for i, data in enumerate(testloader):
            image = data.cuda()
            # forward pass
            output = model(image)
            preds[i,:] = output.data.cpu().numpy()
    return preds

def main():
    #X, Y = get_train_data()
    #train_loader = trainload(X, Y)
    train_loader = get_train_data()
    #print('data loaded')
    testloader = get_test_data()
    
    n_epochs = 100
    losses = train(train_loader, n_epochs=n_epochs)
    preds = predict(model, testloader)
    
    testdf = pd.read_csv(test_path)
    output = pd.DataFrame(preds, columns=label_columns)
    output.insert(0, 'Id', testdf['Id'])
    output.to_csv('/home/ding/densepreds_dia_edge.csv')
    
    plt.figure()
    plt.plot(range(1,n_epochs+1),losses)
    plt.savefig('/home/ding/testdense_dia_edge.png')

if __name__ == '__main__':
    main()
