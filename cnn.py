import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from data import *

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.1),
    nn.Flatten(),
    nn.Linear(3600, 100),
    nn.Linear(100, 14),
    nn.Tanh(),
)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

def train(trainloader, n_epochs=10):
    model.to(cuda0)
    training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}:', end='')
        train_total = 0
        train_correct = 0
        # train
        model.train()
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
    model.to(cuda0)
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader):
            if i == 0:
                print(data)
            image = data.cuda()
            # forward pass
            output = model(image)
            if i == 1:
                print(output.data.size())
            preds[i,:] = output.data.cpu().numpy()
    return preds

def main():
    X, Y = get_train_data()
    _, train_loader = trainload(X, Y)
    #testloader = get_test_data()
    
    losses = train(train_loader)
    #preds = predict(model, testloader)
    #output = pd.DataFrame(preds, columns=label_columns)
    #output.insert(0, 'Id', testdf['Id'])
    #output.to_csv('/home/ding/testpreds.csv')
    plt.figure()
    plt.plot(range(1,11),losses)
    plt.savefig('/home/ding/testtrain.png')

if __name__ == '__main__':
    main()
