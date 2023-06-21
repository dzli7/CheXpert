import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from model14_data import *
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

models = []
optimizers = []
for label in label_columns:
    model = densenet(1)
    models.append(model)
    optimizers.append(optim.Adam(model.parameters()))

criterion = nn.MSELoss()

def train(model, trainloader, optimizer, n_epochs=10, eps=0.00001):
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
        print('\n\tlosses: ' + str(training_loss_history[epoch,0]))
        if epoch > 0 and training_loss_history[epoch-1,0] - training_loss_history[epoch,0] <= eps:
            break
    return training_loss_history

def predict(testx):
    preds = np.zeros((testx.shape[0],14))
    with torch.no_grad():
        for model in models:
            model.eval()
        for i in range(testx.shape[0]):
            orig_image = testx[i][0].cuda()
            dia_image = testx[i][1].cuda()
            edge_image = testx[i][2].cuda()
            # forward pass
            for j,model in enumerate(models):
                output = None
                method = method_dict[label_columns[j]]
                if method == 'o':
                    output = model(orig_image).data.cpu().numpy()
                elif method == 'd':
                    output = model(dia_image).data.cpu().numpy()
                elif method == 'e':
                    output = model(edge_image).data.cpu().numpy()
                preds[i,j] = output[0][0]
    return preds

def main():
    start = time.time()
    trainloaders = []
    for label in label_columns:
        if label in ['Lung Opacity','Pneumothorax',]:
            trainloader = get_train_data(label, method_dict[label])
            trainloaders.append(trainloader)
        else:
            trainloader = torch.load(f'/central/groups/CS156b/2022/team_dirs/candice/label_trainloaders/{label}.pt')
            trainloaders.append(trainloader)
    testx = torch.load(f'/central/groups/CS156b/2022/team_dirs/candice/testx.pt')
    
    n_epochs = 100
    for i in range(14):
        train(models[i], trainloaders[i], optimizers[i], n_epochs=n_epochs)
    preds = predict(testx)
    
    testdf = pd.read_csv(test_path)
    output = pd.DataFrame(preds, columns=label_columns)
    output.insert(0, 'Id', testdf['Id'])
    output.to_csv('/home/ding/preds14.csv')

if __name__ == '__main__':
    main()
