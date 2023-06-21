import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

#['Unnamed: 0', 'Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA',
# 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
#'Fracture', 'Support Devices']
label_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly','Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia','Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other','Fracture', 'Support Devices']

# create 14 separate dfs, one for each disease

data_path = '/central/groups/CS156b/data/'
train_path = data_path + 'student_labels/train.csv'
test_path = data_path + 'student_labels/test_ids.csv'
imagesize = (128, 128)
#degree = 90
cuda0 = torch.device('cuda:0')

#rotate = T.RandomRotation(degree)
#policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
#augmenters = [T.AutoAugment(policy) for policy in policies]
#augmenter = T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)
blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(1,5))
equalize = T.RandomEqualize(p=1.0)
resize = T.Resize(size=imagesize)
totensor = T.ToTensor()
process2 = T.Compose([equalize, blur, resize, totensor])
process = T.Compose([resize, totensor])

#def augment(img):
    #rotated1 = rotate(img)
    #rotated2 = rotate(img)
    #randaug = augmenter(img)
    #blurred1 = blur(img)
    #blurred2 = blur(img)

    #return rotated1, blurred1, blurred2

#def process2(img):
    #return totensor(resize(F.gaussian_blur(F.equalize(img),5)))

def get_train_data():
    df = pd.read_csv(train_path).fillna(0)
    X = []
    Y = []
    
    i = 0
    df_dict = df.to_dict('records')
    for row in df_dict:
        if i >= int(df.shape[0]) - 1:
            break
        image_path = data_path + row['Path']

        image = Image.open(image_path)
        #imcopy = image.copy()
        #imaget = totensor(image)
        
        X.append(process(image).numpy())
        X.append(process2(image).numpy())
        image.close()
        
        labels = []
        for label in label_columns:
            labels.append(row[label])

        Y.append(labels)
        Y.append(labels)

        i += 1

    return trainload(X,Y)

def get_test_data():
    df = pd.read_csv(test_path)
    X = []
    df_dict = df.to_dict('records')
    for row in df_dict:
        image_path = data_path + row['Path']

        image = Image.open(image_path)
        #imcopy = image.copy()
        #.resize(imagesize)
        #image_gray = ImageOps.grayscale(image)
        X.append(process(image).numpy())
        image.close()

    #trans = T.ToTensor()
    #trans = T.Resize(size=imagesize)
    #for i in range(len(X)):
       #im = trans(X[i]).numpy()
       #X[i] = im
    
    testx = torch.from_numpy(np.array(X).astype(np.float32))
    
    testloader = DataLoader(testx, batch_size=1, shuffle=False)
    return testloader

def trainload(X, Y):
    #trans = T.ToTensor()
    #trans = T.Resize(size=imagesize)
    #for i in range(len(X)):
       #threechannel = np.zeros((3,64,64))
       #im = trans(X[i]).numpy()
       #for j in range(3):
           #threechannel[j] = im[0]
       #X[i] = im

    trainx = torch.from_numpy(np.array(X).astype(np.float32))
    trainy = torch.from_numpy(np.array(Y).astype(np.float32))

    traindata = data_utils.TensorDataset(trainx, trainy)
    train_loader = DataLoader(traindata, batch_size=512, shuffle=True, pin_memory=True)
    return train_loader
