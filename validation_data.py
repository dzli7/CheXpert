import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import cv2
from diaphragm import remove_diaphragm
import time
import math

label_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly','Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia','Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other','Fracture', 'Support Devices']

# create 14 separate dfs, one for each disease
method_dict = {'No Finding':'o', 'Enlarged Cardiomediastinum':'d', 'Cardiomegaly':'d','Lung Opacity':'d', 
    'Lung Lesion':'e', 'Edema':'d', 'Consolidation':'d', 'Pneumonia':'o','Atelectasis':'e', 'Pneumothorax':'o', 'Pleural Effusion':'d', 
    'Pleural Other':'d','Fracture':'o', 'Support Devices':'o'}

data_path = '/central/groups/CS156b/data/'
train_path = data_path + 'student_labels/train.csv'
test_path = data_path + 'student_labels/test_ids.csv'
imagesize = (128, 128)

blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(1,5))
equalize = T.RandomEqualize(p=1.0)
resize = T.Resize(size=imagesize)
totensor = T.ToTensor()
orig_process = T.Compose([equalize, blur, resize, totensor])
process = T.Compose([resize, totensor])

def get_train_data():
    df = pd.read_csv(train_path).fillna(0)
    Xtrain = []
    Ytrain = []
    Xvalid = []
    Yvalid = []
    
    i = 0
    df_dict = df.to_dict('records')
    for row in df_dict:
        if i >= int(df.shape[0]*0.1) - 1:
            break
        image_path = data_path + row['Path']
        
        
        image = Image.open(image_path)
        processed = orig_process(image).numpy()
        image.close()
        
        labels = []
        for label in label_columns:
            labels.append(row[label])

        if i <= int(df.shape[0]*0.08):
            Xtrain.append(processed)
            Ytrain.append(labels)
        else:
            Xvalid.append(processed)
            Yvalid.append(labels)
        i += 1

    return trainload(Xtrain,Ytrain), validload(Xvalid,Yvalid)
    

def get_train_data14(label, method):
    df = pd.read_csv(train_path)
    df = df.dropna(how='any',subset=[label])
    Xtrain = []
    Ytrain = []
    Xvalid = []
    Yvalid = []
    
    i = 0
    df_dict = df.to_dict('records')
    for row in df_dict:
        if i >= int(df.shape[0]*0.1) - 1:
            break
        image_path = data_path + row['Path']

        if method == 'd':
            image_nodi = remove_diaphragm(image_path)
            if i <= int(df.shape[0]*0.08):
                Xtrain.append(process(image_nodi).numpy())
            else:
                Xvalid.append(process(image_nodi).numpy())
        elif method == 'e':
            image = Image.open(image_path)
            imcopy = image.copy()
            edges = cv2.Canny(image=np.array(imcopy), threshold1=75, threshold2=120)
            if i <= int(df.shape[0]*0.08):
                Xtrain.append(process(Image.fromarray(edges)).numpy())
            else:
                Xvalid.append(process(Image.fromarray(edges)).numpy())
            image.close()
        elif method == 'o':
            image = Image.open(image_path)
            imcopy = image.copy()
            if i <= int(df.shape[0]*0.08):
                Xtrain.append(orig_process(imcopy).numpy())
            else:
                Xvalid.append(orig_process(imcopy).numpy())
            image.close()
            
        if i <= int(df.shape[0]*0.08):
            Ytrain.append([row[label]])
        else:
            Yvalid.append([row[label]])
        i += 1
    return trainload(Xtrain, Ytrain), validload(Xvalid, Yvalid)

def trainload(X, Y):
    trainx = torch.from_numpy(np.array(X).astype(np.float32))
    trainy = torch.from_numpy(np.array(Y).astype(np.float32))

    traindata = data_utils.TensorDataset(trainx, trainy)
    return DataLoader(traindata, batch_size=128, shuffle=True, pin_memory=True)

def validload(X, Y):
    validx = torch.from_numpy(np.array(X).astype(np.float32))
    validy = torch.from_numpy(np.array(Y).astype(np.float32))

    validdata = data_utils.TensorDataset(validx, validy)
    return DataLoader(validdata, batch_size=1, shuffle=False, pin_memory=True)

