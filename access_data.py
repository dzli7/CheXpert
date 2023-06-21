import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import torch

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

cuda0 = torch.device('cuda:0')

def train_df():
    return pd.read_csv(train_path)

def get_train_data():
    df = train_df()
    X = []
    Y = []

    for index, row in df.iterrows():
        image_path = data_path + row['Path']

        X.append(mpimg.imread(image_path))
        Y.append(row[label_columns])
    
    X_train = torch.Tensor(X, dtype=torch.float32, device=cuda0)
    Y_train = torch.Tensor(Y, dtype=torch.float32, device=cuda0)
    return X_train, Y_train

def main():
    X_train, Y_train = get_train_data()
    print(X_train.size())
    print(Y_train.size())

if __name__ == '__main__':
    main()
