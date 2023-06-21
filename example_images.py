import numpy as np
import pandas as pd
import matplotlib.image as mpimg

#['Unnamed: 0', 'Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA',
# 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
#'Fracture', 'Support Devices']
label_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly','Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia','Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other','Fracture', 'Support Devices']

data_path = '/central/groups/CS156b/data/'
train_path = data_path + 'student_labels/train.csv'


def train_df():
    return pd.read_csv(train_path)

def get_images():
    df = train_df()

    for label in label_columns:
        df_label = df.dropna(subset=[label])
        path = df_label.iloc[0]
        image_path = data_path + row['Path']
        image = mpimg.imread(image_path)

        save_path = '/home/ding/' + label.lower() + '.jpg'
        plt.imsave(save_path, image, cmap='gray')

def main():
    get_images()

if __name__ == '__main__':
    main()
