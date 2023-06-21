# Test file to process the images
import cv2
import numpy as np
import pandas as pd
import matplotlib as plt

PATH = '/groups/CS156b/data/'

fil = pd.read_csv("train.csv")
print(len(fil['Path']))
cnt = 0
for path in fil['Path']:
    if cnt % 1000 == 0:
        print(cnt)
    if cnt < 10:
        img = cv2.imread(PATH + path)
        new_img = cv2.resize(img, (128,128))
        pth = path.replace("/","_")
        cv2.imwrite(f"resized64/{pth}", new_img)
        edges = cv2.Canny(image=new_img, threshold1=75, threshold2=120)
        cv2.imwrite(f"resized64edges/{pth}",edges)
    cnt += 1
