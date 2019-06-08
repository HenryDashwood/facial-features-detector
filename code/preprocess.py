import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from fastprogress import progress_bar
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from os import listdir
from matplotlib import image

def load_coordinates_to_dataframe(labels_path):
    df = pd.read_csv(labels_path)
    
    df.columns.values[-1] = "filename"
    df['filename'] = df['filename'].str.replace('images/', '')
    df = df.set_index('filename')
    
    df = df.drop(['box/_top', 'box/_left', 'box/_width', 'box/_height'], axis=1)
    for col in df:
        if (col[-4:] == 'name'):
            df = df.drop([col], axis=1) 
    
    df.columns = df.columns.str.replace('box/part/', '')
    df.columns = df.columns.str.replace('/', '')

    return df

def resize(image_name, img, df):  
    start_coords = np.array(list(df.loc[image_name])).reshape((11,2))

    keypoints = KeypointsOnImage.from_xy_array(start_coords, shape=img.shape)

    seq = iaa.Sequential([
        iaa.Resize({"height": 80, "width": 80})
    ])

    image_aug, keypoints_aug = seq(image=img, keypoints=keypoints)
        
    coords = keypoints_aug.to_xy_array().flatten()

    df.loc[image_name] = coords

    return image_aug, df

def load(images_path, labels_path):
    y = load_coordinates_to_dataframe(labels_path)
    X = []
                
    for filename in progress_bar(list(y.index)):
        img = np.array(Image.open(images_path+filename))
        if len(img.shape) == 3 and img.shape[2] == 3:
            img, y = resize(filename, img, y)
            X.append(img)
        else:
            y = y.drop(filename)

    X = np.asarray(X)

    return X, y

if __name__ == "__main__":
    X, y = load("../data/images/", "../data/landmarks.csv")
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=42)

    print(X_train.shape, X_val.shape, X_test.shape)