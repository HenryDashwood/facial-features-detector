import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

def load_coordinates_to_dataframe(labels_path):

    df = pd.read_csv(labels_path)

    for col in df:
        if (col[-4:] == 'name'):
            df = df.drop([col], axis=1)
    df = df.drop(['box/_top'], axis=1)
    df = df.drop(['box/_left'], axis=1)
    df = df.drop(['box/_width'], axis=1)
    df = df.drop(['box/_height'], axis=1)

    df.columns.values[-1] = "filename"
    for i in range(df.shape[0]):
        new = {df['filename'].values[i].replace('images/', '') for x in df['filename'].values[i]}
        new = repr(new)
        new = new[2:]
        new = new[:-2]
        df['filename'].values[i] = new

    i = 0
    j = 0
    while (i < 22):
        df.columns.values[i] = (str(j) + "_x")
        df.columns.values[i+1] = (str(j) + "_y")
        i += 2
        j += 1

    df = df.set_index('filename')

    return df

def resize(image_name, img, df):
    
    start_coords = np.array(list(df.loc[image_name])).reshape((11,2))

    keypoints = KeypointsOnImage.from_xy_array(start_coords, shape=img.shape)

    seq = iaa.Sequential([
        iaa.Resize({"height": 80, "width": 80})
    ])

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([img])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
        
    coords = keypoints_aug.to_xy_array().flatten()

    df.loc[image_name] = coords

    return image_aug, df

def load(images_path, labels_path):

    y = load_coordinates_to_dataframe(labels_path)
    X = []

    for filename in tqdm(list(y.index)):
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
