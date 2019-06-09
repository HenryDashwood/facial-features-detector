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

def augment(imgs, kps):
    seq = iaa.Sequential([
                          iaa.Fliplr(0.5),
                          iaa.Affine(
                              scale=(0.5, 1),
                              mode="symmetric"
                          )  
                        ])
    
    keypoints = [KeypointsOnImage.from_xy_array(kps[i], shape=imgs[i].shape) for i in range(kps.shape[0])]
    imgs_aug, kps_aug = seq(images=imgs, keypoints=keypoints)
    kps_aug = [kps_aug[i].to_xy_array().flatten() for i in range(len(kps_aug))]
    
    return imgs_aug, kps_aug

def generator(X, y, batch_size, target_size):
    
    temp_X = X.copy()
    temp_y = y.copy()
    
    batch_features = np.zeros((batch_size, target_size, target_size, 3))
    batch_labels = np.zeros((batch_size, 22)) 
    
    while True:
        indices = np.random.choice(temp_X.shape[0], batch_size)
        ks = temp_y.iloc[indices].values.reshape(batch_size,11,2)
        random_augmented_image, random_augmented_labels = augment(temp_X[indices], ks)

        batch_features = random_augmented_image
        batch_labels = random_augmented_labels
        
        batch_features = batch_features / 255
        batch_labels = (pd.DataFrame(batch_labels) - (target_size/2)) / (target_size/2)
                        
        yield batch_features, batch_labels

if __name__ == "__main__":
    X, y = load("../data/images/", "../data/landmarks.csv")
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=42)

    print(X_train.shape, X_val.shape, X_test.shape)