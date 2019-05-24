import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
from math import floor
import imgaug as ia
from imgaug import augmenters as iaa

class Prepare():

    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

    def get_file_list_from_dir(self):
        all_files = os.listdir(self.images_path)
        data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
        shuffle(data_files)
        return data_files

    def load_coordinates_to_dataframe(self, full_size=False):

        df = pd.read_csv(self.labels_path)

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

        if (full_size==False):
            lst_imgs = [l for l in df.index.values]
            for img in lst_imgs:
                df.loc[img] = df.loc[img].round(0).astype(int)

        return df

    def convert_image_to_array(self, img):

        image = Image.open(self.images_path + img) # Use this line for rgb
    #     pic = Image.open(file_path + img).convert("L") # Use this line for greyscale
        image = np.array(image)

        return image

    def resize(self, image_name, img, df):

        keypoints = ia.KeypointsOnImage([
            ia.Keypoint(x=df.loc[image_name].values[i], y=df.loc[image_name].values[i+1]) for i in range(0,22,2)
        ], shape=img.shape)

        seq = iaa.Sequential([
            iaa.Resize({"height": 80, "width": 80})
        ])

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([img])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

        coords = []
        for i in range(11):
            coords.append(keypoints_aug.keypoints[i].x)
            coords.append(keypoints_aug.keypoints[i].y)

        df.loc[image_name] = coords

        return image_aug, df

    def load(self, test=False):

        file_list = self.get_file_list_from_dir()

        X = []
        y = self.load_coordinates_to_dataframe()

        lst_imgs = [l for l in y.index.values]

        for img in tqdm(lst_imgs):
            if img in file_list:
                image = self.convert_image_to_array(img)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image, y = self.resize(img, image, y)
                    X.append(image)
                else:
                    y = y.drop([img])
            else:
                y = y.drop([img])

        X = np.asarray(X)
        
        return X, y

if __name__ =="__main__":
    P = Prepare("../data/full_dataset/images_all/", "../data/full_dataset/landmarks_all.csv")
    X, y = P.load()
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=42)

    print(X_train.shape, X_val.shape, X_test.shape)
