import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

class Prepare():

    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

    def load_coordinates_to_dataframe(self):

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

        return df

    def resize(self, image_name, img, df):

        keypoints = KeypointsOnImage([Keypoint(x=df.loc[image_name].values[i],
                                               y=df.loc[image_name].values[i+1]) for i in range(0,22,2)],
                                     shape=img.shape)

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

    def load(self):

        y = self.load_coordinates_to_dataframe()
        X = []

        for filename in tqdm(list(y.index)):
            img = np.array(Image.open(self.images_path+filename))
            if len(img.shape) == 3 and img.shape[2] == 3:
                img, y = self.resize(filename, img, y)
                X.append(img)
            else:
                y = y.drop(filename)

        X = np.asarray(X)

        return X, y

if __name__ == "__main__":
    P = Prepare("../data/images/", "../data/landmarks.csv")
    X, y = P.load()
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=42)

    print(X_train.shape, X_val.shape, X_test.shape)
