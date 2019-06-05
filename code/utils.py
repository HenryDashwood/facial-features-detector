import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_labelled_image(X, y, target_size, normalised=True):
    plt.imshow(X)
    if normalised:
        plt.scatter(target_size*y[0::2], target_size*y[1::2])
    else:
        plt.scatter(y[0::2], y[1::2])
        
def plot_labelled_sample(X, y, target_size, normalised=True):
    fig = plt.figure(figsize=(7, 7))
    fig.subplots_adjust(hspace=0.13, wspace=0.0001, left=0, right=1, bottom=0, top=1)
    Npicture = 9
    count = 1
    for irow in range(Npicture):
        ipic = np.random.choice(X.shape[0])
        ax = fig.add_subplot(Npicture/3, 3, count, xticks=[], yticks=[])        
        ax.imshow(X[ipic].reshape(target_size, target_size, 3), cmap="gray")
        if normalised:
            ax.scatter(target_size*y.iloc[ipic][0::2], target_size*y.iloc[ipic][1::2])
        else:
            ax.scatter(y.iloc[ipic][0::2], y.iloc[ipic][1::2])
        ax.set_title("picture "+ str(ipic))
        count += 1
    plt.show()