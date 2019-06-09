import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam

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
            ax.scatter(
                target_size/2*y.iloc[ipic][0::2]+target_size/2, 
                target_size/2*y.iloc[ipic][1::2]+target_size/2
            )
        else:
            ax.scatter(y.iloc[ipic][0::2], y.iloc[ipic][1::2])
        ax.set_title("picture "+ str(ipic))
        count += 1
    plt.show()
    
def plot_loss(hist, name, RMSE_TF=False):
    '''
    RMSE_TF: if True, then RMSE is plotted with original scale 
    '''
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    if RMSE_TF:
        loss = np.sqrt(np.array(loss)) * target_size/2 
        val_loss = np.sqrt(np.array(val_loss)) * target_size/2 

    plt.figure(figsize=(8,8))
    plt.plot(loss,"--",linewidth=3,label="train:"+name)
    plt.plot(val_loss,linewidth=3,label="val:"+name)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    
def save_model(model, name):
    json_string = model.to_json()
    open("../models/"+name+'_architecture.json', 'w').write(json_string)
    model.save_weights("../models/"+name+'_weights.h5')
    model.save("../models/"+name+'_weights_and_all.h5')

def load_model(name):
    model = model_from_json(open("../models/"+name+'_architecture.json').read())
    model.load_weights("../models/"+name + '_weights.h5')
    model.compile(loss="mean_squared_error", optimizer=Adam())
    return model