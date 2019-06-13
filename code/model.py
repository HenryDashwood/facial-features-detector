from keras.models import Sequential
from keras.layers import (Dense, SeparableConv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, 
                          GlobalAveragePooling2D)
from keras.optimizers import Adam
# from keras.applications import VGG16

# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

def MyModel(target_size, max_lr):

    model = Sequential()
    model.add(SeparableConv2D(32, 3, activation='relu', input_shape = (target_size, target_size, 3)))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(22))

    model.compile(loss="mean_squared_error", optimizer=Adam(lr=max_lr))

    return model
