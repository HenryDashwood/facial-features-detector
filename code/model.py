from keras.models import Sequential
from keras.layers import (Dense, SeparableConv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, 
                          GlobalAveragePooling2D)
from keras.optimizers import Adam

def MyModel():

    model = Sequential()
    model.add(SeparableConv2D(32, 3, activation='relu', input_shape = (80, 80, 3)))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    model.add(Flatten())
#     model.add(GlobalAveragePooling2D());
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))

#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(rate=0.2))

    model.add(Dense(22))

    model.compile(loss="mean_squared_error", optimizer=Adam(lr=5e-5))

    return model
