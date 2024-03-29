from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, MaxPooling2D
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.core import Activation, Dropout
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils import plot_model


def denseBlock(t, nb_layers):
    for _ in range(nb_layers):
        tmp = t
        t = BatchNormalization(axis=1,
                                gamma_regularizer=l2(0.0001),
                                beta_regularizer=l2(0.0001))(t)

        t = Activation('relu')(t)
        t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
        t = Dropout(0.2)(t)
        t = concatenate([t, tmp])
    return t

def transitionDown(t, nb_features):
    t = BatchNormalization(axis=1,
                            gamma_regularizer=l2(0.0001),
                            beta_regularizer=l2(0.0001))(t)
    t = Activation('relu')(t)
    t = Conv2D(nb_features, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last')(t)
    return t

def Tiramisu(layer_per_block, n_pool=5, growth_rate=12):
    input_layer = Input(shape=(256, 256, 3))
    t = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    #dense block
    nb_features = 48
    skip_connections = []
    for i in range(n_pool):
        t = denseBlock(t, layer_per_block[i])
        print(t)
        skip_connections.append(t)
        nb_features += growth_rate * layer_per_block[i]
        t = transitionDown(t, nb_features)
        print(t)

    t = denseBlock(t, layer_per_block[n_pool]) # bottle neck

    skip_connections = skip_connections[::-1] #subvert the array

    for i in range(n_pool):
        keep_nb_features = growth_rate * layer_per_block[n_pool + i]
        t = Conv2DTranspose(keep_nb_features, strides=2, kernel_size=(3, 3), padding='same', data_format='channels_last')(t) # transition Up
        t = concatenate([t, skip_connections[i]])

        t = denseBlock(t, layer_per_block[n_pool+i+1])
        print(t)
    t = Conv2D(3, kernel_size=(1, 1), padding='same', activation='tanh', kernel_initializer='he_uniform', data_format='channels_last')(t)
    #output_layer = Activation('tanh')(t)
    print(t)
    return Model(inputs=input_layer, outputs=t)







