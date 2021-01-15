"""Convolutional Neural Networks.
===============================================
Version |Date |   Author|   Comment
-----------------------------------------------
0.0 | 31 Oct 2020 | J. Charlier | initial version
0.1 | 11 Nov 2020 | J. Charlier | bug update for 8x23 encoding
===============================================
"""
#
#
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
#import tensorflow.python.keras as tfkeras
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (
    Conv2D, MaxPooling2D, MaxPool2D,
    concatenate, BatchNormalization, 
    Dense, Dropout, Flatten, Input)
import matplotlib.pyplot as plt
p = print
flpath = 'drive/My Drive/crispor/models/'
#
#
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
        num_classes):
    if K.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], 1, imgrows, imgcols)
        xtest = xtest.reshape(xtest.shape[0], 1, imgrows, imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows, imgcols, 1)
        xtest = xtest.reshape(xtest.shape[0], imgrows, imgcols, 1)
        input_shape = (imgrows, imgcols, 1)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xtrain /= 255
    xtest /= 255
    p('xtrain shape:', xtrain.shape)
    p(xtrain.shape[0], 'train samples')
    p(xtest.shape[0], 'test samples')
    #
    # convert class vectors to binary class matrices
    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape
#
#
def cnnthree(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        cnn3 = Sequential()
        cnn3.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape
            )
        )
        cnn3.add(MaxPooling2D(pool_size=(2, 2)))
        cnn3.add(Dropout(0.25))
        cnn3.add(Flatten())
        cnn3.add(Dense(128, activation='relu'))
        cnn3.add(Dropout(0.5))
        cnn3.add(Dense(num_classes, activation='softmax'))
        #
        cnn3.compile(
            loss=categorical_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
            metrics=['accuracy']
        )
        #
        history_cnn3 = cnn3.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        score = cnn3.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        if True:
            plt.figure()
            plt.plot(history_cnn3.history['loss'], label='train loss')
            plt.plot(history_cnn3.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if np.cumprod(input_shape)[-1] == 92:
            cnn3 = tf.keras.models.load_model(
                flpath+'saved_model_4x23/cnn3_4x23'
            )
        else:
            if tl:
                cnn3 = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/cnn3_8x23'
                )
            else:
                cnn3 = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/cnn3crispr_8x23'
                )
    p("CNN3: Done")
    return cnn3
#
#
def cnnfive(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        cnn5 = Sequential()
        cnn5.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape
            )
        )
        cnn5.add(Conv2D(64, (1, 1), activation='relu'))
        cnn5.add(MaxPooling2D(pool_size=(2, 2)))
        cnn5.add(Dropout(0.25))
        cnn5.add(Flatten())
        cnn5.add(Dense(128, activation='relu'))
        cnn5.add(Dropout(0.5))
        cnn5.add(Dense(num_classes, activation='softmax'))
        #
        cnn5.compile(
            loss=categorical_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
            metrics=['accuracy']
        )
        #
        history_cnn5layers = cnn5.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        score = cnn5.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        if True:
            plt.figure()
            plt.plot(history_cnn5layers.history['loss'], label='train loss')
            plt.plot(history_cnn5layers.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if np.cumprod(input_shape)[-1] == 92:
            cnn5 = tf.keras.models.load_model(
                flpath+'saved_model_4x23/cnn5_4x23'
            )
        else:
            if tl:
                cnn5 = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/cnn5_8x23'
                )
            else:
                cnn5 = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/cnn5crispr_8x23'
                )
    p("CNN5: Done")
    return cnn5
#
#
def cnnten(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        cnn10 = Sequential()
        cnn10.add(
            Conv2D(
                32,
                kernel_size=(1, 1),
                padding="same",
                activation='relu',
                input_shape=input_shape
            )
        )
        cnn10.add(MaxPooling2D(pool_size=(2, 2)))
        cnn10.add(Conv2D(64, (1, 1), padding="same", activation='relu'))
        cnn10.add(MaxPooling2D(pool_size=(2, 2)))
        cnn10.add(Dropout(0.25))
        cnn10.add(Flatten())
        cnn10.add(Dense(128, activation='relu'))
        cnn10.add(Dropout(0.5))
        cnn10.add(Dense(64, activation='relu'))
        cnn10.add(Dropout(0.5))
        cnn10.add(Dense(num_classes, activation='softmax'))
        #
        #
        cnn10.compile(
            loss=categorical_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
            metrics=['accuracy']
        )
        #
        history_cnn10layers = cnn10.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=100,
            verbose=0,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        score = cnn10.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        if True:
            plt.figure()
            plt.plot(history_cnn10layers.history['loss'], label='train loss')
            plt.plot(history_cnn10layers.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if np.cumprod(input_shape)[-1] == 92:
            cnn10 = tf.keras.models.load_model(
                flpath+'saved_model_4x23/cnn10_4x23'
            )
        else:
            if tl:
                cnn10 = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/cnn10_8x23'
                )
            else:
                cnn10 = tf.keras.models.load_model(
                   flpath+'saved_model_crispr_8x23/cnn10crispr_8x23'
                )
    p("CNN10: Done")
    return cnn10
#
#
def cnnlin(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        inputs = Input(shape=input_shape, name='main_input')
        conv_1 = Conv2D(10, (1,1), padding='same', activation='relu')(inputs)
        conv_2 = Conv2D(10, (1,2), padding='same', activation='relu')(inputs)
        conv_3 = Conv2D(10, (1,3), padding='same', activation='relu')(inputs)
        conv_4 = Conv2D(10, (1,5), padding='same', activation='relu')(inputs)
        #
        conv_output = concatenate([conv_1, conv_2, conv_3, conv_4])
        bn_output = BatchNormalization()(conv_output)
        pooling_output = MaxPool2D(pool_size=(1,5), strides=None, padding='valid')(bn_output)
        flatten_output = Flatten()(pooling_output)
        #
        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(0.15)(x)
        predictions = Dense(num_classes, name='main_output')(x)
        #
        cnnlin = Model(inputs, predictions)
        adamopt = tf.keras.optimizers.Adam(lr=1e-4)
        cnnlin.compile(
            loss='binary_crossentropy', 
            optimizer=adamopt,
            metrics=['accuracy']
        )
        #
        history_cnnreplica = cnnlin.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=20,
            verbose=1,
            validation_data=(xtest, ytest)
        )
        #
        if True:
            plt.figure()
            plt.plot(history_cnnreplica.history['loss'], label='train loss')
            plt.plot(history_cnnreplica.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if np.cumprod(input_shape)[-1] == 92:
            cnnlin = tf.keras.models.load_model(
                flpath+'saved_model_4x23/cnnlinn_4x23'
            )
        else:
            if tl:
                cnnlin = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/cnnlinn_8x23'
                )
            else:
                cnnlin = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/cnnlinncrispr_8x23'
                )
    p("CNN Lin: Done")
    return cnnlin
#
# Last card of module cnns.
#
