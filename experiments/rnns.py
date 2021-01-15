"""Recurrent Neural Networks.
===============================================
Version |    Date     |   Author    |   Comment
-----------------------------------------------
0.0     | 19 Dec 2020 | J. Charlier | initial version
0.1     | 29 Dec 2020 | J. Charlier | architecture update for LSTM & GRU
===============================================
"""
#
#
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (
    Conv2D, MaxPooling2D, MaxPool2D,
    concatenate, BatchNormalization, 
    Dense, Dropout, Flatten, Input
)
import matplotlib.pyplot as plt
flpath = 'drive/My Drive/crispor/models/'
#
#
def lstmmdl(
        xtrain, ytrain,
        xtest, ytest,
        num_classes,
        batch_size,
        epochs,
        callbacks,
        imgrows,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        mdl = Sequential()
        mdl.add(layers.LSTM(92))
        mdl.add(layers.Dense(92))
        mdl.add(layers.BatchNormalization())
        mdl.add(layers.Dense(92))
        mdl.add(layers.Dropout(0.25))
        mdl.add(layers.Dense(2, activation='softmax'))
        mdl.compile(
                    loss=binary_crossentropy,
                    optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
                    metrics=['accuracy'])
        histmdl = mdl.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(xtest, ytest),
                    callbacks=callbacks)
        score = mdl.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        # display learning curves
        if True:
            plt.figure()
            plt.plot(histmdl.history['loss'], label='train loss')
            plt.plot(histmdl.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if imgrows == 4:
            mdl = tf.keras.models.load_model(
                flpath+'saved_model_4x23/lstm_4x23'
            )
        else:
            if tl:
                mdl = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/lstm_8x23'
                )
            else:
                mdl = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/lstmcrispr_8x23'
                )
    print("LSTM: Done")
    return mdl
#
#
def grumdl(
        xtrain, ytrain,
        xtest, ytest,
        num_classes,
        batch_size,
        epochs,
        callbacks,
        imgrows,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        mdl = Sequential()
        mdl.add(layers.GRU(92))
        mdl.add(layers.Dense(92))
        mdl.add(layers.BatchNormalization())
        mdl.add(layers.Dense(92))
        mdl.add(layers.Dropout(0.25))
        mdl.add(layers.Dense(2, activation='softmax'))
        mdl.compile(
                    loss=binary_crossentropy,
                    optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
                    metrics=['accuracy'])
        histmdl = mdl.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(xtest, ytest),
                    callbacks=callbacks)
        score = mdl.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        # display learning curves
        if True:
            plt.figure()
            plt.plot(histmdl.history['loss'], label='train loss')
            plt.plot(histmdl.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if imgrows == 4:
            mdl = tf.keras.models.load_model(
                flpath+'saved_model_4x23/gru_4x23'
            )
        else:
            if tl:
                mdl = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/gru_8x23'
                )
            else:
                mdl = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/grucrispr_8x23'
                )
    print("GRU: Done")
    return mdl
#
# Last card of module rnns.
#