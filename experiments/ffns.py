"""FeedForward Networks.
===============================================
Version |Date |   Author|   Comment
-----------------------------------------------
0.0 | 31 Oct 2020 | J. Charlier | initial version
0.1 | 11 Nov 2020 | J. Charlier | bug update for 8x23 encoding
0.2 | 12 Nov 2020 | J. Charlier | bug fix
===============================================
"""
#
#
import tensorflow as tf
from keras.utils import to_categorical
#import tensorflow.python.keras as tfkeras
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (
    Conv2D, MaxPooling2D, MaxPool2D,
    concatenate, BatchNormalization, 
    Dense, Dropout, Flatten, Input
)
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
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows*imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows*imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows*imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows*imgcols)
    input_shape = (imgrows*imgcols)
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
def ffnthree(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        ffn3 = Sequential()
        ffn3.add(
            Dense(
                100, 
                input_dim=input_shape, 
                kernel_initializer="lecun_uniform", 
                activation="relu"
                )
            )
        ffn3.add(BatchNormalization())
        ffn3.add(Dense(50, activation="relu", kernel_initializer="uniform"))
        ffn3.add(Dropout(0.5))
        ffn3.add(Dense(10, activation="relu", kernel_initializer="uniform"))
        ffn3.add(Dense(num_classes, activation='softmax'))
        #
        ffn3.compile(
            loss=binary_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
            metrics=['accuracy']
        )
        #
        historyffn3 = ffn3.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        score = ffn3.evaluate(xtest, ytest, verbose=0)
        p('Test loss:', score[0])
        p('Test accuracy:', score[1])
        #
        # display learning curves
        if True:
            plt.figure()
            plt.plot(historyffn3.history['loss'], label='train loss')
            plt.plot(historyffn3.history['val_loss'], label='test loss')
            plt.title('Learning Curves')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    else:
        if input_shape == 92:
            ffn3 = tf.keras.models.load_model(
                flpath+'saved_model_4x23/ffn3_4x23'
            )
        else:
            if tl:
                ffn3 = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/ffn3_8x23'
                )
            else:
                ffn3 = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/ffn3crispr_8x23'
                )
    p("FFN3: Done")
    return ffn3
#
#
def ffnfive(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        ffn = Sequential()
        ffn.add(
            Dense(
                100,
                input_dim=input_shape,
                kernel_initializer="uniform",
                activation="relu"
            )
        )
        ffn.add(BatchNormalization())
        ffn.add(Dense(75, activation="relu", kernel_initializer="uniform"))
        ffn.add(BatchNormalization())
        ffn.add(Dense(50, activation="relu", kernel_initializer="uniform"))
        ffn.add(Dropout(0.25))
        ffn.add(Dense(25, activation="relu", kernel_initializer="uniform"))
        ffn.add(Dropout(0.5))
        ffn.add(Dense(10, activation="relu", kernel_initializer="uniform"))
        ffn.add(Dense(num_classes, activation='softmax'))
        #
        ffn.compile(
            loss=categorical_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
            metrics=['accuracy']
        )
        #
        historyffn = ffn.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        score = ffn.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        if True:
            plt.figure()
            plt.plot(historyffn.history['loss'], label='train loss')
            plt.plot(historyffn.history['val_loss'], label='test loss')
            #plt.title('Learning Curve of FFN5')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend(loc='best')
            plt.savefig('epochslossffn5.pdf')
    else:
        if input_shape == 92:
            ffn = tf.keras.models.load_model(
                flpath+'saved_model_4x23/ffn5_4x23'
            )
        else:
            if tl:
                ffn = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/ffn5_8x23'
                )
            else:
                ffn = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/ffn5crispr_8x23'
                )
    p("FFN5: Done")
    return ffn
#
#
def ffnten(
        xtrain, ytrain,
        xtest, ytest,
        input_shape, num_classes,
        batch_size, epochs,
        callbacks,
        ismodelsaved=False,
        tl=False):
    if ismodelsaved == False:
        # model definition
        ffn10 = Sequential()
        ffn10.add(
            Dense(
                200,
                input_dim=input_shape,
                kernel_initializer="uniform",
                activation="relu"
            )
        )
        ffn10.add(BatchNormalization())
        ffn10.add(Dense(175, activation="relu", kernel_initializer="uniform"))
        ffn10.add(BatchNormalization())
        ffn10.add(Dense(150, activation="relu", kernel_initializer="uniform"))
        ffn10.add(Dropout(0.25))
        ffn10.add(Dense(125, activation="relu", kernel_initializer="uniform"))
        ffn10.add(BatchNormalization())
        ffn10.add(Dense(100, activation="relu", kernel_initializer="uniform"))
        ffn10.add(BatchNormalization())
        ffn10.add(Dense(100, activation="relu", kernel_initializer="uniform"))
        ffn10.add(Dropout(0.25))
        ffn10.add(Dense(75, activation="relu", kernel_initializer="uniform"))
        ffn10.add(BatchNormalization())
        ffn10.add(Dense(50, activation="relu", kernel_initializer="uniform"))
        ffn10.add(Dropout(0.25))
        ffn10.add(Dense(25, activation="relu", kernel_initializer="uniform"))
        ffn10.add(Dropout(0.5))
        ffn10.add(Dense(10, activation="relu", kernel_initializer="uniform"))
        ffn10.add(Dense(num_classes, activation='softmax'))
        #
        ffn10.compile(
            loss=categorical_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
            metrics=['accuracy']
        )
        #
        historyffn10 = ffn10.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        score = ffn10.evaluate(xtest, ytest, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #
        if True:
            plt.figure()
            plt.plot(historyffn10.history['loss'], label='train loss')
            plt.plot(historyffn10.history['val_loss'], label='test loss')
            #plt.title('Learning Curve of FFN10')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig('epochslossffn10.pdf')
    else:
        if input_shape == 92:
            ffn10 = tf.keras.models.load_model(
                flpath+'saved_model_4x23/ffn10_4x23'
            )
        else:
            if tl:
                ffn10 = tf.keras.models.load_model(
                    flpath+'saved_model_guideseq_8x23/ffn10_8x23'
                )
            else:
                ffn10 = tf.keras.models.load_model(
                    flpath+'saved_model_crispr_8x23/ffn10crispr_8x23'
                )
    p("FFN10: Done")
    return ffn10
#
# Last card of module ffns.
#