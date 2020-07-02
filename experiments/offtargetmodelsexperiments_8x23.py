# -*- coding: utf-8 -*-
"""offtargetmodelsexperiments_8x23.ipynb ."""

from __future__ import print_function

import os
import time
import random

import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV)
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score,
                             roc_curve, precision_score, recall_score,
                             auc, average_precision_score,
                             precision_recall_curve, accuracy_score)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC

import tensorflow as tf
import tensorflow.python.keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (Conv2D, MaxPooling2D, MaxPool2D,
                                            concatenate, BatchNormalization,
                                            Dense, Dropout, Flatten, Input)
from tensorflow.python.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, array_to_img)
import tensorflow.python.keras as tfkeras

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
random.seed(42)


# user defined founction for the vgg16 and vgg19
def grayscale_to_rgb(images, channel_axis=-1):
    """Transform a gray scale image to a RGB format image."""
    images = K.expand_dims(images, axis=channel_axis)
    tiling = [1] * 4    # 4 dimensions: B, H, W, C
    tiling[channel_axis] *= 3
    images = K.tile(images, tiling)
    return images


def dispConfMatrixAsArray(y_test, ypred, disp=True):
    """Display confusion matrix as an array."""
    confData = {'y': y_test, 'ypred': ypred}
    confdf = pd.DataFrame(confData, columns=['y', 'ypred'])
    confmatrix = pd.crosstab(
        confdf.y, confdf.ypred,
        rownames=['target'], colnames=['predicted'])
    if disp is True:
        print('\nConfusion Matrix')
        if len(np.unique(ypred)) >= 2:
            print("%-3s" % 'TN:', "%-5s" % confmatrix.loc[0, 0],
                "|  %-3s" % 'FP:', "%-5s" % confmatrix.loc[0, 1])
            print("%-3s" % 'FN:', "%-5s" % confmatrix.loc[1, 0],
                "|  %-3s" % 'TP:', "%-5s" % confmatrix.loc[1, 1])
        else:
            print("%-3s" % 'TN:', "%-5s" % confmatrix.loc[0, 0])
            print("%-3s" % 'FN:', "%-5s" % confmatrix.loc[1, 0])
    return confmatrix


def plotConfusionMatrix(y_test, ypred):
    """Plot confusion matrix with seaborn."""
    confData = {'y': y_test, 'ypred': ypred}
    confdf = pd.DataFrame(confData, columns=['y', 'ypred'])
    confMatrix = pd.crosstab(
        confdf.y, confdf.ypred,
        rownames=['Target'], colnames=['Predicted'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(confMatrix, annot=True, fmt='d')
    plt.show()


def getClassificationMetrics(y_test, yscore, ypred):
    """Get classification metrics."""
    posLabel = np.unique(y_test)
    print("\nModel Metrics:")
    print("%-40s" % ("ROC AUC Score:"), "{:.3f}".format(roc_auc_score(
          y_test, yscore[:, 1])))
    for n in posLabel:
        print("%-40s" % ("F1 Score Class " + str(n) + " :"),
              "{:.3f}".format(f1_score(
                  y_test, ypred, pos_label=n)))
        print("%-40s" % ("Recall Score Class " + str(n) + " :"),
              "{:.3f}".format(recall_score(y_test, ypred, pos_label=n)))
        print("%-40s" % ("Avrg Precision Score Class "+str(n)+" :"),
              "{:.3f}".format(average_precision_score(
                    y_test, yscore[:, 1], pos_label=n)))


def brierScore(y_test, yscore):
    """Compute the Brier score, 0 is the best achievable, 1 the worst."""
    bscore = (1/len(y_test))
    bscore *= np.sum(np.power(yscore[:, 1] - y_test, 2))
    return bscore


# Incorporating reduced learning and early stopping for NN callback
reduce_learning = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=8, verbose=1,
    mode='auto', min_delta=0.02,
    cooldown=0, min_lr=0)
eary_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0001,
    patience=20, verbose=1, mode='auto')
callbacks = [reduce_learning, eary_stopping]

ismodelsaved = True
isguideseq = True
if ismodelsaved is True:
    if isguideseq is False:
        ffn3 = tf.keras.models.load_model('/saved_model_crispr_8x23/ffn3crispr_8x23')
        ffn = tf.keras.models.load_model('/saved_model_crispr_8x23/ffn5crispr_8x23')
        ffn10 = tf.keras.models.load_model('/saved_model_crispr_8x23/ffn10crispr_8x23')
        model_cnn3 = tf.keras.models.load_model('/saved_model_crispr_8x23/cnn3crispr_8x23')
        model_cnn5layers = tf.keras.models.load_model('/saved_model_crispr_8x23/cnn5crispr_8x23')
        model_cnn10layers = tf.keras.models.load_model('/saved_model_crispr_8x23/cnn10crispr_8x23')
        model_cnnreplica = tf.keras.models.load_model('/saved_model_crispr_8x23/cnnlinncrispr_8x23')
    else:
        ffn3 = tf.keras.models.load_model('/saved_model_guideseq_8x23/ffn3_8x23')
        ffn = tf.keras.models.load_model('/saved_model_guideseq_8x23/ffn5_8x23')
        ffn10 = tf.keras.models.load_model('/saved_model_guideseq_8x23/ffn10_8x23')
        model_cnn3 = tf.keras.models.load_model('/saved_model_guideseq_8x23/cnn3_8x23')
        model_cnn5layers = tf.keras.models.load_model('/saved_model_guideseq_8x23/cnn5_8x23')
        model_cnn10layers = tf.keras.models.load_model('/saved_model_guideseq_8x23/cnn10_8x23')
        model_cnnreplica = tf.keras.models.load_model('/saved_model_guideseq_8x23/cnnlinn_8x23')

# data read
# -*-*-*-*-
imgrows = 8
nexp = 3

imgcols = 23
num_classes = 2
epochs = 500
batch_size = 64
undersampling = False

# we import the pkl file containing the data
if imgrows == 4:
    print('refer to experiments 8x23 py file')
else:
    if nexp == 1:
        loaddata = pkl.load(open('encoded8x23.pkl', 'rb'), encoding='latin1')
    elif nexp == 2:
        loaddata = pkl.load(open(
            'encoded8x23linn.pkl', 'rb'), encoding='latin1')
    elif nexp == 3:
        if isguideseq is False:
            loaddata = pkl.load(open(
                'encoded8x23withoutTsai.pkl', 'rb'), encoding='latin1')
        else:
            loaddata = pkl.load(open(
                'encoded8x23withouttsai2.pkl', 'rb'), encoding='latin1')
    elif nexp == 4:
        loaddata = pkl.load(open('guideseq8x23.pkl', 'rb'), encoding='latin1')

# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images, pd.Series(loaddata.target),
    test_size=0.3, shuffle=True,
    random_state=42)

if undersampling is True:
    # Divide by class
    ratio = 0.2
    xtrainclass0 = x_train[y_train == 0]
    xtrainclass1 = x_train[y_train == 1]
    ytrainclass0 = y_train[y_train == 0]
    ytrainclass1 = y_train[y_train == 1]
    indx = random.sample(
        list(np.arange(0, len(xtrainclass0))),
        np.int64(len(xtrainclass1)/ratio))
    xtrainclass0under = xtrainclass0[indx]
    ytrainclass0under = ytrainclass0[indx]
    x_train = np.concatenate((xtrainclass0under, xtrainclass1), axis=0)
    y_train = np.concatenate((ytrainclass0under, ytrainclass1), axis=0)

# Feed Forward Network - 3 layers
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# input image dimensions
img_rows, img_cols = imgrows, imgcols

if K.image_data_format() == 'channels_first':
    x_trainffn = x_train.reshape(x_train.shape[0], img_rows*img_cols)
    x_testffn = x_test.reshape(x_test.shape[0], img_rows*img_cols)
    input_shape = (img_rows*img_cols)
else:
    x_trainffn = x_train.reshape(x_train.shape[0], img_rows*img_cols)
    x_testffn = x_test.reshape(x_test.shape[0], img_rows*img_cols)
    input_shape = (img_rows*img_cols)

x_trainffn = x_trainffn.astype('float32')
x_testffn = x_testffn.astype('float32')
x_trainffn /= 255
x_testffn /= 255
print('x_train shape:', x_trainffn.shape)
print(x_trainffn.shape[0], 'train samples')
print(x_testffn.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_trainffn = to_categorical(y_train, num_classes)
y_testffn = to_categorical(y_test, num_classes)

if ismodelsaved is False:
    # model definition
    ffn3 = Sequential()
    ffn3.add(Dense(
        100, input_dim=input_shape,
        kernel_initializer="lecun_uniform", activation="relu"))
    ffn3.add(BatchNormalization())
    ffn3.add(Dense(50, activation="relu", kernel_initializer="uniform"))
    ffn3.add(Dropout(0.5))
    ffn3.add(Dense(10, activation="relu", kernel_initializer="uniform"))
    ffn3.add(Dense(num_classes, activation='softmax'))

    ffn3.compile(
        loss=tfkeras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
        metrics=['accuracy'])

    historyffn3 = ffn3.fit(
        x_trainffn, y_trainffn,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_testffn, y_testffn),
        callbacks=callbacks)
    score = ffn3.evaluate(x_testffn, y_testffn, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

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

# Feed Forward Network - 5 layers
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
if ismodelsaved is False:
    # model definition
    ffn = Sequential()
    ffn.add(Dense(
        100, input_dim=input_shape,
        kernel_initializer="uniform", activation="relu"))
    ffn.add(BatchNormalization())
    ffn.add(Dense(75, activation="relu", kernel_initializer="uniform"))
    ffn.add(BatchNormalization())
    ffn.add(Dense(50, activation="relu", kernel_initializer="uniform"))
    ffn.add(Dropout(0.25))
    ffn.add(Dense(25, activation="relu", kernel_initializer="uniform"))
    ffn.add(Dropout(0.5))
    ffn.add(Dense(10, activation="relu", kernel_initializer="uniform"))
    ffn.add(Dense(num_classes, activation='softmax'))

    ffn.compile(loss=tfkeras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
                metrics=['accuracy'])

    historyffn = ffn.fit(
        x_trainffn, y_trainffn,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_testffn, y_testffn),
        callbacks=callbacks)
    score = ffn.evaluate(x_testffn, y_testffn, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if True:
        plt.figure()
        plt.plot(historyffn.history['loss'], label='train loss')
        plt.plot(historyffn.history['val_loss'], label='test loss')
        plt.title('Learning Curves')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

# Feed Forward Network - 10 layers
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
if ismodelsaved is False:
    # model definition
    ffn10 = Sequential()
    ffn10.add(Dense(
        200, input_dim=input_shape,
        kernel_initializer="uniform", activation="relu"))
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

    ffn10.compile(
        loss=tfkeras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
        metrics=['accuracy'])

    historyffn10 = ffn10.fit(
        x_trainffn, y_trainffn,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_testffn, y_testffn),
        callbacks=callbacks)
    score = ffn10.evaluate(x_testffn, y_testffn, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if True:
        plt.figure()
        plt.plot(historyffn10.history['loss'], label='train loss')
        plt.plot(historyffn10.history['val_loss'], label='test loss')
        plt.title('Learning Curves')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

# CNN 5 layers
# -*-*-*-*-*-*

# input image dimensions
img_rows, img_cols = imgrows, imgcols

if K.image_data_format() == 'channels_first':
    x_traincnn = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_testcnn = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_traincnn = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_testcnn = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_traincnn = x_traincnn.astype('float32')
x_testcnn = x_testcnn.astype('float32')
x_traincnn /= 255
x_testcnn /= 255
print('x_train shape:', x_traincnn.shape)
print(x_traincnn.shape[0], 'train samples')
print(x_testcnn.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_traincnn = to_categorical(y_train, num_classes)
y_testcnn = to_categorical(y_test, num_classes)

if ismodelsaved is False:
    # model definition
    model_cnn5layers = Sequential()
    model_cnn5layers.add(Conv2D(
        32, kernel_size=(3, 3),
        activation='relu', input_shape=input_shape))
    model_cnn5layers.add(Conv2D(64, (1, 1), activation='relu'))
    model_cnn5layers.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn5layers.add(Dropout(0.25))
    model_cnn5layers.add(Flatten())
    model_cnn5layers.add(Dense(128, activation='relu'))
    model_cnn5layers.add(Dropout(0.5))
    model_cnn5layers.add(Dense(num_classes, activation='softmax'))

    model_cnn5layers.compile(
        loss=tfkeras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
        metrics=['accuracy'])

    history_cnn5layers = model_cnn5layers.fit(
        x_traincnn, y_traincnn,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_testcnn, y_testcnn),
        callbacks=callbacks)
    score = model_cnn5layers.evaluate(x_testcnn, y_testcnn, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if True:
        plt.figure()
        plt.plot(history_cnn5layers.history['loss'], label='train loss')
        plt.plot(history_cnn5layers.history['val_loss'], label='test loss')
        plt.title('Learning Curves')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

# CNN 3 layers
# -*-*-*-*-*-*
if ismodelsaved is False:
    model_cnn3 = Sequential()
    model_cnn3.add(Conv2D(
        32, kernel_size=(3, 3),
        activation='relu', input_shape=input_shape))
    model_cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn3.add(Dropout(0.25))
    model_cnn3.add(Flatten())
    model_cnn3.add(Dense(128, activation='relu'))
    model_cnn3.add(Dropout(0.5))
    model_cnn3.add(Dense(num_classes, activation='softmax'))

    model_cnn3.compile(
        loss=tfkeras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(0.001, rho=0.9),
        metrics=['accuracy'])

    history_cnn3 = model_cnn3.fit(
        x_traincnn, y_traincnn,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_testcnn, y_testcnn),
        callbacks=callbacks)
    score = model_cnn3.evaluate(x_testcnn, y_testcnn, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if True:
        plt.figure()
        plt.plot(history_cnn3.history['loss'], label='train loss')
        plt.plot(history_cnn3.history['val_loss'], label='test loss')
        plt.title('Learning Curves')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

# Random Forest, SVM and Logistic Regression
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
x_trainrf = x_train.astype('float32').reshape(-1, imgrows*imgcols)
x_testrf = x_test.astype('float32').reshape(-1, imgrows*imgcols)
x_trainrf /= 255
x_testrf /= 255
print('x_train shape:', x_trainrf.shape)
print(x_trainrf.shape[0], 'train samples')
print(x_testrf.shape[0], 'test samples')

# random forest definition and training
rf = RandomForestClassifier(
    bootstrap=True, ccp_alpha=0.0, class_weight=None,
    criterion='gini', max_depth=None, max_features='auto',
    max_leaf_nodes=None, max_samples=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=1000,
    n_jobs=None, oob_score=False, random_state=42,
    verbose=0, warm_start=False)
rf.fit(x_trainrf, y_train)
print("RF Training: Done")

fpr_ffn3, tpr_ffn3, _ = roc_curve(
    y_testffn[:, 1], ffn3.predict(x_testffn)[:, 1])
fpr_ffn, tpr_ffn, _ = roc_curve(
    y_testffn[:, 1], ffn.predict(x_testffn)[:, 1])
fpr_ffn10, tpr_ffn10, _ = roc_curve(
    y_testffn[:, 1], ffn10.predict(x_testffn)[:, 1])

fpr_cnn3, tpr_cnn3, _ = roc_curve(
    y_testcnn[:, 1], model_cnn3.predict(x_testcnn)[:, 1])
fpr_cnn5layers, tpr_cnn5layers, _ = roc_curve(
    y_testcnn[:, 1], model_cnn5layers.predict(x_testcnn)[:, 1])

fpr_rf, tpr_rf, _ = roc_curve(
    y_test, rf.predict_proba(x_testrf)[:, 1])

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(
    fpr_ffn3, tpr_ffn3,
    label='FNN 3 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_ffn3, tpr_ffn3), 3)))
plt.plot(
    fpr_ffn, tpr_ffn,
    label='FNN 5 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_ffn, tpr_ffn), 3)))
plt.plot(
    fpr_ffn10, tpr_ffn10,
    label='FNN 10 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_ffn10, tpr_ffn10), 3)))
plt.plot(
    fpr_cnn3, tpr_cnn3,
    label='CNN 3 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_cnn3, tpr_cnn3), 3)))
plt.plot(
    fpr_cnn5layers, tpr_cnn5layers,
    label='CNN 5 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_cnn5layers, tpr_cnn5layers), 3)))
plt.plot(
    fpr_rf, tpr_rf,
    label='RF (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_rf, tpr_rf), 3)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best')
plt.savefig('roccurvecrispr8x23.pdf')
plt.show()

# PRECISION RECALL CURVE
# -*-*-*-*-*-*-*-*-*-*-*

icol = 1

pre_ffn3, rec_ffn3, _ = precision_recall_curve(
    y_testffn[:, icol], ffn3.predict(x_testffn)[:, icol], pos_label=1)
pre_ffn, rec_ffn, _ = precision_recall_curve(
    y_testffn[:, icol], ffn.predict(x_testffn)[:, icol], pos_label=1)
pre_ffn10, rec_ffn10, _ = precision_recall_curve(
    y_testffn[:, icol], ffn10.predict(x_testffn)[:, icol], pos_label=1)
pre_cnn3, rec_cnn3, _ = precision_recall_curve(
    y_testcnn[:, icol], model_cnn3.predict(x_testcnn)[:, icol], pos_label=1)
pre_cnn5, rec_cnn5, _ = precision_recall_curve(
    y_testcnn[:, icol],
    model_cnn5layers.predict(x_testcnn)[:, icol], pos_label=1)
pre_rf, rec_rf, _ = precision_recall_curve(
    y_test, rf.predict_proba(x_testrf)[:, icol], pos_label=1)
noskill = len(y_test[y_test == 1]) / len(y_test)

plt.figure(1)
plt.plot([0, 1], [noskill, noskill], 'k--')
plt.plot(
    rec_ffn3, pre_ffn3,
    label='FNN 3 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_ffn3, pre_ffn3), 3)))
plt.plot(
    rec_ffn, pre_ffn,
    label='FNN 5 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_ffn, pre_ffn), 3)))
plt.plot(
    rec_ffn10, pre_ffn10,
    label='FNN 10 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_ffn10, pre_ffn10), 3)))
plt.plot(
    rec_cnn3, pre_cnn3,
    label='CNN 3 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_cnn3, pre_cnn3), 3)))
plt.plot(
    rec_cnn5, pre_cnn5,
    label='CNN 5 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_cnn5, pre_cnn5), 3)))
plt.plot(
    rec_rf, pre_rf,
    label='RF (AUC: %s \u00B1 0.001)' % (np.round(auc(rec_rf, pre_rf), 3)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.show()

yscore = []
yscore.append(ffn3.predict(x_testffn))
yscore.append(ffn.predict(x_testffn))
yscore.append(ffn10.predict(x_testffn))
yscore.append(model_cnn3.predict(x_testcnn))
yscore.append(model_cnn5layers.predict(x_testcnn))

ypred = []
for n in range(len(yscore)):
    ypred.append(np.argmax(yscore[n], axis=1))

print('\nAccuracy Score')
print('FNN 3:', np.round(accuracy_score(y_test, ypred[0]), 3))
print('FNN 5:', np.round(accuracy_score(y_test, ypred[1]), 3))
print('FNN 10:', np.round(accuracy_score(y_test, ypred[2]), 3))
print('CNN 3:', np.round(accuracy_score(y_test, ypred[3]), 3))
print('CNN 5:', np.round(accuracy_score(y_test, ypred[4]), 3))
print('RF:', np.round(accuracy_score(y_test, ypred[5]), 3))

print('\nBrier Score')
print('FNN 3:', np.round(brierScore(y_test, yscore[0]), 3))
print('FNN 5:', np.round(brierScore(y_test, yscore[1]), 3))
print('FNN 10:', np.round(brierScore(y_test, yscore[2]), 3))
print('CNN 3:', np.round(brierScore(y_test, yscore[3]), 3))
print('CNN 5:', np.round(brierScore(y_test, yscore[4]), 3))
print('RF:', np.round(brierScore(y_test, yscore[5]), 3))

print('\nF1 Score')
print('FNN 3:', np.round(f1_score(y_test, ypred[0]), 3))
print('FNN 5:', np.round(f1_score(y_test, ypred[1]), 3))
print('FNN 10:', np.round(f1_score(y_test, ypred[2]), 3))
print('CNN 3:', np.round(f1_score(y_test, ypred[3]), 3))
print('CNN 5:', np.round(f1_score(y_test, ypred[4]), 3))
print('RF:', np.round(f1_score(y_test, ypred[5]), 3))

print('\nPrecision Score')
print('FNN 3:', np.round(
    precision_score(y_test, ypred[0], zero_division=0), 3))
print('FNN 5:', np.round(
    precision_score(y_test, ypred[1], zero_division=0), 3))
print('FNN 10:', np.round(
    precision_score(y_test, ypred[2], zero_division=0), 3))
print('CNN 3:', np.round(
    precision_score(y_test, ypred[3], zero_division=0), 3))
print('CNN 5:', np.round(
    precision_score(y_test, ypred[4], zero_division=0), 3))
print('RF:', np.round(
    precision_score(y_test, ypred[5], zero_division=0), 3))

print('\nRecall Score')
print('FNN 3:', np.round(recall_score(y_test, ypred[0]), 3))
print('FNN 5:', np.round(recall_score(y_test, ypred[1]), 3))
print('FNN 10:', np.round(recall_score(y_test, ypred[2]), 3))
print('CNN 3:', np.round(recall_score(y_test, ypred[3]), 3))
print('CNN 5:', np.round(recall_score(y_test, ypred[4]), 3))
print('RF:', np.round(recall_score(y_test, ypred[5]), 3))

# display the top results based of class 1 based on ascending proba of class 1
predictions = ffn10.predict(x_testffn)
classpredictions = np.argmax(predictions, axis=1)

class1predproba = []
class1predproba_row = []
k = 0
for n in classpredictions:
    if n == 1:
        class1predproba.append(predictions[k, n])
        class1predproba_row.append(k)
    k += 1

maxpredproba_indx = np.argsort(class1predproba)[::-1]
class1predproba_sort = np.asarray(class1predproba)[maxpredproba_indx]
print(y_test.iloc[np.asarray(class1predproba_row)[maxpredproba_indx]])

# we decode the encoded off-targets (for paper publication)
indx = np.asarray(class1predproba_row)[maxpredproba_indx]
nindx = 30

dc = ['A', 'G', 'C', 'T', 'A', 'G', 'C', 'T']
seq_sgRNA_DNA = np.chararray((2 * nindx, 23))
seq_sgRNA_DNA[:] = ''
indx_counter = 0
indx_seq = 0

for iline in range(nindx):
    arr = x_test[indx[iline]]
    if imgrows == 4:
        arr = arr.reshape((4, 23), order='F')
    else:
        arr = arr.reshape((8, 23), order='F')

    for n in range(arr.shape[1]):
        loc_bp = np.where(arr[:, n] == 254)[0]
        indx_seq = 0
        for indx_loc_bp in loc_bp:
            seq_sgRNA_DNA[indx_counter + indx_seq, n] = dc[indx_loc_bp]

            if len(loc_bp) == 254:
                seq_sgRNA_DNA[indx_counter + indx_seq + 1, n] = (
                    seq_sgRNA_DNA[indx_counter + indx_seq, n])

            indx_seq += 1

    indx_counter += 2

# we post process the encoded 8x23
for iline in range(0, nindx*2, 2):
    for n in range(23):
        if (seq_sgRNA_DNA[iline, n] == seq_sgRNA_DNA[iline+1, n]):
            seq_sgRNA_DNA[iline+1, n] = ''

seq_sgRNA_DNA[0].decode()
pd.DataFrame(seq_sgRNA_DNA.decode())

indxnames = y_test.keys()[np.asarray(class1predproba_row)[maxpredproba_indx]]
print(loaddata.target_names[np.asarray(indxnames)])


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#           PREDICTIONS ON GUIDE SEQ
#           RESULTS FOR PUBLICATION
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# we import the pkl file containing the data
if imgrows == 4:
    print('refer to experiments 4x23 py file')
else:
    loadguideseq = pkl.load(open('guideseq8x23.pkl', 'rb'), encoding='latin1')

# the data, split between train and test sets
x_guideseq = loadguideseq.images
y_guideseq = loadguideseq.target
y_guideseqffn = to_categorical(y_guideseq, num_classes)
y_guideseqcnn = to_categorical(y_guideseq, num_classes)
yguideseqdf = pd.DataFrame(loadguideseq.target, columns=['target'])
yguideseqdf['targetnames'] = loadguideseq.target_names

if K.image_data_format() == 'channels_first':
    x_guideseqffn = x_guideseq.reshape(x_guideseq.shape[0], img_rows*img_cols)
    input_shape = (img_rows*img_cols)

    x_guideseqcnn = x_guideseq.reshape(
        x_guideseq.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_guideseqffn = x_guideseq.reshape(
        x_guideseq.shape[0], img_rows*img_cols)
    input_shape = (img_rows*img_cols)

    x_guideseqcnn = x_guideseq.reshape(
        x_guideseq.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_guideseqffn = x_guideseqffn.astype('float32')
x_guideseqffn /= 255
x_guideseqcnn = x_guideseqcnn.astype('float32')
x_guideseqcnn /= 255
x_guideseqrf = x_guideseq.astype('float32').reshape(-1, imgrows*imgcols)
x_guideseqrf /= 255

# ROC CURVE
# -*-*-*-*-

icol = 1
saveplot = False

fpr_ffn3, tpr_ffn3, _ = roc_curve(
    y_guideseqffn[:, icol], ffn3.predict(x_guideseqffn)[:, icol])
fpr_ffn, tpr_ffn, _ = roc_curve(
    y_guideseqffn[:, icol], ffn.predict(x_guideseqffn)[:, icol])
fpr_ffn10, tpr_ffn10, _ = roc_curve(
    y_guideseqffn[:, icol], ffn10.predict(x_guideseqffn)[:, icol])
fpr_cnn3, tpr_cnn3, _ = roc_curve(
    y_guideseqcnn[:, icol], model_cnn3.predict(x_guideseqcnn)[:, icol])
fpr_cnn5layers, tpr_cnn5layers, _ = roc_curve(
    y_guideseqcnn[:, icol], model_cnn5layers.predict(x_guideseqcnn)[:, icol])
fpr_rf, tpr_rf, _ = roc_curve(
    y_guideseq, rf.predict_proba(x_guideseqrf)[:, icol])

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(
    fpr_ffn3, tpr_ffn3,
    label='FNN 3 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_ffn3, tpr_ffn3), 3)))
plt.plot(
    fpr_ffn, tpr_ffn,
    label='FNN 5 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_ffn, tpr_ffn), 3)))
plt.plot(
    fpr_ffn10, tpr_ffn10,
    label='FNN 10 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_ffn10, tpr_ffn10), 3)))
plt.plot(
    fpr_cnn3, tpr_cnn3,
    label='CNN 3 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_cnn3, tpr_cnn3), 3)))
plt.plot(
    fpr_cnn5layers, tpr_cnn5layers,
    label='CNN 5 Layers (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_cnn5layers, tpr_cnn5layers), 3)))
plt.plot(
    fpr_rf, tpr_rf,
    label='RF (AUC: %s \u00B1 0.001)' % (
        np.round(auc(fpr_rf, tpr_rf), 3)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best')
plt.savefig("roccurveguideseq8x23.pdf")

# PRECISION RECALL CURVE
# -*-*-*-*-*-*-*-*-*-*-*

icol = 1

pre_ffn3, rec_ffn3, _ = precision_recall_curve(
    y_guideseqffn[:, icol], ffn3.predict(x_guideseqffn)[:, icol], pos_label=1)
pre_ffn, rec_ffn, _ = precision_recall_curve(
    y_guideseqffn[:, icol], ffn.predict(x_guideseqffn)[:, icol], pos_label=1)
pre_ffn10, rec_ffn10, _ = precision_recall_curve(
    y_guideseqffn[:, icol], ffn10.predict(x_guideseqffn)[:, icol], pos_label=1)
pre_cnn3, rec_cnn3, _ = precision_recall_curve(
    y_guideseqcnn[:, icol],
    model_cnn3.predict(x_guideseqcnn)[:, icol], pos_label=1)
pre_cnn5, rec_cnn5, _ = precision_recall_curve(
    y_guideseqcnn[:, icol],
    model_cnn5layers.predict(x_guideseqcnn)[:, icol], pos_label=1)
pre_rf, rec_rf, _ = precision_recall_curve(
    y_guideseq, rf.predict_proba(x_guideseqrf)[:, icol], pos_label=1)
noskill = len(y_guideseq[y_guideseq == 1]) / len(y_guideseq)

plt.figure(1)
plt.plot([0, 1], [noskill, noskill], 'k--')
plt.plot(
    rec_ffn3, pre_ffn3,
    label='FNN 3 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_ffn3, pre_ffn3), 3)))
plt.plot(
    rec_ffn, pre_ffn,
    label='FNN 5 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_ffn, pre_ffn), 3)))
plt.plot(
    rec_ffn10, pre_ffn10,
    label='FNN 10 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_ffn10, pre_ffn10), 3)))
plt.plot(
    rec_cnn3, pre_cnn3,
    label='CNN 3 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_cnn3, pre_cnn3), 3)))
plt.plot(
    rec_cnn5, pre_cnn5,
    label='CNN 5 Layers(AUC: %s \u00B1 0.001)' % (
        np.round(auc(rec_cnn5, pre_cnn5), 3)))
plt.plot(
    rec_rf, pre_rf,
    label='RF (AUC: %s \u00B1 0.001)' % (np.round(auc(rec_rf, pre_rf), 3)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')

if saveplot is True:
    plt.savefig("precisionrecallcurve.pdf")
else:
    plt.show()

yscore = []
yscore.append(ffn3.predict(x_guideseqffn))
yscore.append(ffn.predict(x_guideseqffn))
yscore.append(ffn10.predict(x_guideseqffn))
yscore.append(model_cnn3.predict(x_guideseqcnn))
yscore.append(model_cnn5layers.predict(x_guideseqcnn))

ypred = []
for n in range(len(yscore)):
    ypred.append(np.argmax(yscore[n], axis=1))

print('\nAccuracy Score')
print('FNN 3:', np.round(accuracy_score(y_guideseq, ypred[0]), 3))
print('FNN 5:', np.round(accuracy_score(y_guideseq, ypred[1]), 3))
print('FNN 10:', np.round(accuracy_score(y_guideseq, ypred[2]), 3))
print('CNN 3:', np.round(accuracy_score(y_guideseq, ypred[3]), 3))
print('CNN 5:', np.round(accuracy_score(y_guideseq, ypred[4]), 3))
print('RF:', np.round(accuracy_score(y_guideseq, ypred[5]), 3))

print('\nBrier Score')
print('FNN 3:', np.round(brierScore(y_guideseq, yscore[0]), 3))
print('FNN 5:', np.round(brierScore(y_guideseq, yscore[1]), 3))
print('FNN 10:', np.round(brierScore(y_guideseq, yscore[2]), 3))
print('CNN 3:', np.round(brierScore(y_guideseq, yscore[3]), 3))
print('CNN 5:', np.round(brierScore(y_guideseq, yscore[4]), 3))
print('RF:', np.round(brierScore(y_guideseq, yscore[5]), 3))

print('\nF1 Score')
print('FNN 3:', np.round(f1_score(y_guideseq, ypred[0]), 3))
print('FNN 5:', np.round(f1_score(y_guideseq, ypred[1]), 3))
print('FNN 10:', np.round(f1_score(y_guideseq, ypred[2]), 3))
print('CNN 3:', np.round(f1_score(y_guideseq, ypred[3]), 3))
print('CNN 5:', np.round(f1_score(y_guideseq, ypred[4]), 3))
print('RF:', np.round(f1_score(y_guideseq, ypred[5]), 3))

print('\nPrecision Score')
print('FNN 3:', np.round(
    precision_score(y_guideseq, ypred[0], zero_division=0), 3))
print('FNN 5:', np.round(
    precision_score(y_guideseq, ypred[1], zero_division=0), 3))
print('FNN 10:', np.round(
    precision_score(y_guideseq, ypred[2], zero_division=0), 3))
print('CNN 3:', np.round(
    precision_score(y_guideseq, ypred[3], zero_division=0), 3))
print('CNN 5:', np.round(
    precision_score(y_guideseq, ypred[4], zero_division=0), 3))
print('RF:', np.round(
    precision_score(y_guideseq, ypred[5], zero_division=0), 3))

print('\nRecall Score')
print('FNN 3:', np.round(recall_score(y_guideseq, ypred[0]), 3))
print('FNN 5:', np.round(recall_score(y_guideseq, ypred[1]), 3))
print('FNN 10:', np.round(recall_score(y_guideseq, ypred[2]), 3))
print('CNN 3:', np.round(recall_score(y_guideseq, ypred[3]), 3))
print('CNN 5:', np.round(recall_score(y_guideseq, ypred[4]), 3))
print('RF:', np.round(recall_score(y_guideseq, ypred[5]), 3))

# CONFUSION MATRIX
# -*-*-*-*-*-*-*-*
modellist = [
    ffn3, ffn,
    model_cnn3, model_cnn5layers,
    model_cnn5layers]
xlist = [
    x_guideseqffn, x_guideseqffn,
    x_guideseqcnn, x_guideseqcnn,
    x_guideseqcnn]
ylist = [
    y_guideseqffn, y_guideseqffn,
    y_guideseqcnn, y_guideseqcnn,
    y_guideseqcnn]

for n in range(len(modellist)):
    predictions = modellist[n].predict(xlist[n])
    classpredictions = np.argmax(predictions, axis=1)
    print('\nConf Matrix of ', modellist[n])
    dispConfMatrixAsArray(ylist[n][:, 1], classpredictions, disp=True)

# display the top results based of class 1 based on ascending proba of class 1
predictions = ffn3.predict(x_guideseqffn)
classpredictions = np.argmax(predictions, axis=1)

class1predproba = []
class1predproba_row = []
k = 0
for n in classpredictions:
    if n == 1:
        class1predproba.append(predictions[k, n])
        class1predproba_row.append(k)
    k += 1

maxpredproba_indx = np.argsort(class1predproba)[::-1]
class1predproba_sort = np.asarray(class1predproba)[maxpredproba_indx]
print(yguideseqdf.iloc[np.asarray(
    class1predproba_row)[maxpredproba_indx]].head(30))

# we decode the encoded off-targets (for paper publication)
indx = np.asarray(class1predproba_row)[maxpredproba_indx]
nindx = len(indx)

dc = ['A', 'G', 'C', 'T', 'A', 'G', 'C', 'T']
seq_sgRNA_DNA = np.chararray((2 * nindx, 23))
seq_sgRNA_DNA[:] = ''
indx_counter = 0
indx_seq = 0

for iline in range(nindx):
    arr = x_test[indx[iline]]
    if imgrows == 4:
        arr = arr.reshape((4, 23), order='F')
    else:
        arr = arr.reshape((8, 23), order='F')

    for n in range(arr.shape[1]):
        loc_bp = np.where(arr[:, n] == 254)[0]
        indx_seq = 0
        for indx_loc_bp in loc_bp:
            seq_sgRNA_DNA[indx_counter + indx_seq, n] = dc[indx_loc_bp]

            if len(loc_bp) == 254:
                seq_sgRNA_DNA[indx_counter + indx_seq + 1, n] = (
                    seq_sgRNA_DNA[indx_counter + indx_seq, n])

            indx_seq += 1

    indx_counter += 2

# we post process the encoded 8x23
for iline in range(0, nindx*2, 2):
    for n in range(23):
        if (seq_sgRNA_DNA[iline, n] == seq_sgRNA_DNA[iline+1, n]):
            seq_sgRNA_DNA[iline+1, n] = ''

seq_sgRNA_DNA[0].decode()
pd.DataFrame(seq_sgRNA_DNA.decode())

# display the top results based of class 1 based on ascending proba of class 1
class1predproba = []
class1predproba_row = []
k = 0
for n in classpredictions:
    if n == 1:
        class1predproba.append(predictions[k, n])
        class1predproba_row.append(k)
    k += 1

maxpredproba_indx = np.argsort(class1predproba)[::-1]
class1predproba_sort = np.asarray(class1predproba)[maxpredproba_indx]
print(yguideseqdf.iloc[np.asarray(class1predproba_row)[maxpredproba_indx]])
