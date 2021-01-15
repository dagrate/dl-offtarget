# -*- coding: utf-8 -*-
"""Perform the Experiments for Off-Target Predictions.
===============================================
Version |    Date     |   Author    |   Comment
-----------------------------------------------
0.0     | 25 Oct 2020 | J. Charlier | initial version
0.1     | 09 Nov 2020 | J. Charlier | update new version
0.2     | 11 Nov 2020 | J. Charlier | bug fix for 8x23 encoding
0.3     | 12 Nov 2020 | J. Charlier | bug fix for 8x23 encoding
0.4     | 13 Nov 2020 | J. Charlier | bug fix for 8x23 encoding
0.5     | 13 Dec 2020 | J. Charlier | update for NB & LR
0.6     | 26 Dec 2020 | J. Charlier | update for LSTM & GRU
0.7     | 27 Dec 2020 | J. Charlier | update for LSTM & GRU
0.8     | 28 Dec 2020 | J. Charlier | update for LSTM & GRU
===============================================
"""
#
#
from __future__ import print_function
import os
import time
import random
random.seed(42)
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score,
                             roc_curve, precision_score, recall_score,
                             auc, average_precision_score, 
                             precision_recall_curve, accuracy_score)
from sklearn.preprocessing import normalize
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
from tensorflow.python.keras.preprocessing.image import (ImageDataGenerator,
                                       img_to_array, 
                                       array_to_img)
# import tensorflow.python.keras as tfkeras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
#
import utilities, ffns, cnns, mltrees, rnns
#
#
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
#
#
# data read
# -*-*-*-*-
imgrows = 4
nexp = 3
imgcols = 23
num_classes = 2
epochs = 500
batch_size = 64
ismodelsaved = True   
undersampling = False
flpath = '/data/'
#
print('\n\n')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!       PREDICTIONS ON CRISPOR             !!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#
print('\n!!! DATA PIPELINE !!!\n')
loaddata = utilities.importData(
    flpath=flpath,
    encoding=str(imgrows) + 'x' + str(imgcols),
    sim='crispor',
    tl=False)
x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images,
    pd.Series(loaddata.target), #loaddata.target,
    test_size=0.3,
    shuffle=True, 
    random_state=42)
xtraincnn, xtestcnn, ytraincnn, ytestcnn, inputshapecnn = cnns.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols,
    num_classes)
xtrainffn, xtestffn, ytrainfnn, ytestffn, inputshapeffn = ffns.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols,
    num_classes)
xtrainrf, xtestrf, ytrainrf, ytestrf = mltrees.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols)
#
print('\n!!! TRAINING PIPELINE !!!\n')
print('\n!!! train fnns !!!\n')
ffn3 = ffns.ffnthree(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
ffn5 = ffns.ffnfive(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
ffn10 = ffns.ffnten(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
#
print('\n!!! train rnns !!!\n')
lstmrnn = rnns.lstmmdl(
    x_train,
    ytrainfnn,
    x_test,
    ytestffn,
    num_classes,
    batch_size,
    epochs,
    callbacks,
    imgrows,
    ismodelsaved=ismodelsaved)
grurnn = rnns.grumdl(
    x_train,
    ytrainfnn,
    x_test,
    ytestffn,
    num_classes,
    batch_size,
    epochs,
    callbacks,
    imgrows,
    ismodelsaved=ismodelsaved)
#
print('\n!!! train cnns !!!\n')
cnn3 = cnns.cnnthree(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
cnn5 = cnns.cnnfive(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
cnn10 = cnns.cnnten(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
cnnlin = cnns.cnnlin(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved) 
#
print('\n!!! train random forest !!!\n')
rf = mltrees.initFitRF(xtrainrf, ytrainrf)
nb = mltrees.initFitNaiveBayes(xtrainrf, ytrainrf)
lr = mltrees.initFitLogReg(xtrainrf, ytrainrf)
#
print('\n!!! TESTING PIPELINE !!!\n')
print('\n!!!! roc curve on crispor data !!!\n')
mdls = [
    ffn3, ffn5, ffn10,
    cnn3, cnn5, 
    #cnn10, cnnlin,
    lstmrnn, grurnn,
    rf, nb, lr]
mdllbls = [
    'FNN3', 'FNN5', 'FNN10',
    'CNN3', 'CNN5',
    #'CNN10', 'CNN Lin',
    'LSTM', 'GRU',
    'RF', 'NB', 'LR']
mdlxte = [
    xtestffn, xtestffn, xtestffn,
    xtestcnn, xtestcnn,
    #xtestcnn, xtestcnn,
    x_test, x_test,
    xtestrf, xtestrf, xtestrf]
mdlyte = [
    ytestffn, ytestffn, ytestffn,
    ytestcnn, ytestcnn,
    #ytestcnn, ytestcnn,
    ytestffn, ytestffn,
    y_test, y_test, y_test]
utilities.plotRocCurve(
    mdls, mdllbls,
    mdlxte, mdlyte,
    'roccurvecrispr4x23.pdf')
print('\n!!!! precision recall curve on crispor data !!!\n')
utilities.plotPrecisionRecallCurve(
    mdls, mdllbls,
    mdlxte, mdlyte,
    'precisionrecallcurvecrispr4x23.pdf')
#
preds = utilities.collectPreds(mdls, mdlxte)
# correct predictions of Lin et al.
# preds.yscore[-4][:, 1] = np.abs(preds.yscore[-4][:, 1])
# for n in range(len(preds.yscore[-4])):
#     under = preds.yscore[-4][n, 0] + preds.yscore[-4][n, 1]
#     preds.yscore[-4][n, 0] = preds.yscore[-4][n, 0] / (under)
#     preds.yscore[-4][n, 1] = preds.yscore[-4][n, 1] / (under)
#
objfuns = [
    utilities.brierScore, accuracy_score,
    f1_score, precision_score, recall_score]
for objfun in objfuns:
    if 'brier' in str(objfun):
        utilities.computeScore(
            objfun,
            y_test,
            preds.yscore,
            mdllbls)
    else:
        utilities.computeScore(
            objfun,
            y_test,
            preds.ypred,
            mdllbls)
#
utilities.printTopPreds(
    cnn3,
    xtestcnn,
    y_test,
    loaddata.target_names,
    imgrows)
#
#
print('\n\n')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!       PREDICTIONS ON GUIDE SEQ           !!!')
print('!!!       RESULTS FOR PUBLICATION            !!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#
print('\n!!! DATA PIPELINE !!!\n')
loadguideseq = utilities.importData(
    flpath=flpath,
    encoding=str(imgrows)+'x'+str(imgcols),
    sim='guideseq',
    tl=False)
gseq = utilities.transformGuideSeqImages(
		loadguideseq,
		num_classes,
		imgrows, imgcols)
#
print('\n!!! TRAINING PIPELINE !!!\n')
print('transfer learning: no training required.')
#
print('\n!!! TESTING PIPELINE !!!\n')
print('\n!!!! roc curve on guideseq data !!!\n')
mdlxte = [
    gseq.xgseqffn, gseq.xgseqffn, gseq.xgseqffn,
    gseq.xgseqcnn, gseq.xgseqcnn,
    #gseq.xgseqcnn, gseq.xgseqcnn,
    loadguideseq.images, loadguideseq.images,
    gseq.xgseqrf, gseq.xgseqrf, gseq.xgseqrf]
mdlyte = [
    gseq.ygseqffn, gseq.ygseqffn, gseq.ygseqffn,
    gseq.ygseqcnn, gseq.ygseqcnn,
    #gseq.ygseqcnn, gseq.ygseqcnn,
    gseq.ygseqffn, gseq.ygseqffn,
    gseq.ygseqrf, gseq.ygseqrf, gseq.ygseqrf]
utilities.plotRocCurve(
    mdls, mdllbls,
    mdlxte, mdlyte,
    'roccurveguideseq4x23.pdf')
print('\n!!!! precision recall curve on guideseq data !!!\n')
utilities.plotPrecisionRecallCurve(
    mdls, mdllbls,
    mdlxte, mdlyte,
    'precisionrecallcurveguideseq4x23.pdf')
predsgseq = utilities.collectPreds(mdls, mdlxte)
#
# correct predictions of Linn et al.
# predsgseq.yscore[-4][:, 1] = np.abs(predsgseq.yscore[-4][:, 1])
# for n in range(len(predsgseq.yscore[-4])):
#     under = predsgseq.yscore[-4][n, 0] + predsgseq.yscore[-4][n, 1]
#     predsgseq.yscore[-4][n, 0] = predsgseq.yscore[-4][n, 0] / (under)
#     predsgseq.yscore[-4][n, 1] = predsgseq.yscore[-4][n, 1] / (under)
#
for objfun in objfuns:
    if 'brier' in str(objfun):
        utilities.computeScore(
            objfun,
            loadguideseq.target,
            predsgseq.yscore,
            mdllbls)
    else:
        utilities.computeScore(
            objfun,
            loadguideseq.target,
            predsgseq.ypred,
            mdllbls)
#
utilities.printTopPreds(
    cnn3,
    xtestcnn,
    gseq.ygseqdf,
    loadguideseq.target_names,
    imgrows)
#
# End of experiments with 4 x 23 encoding.
#
"""Perform the Experiments for Off-Target Predictions with 8x23 encoding.
===============================================
Version |    Date     |   Author    |   Comment
-----------------------------------------------
0.0     | 25 Oct 2020 | J. Charlier | initial version
0.1     | 09 Oct 2020 | J. Charlier | update new version
0.2     | 11 Oct 2020 | J. Charlier | bug fix for 8x23 encoding
0.3     | 12 Oct 2020 | J. Charlier | bug fix for 8x23 encoding
0.4     | 13 Dec 2020 | J. Charlier | update for NB & LR
0.5     | 28 Dec 2020 | J. Charlier | update for LSTM & GRU
===============================================
"""
#
#
from __future__ import print_function
import os
import time
import random
random.seed(42)
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score,
                             roc_curve, precision_score, recall_score,
                             auc, average_precision_score, 
                             precision_recall_curve, accuracy_score)
from sklearn.preprocessing import normalize
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
from tensorflow.python.keras.preprocessing.image import (ImageDataGenerator,
                                       img_to_array, 
                                       array_to_img)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
#
import utilities, ffns, cnns, mltrees, rnns
#
#
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
#
# global variables
imgrows = 8
nexp = 3
imgcols = 23
num_classes = 2
epochs = 500
batch_size = 64
ismodelsaved = True
undersampling = False
flpath = '/data/'
#
print('\n\n')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!       PREDICTIONS ON CRISPOR             !!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#
print('\n!!! DATA PIPELINE !!!\n')
loaddata = utilities.importData(
    flpath=flpath,
    encoding=str(imgrows) + 'x' + str(imgcols),
    sim='crispor',
    tl=False)
x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images,
    pd.Series(loaddata.target), #loaddata.target,
    test_size=0.3,
    shuffle=True, 
    random_state=42)
xtrainffn, xtestffn, ytrainfnn, ytestffn, inputshapeffn = ffns.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols,
    num_classes)
xtraincnn, xtestcnn, ytraincnn, ytestcnn, inputshapecnn = cnns.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols,
    num_classes)
xtrainrf, xtestrf, ytrainrf, ytestrf = mltrees.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols)
#
print('\n!!! TRAINING PIPELINE !!!\n')
print('\n!!! train fnns !!!\n')
ffn3 = ffns.ffnthree(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
ffn5 = ffns.ffnfive(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
ffn10 = ffns.ffnten(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
#
print('\n!!! train rnns !!!\n')
lstmrnn = rnns.lstmmdl(
    x_train,
    ytrainfnn,
    x_test,
    ytestffn,
    num_classes,
    batch_size,
    epochs,
    callbacks,
    imgrows,
    ismodelsaved=ismodelsaved)
grurnn = rnns.grumdl(
    x_train,
    ytrainfnn,
    x_test,
    ytestffn,
    num_classes,
    batch_size,
    epochs,
    callbacks,
    imgrows,
    ismodelsaved=ismodelsaved)
#
print('\n!!! train cnns !!!\n')
cnn3 = cnns.cnnthree(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
cnn5 = cnns.cnnfive(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
cnn10 = cnns.cnnten(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved)
cnnlin = cnns.cnnlin(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved) 
#
print('\n!!! train random forest !!!\n')
rf = mltrees.initFitRF(xtrainrf, ytrainrf)
print('\n!!! train naive bayes !!!\n')
nb = mltrees.initFitNaiveBayes(xtrainrf, ytrainrf)
print('\n!!! train logistic regression !!!\n')
lr = mltrees.initFitLogReg(xtrainrf, ytrainrf)
#
print('\n!!! TESTING PIPELINE !!!\n')
print('\n!!!! roc curve on crispor data !!!\n')
mdls = [
    ffn3, ffn5, ffn10,
    cnn3, cnn5, 
    #cnn10, cnnlin,
    lstmrnn, grurnn,
    rf, nb, lr]
mdllbls = [
    'FNN3', 'FNN5', 'FNN10',
    'CNN3', 'CNN5',
    #'CNN10', 'CNN Lin',
    'LSTM', 'GRU',
    'RF', 'NB', 'LR']
mdlxte = [
    xtestffn, xtestffn, xtestffn,
    xtestcnn, xtestcnn,
    #xtestcnn, xtestcnn,
    x_test, x_test,
    xtestrf, xtestrf, xtestrf]
mdlyte = [
    ytestffn, ytestffn, ytestffn,
    ytestcnn, ytestcnn,
    #ytestcnn, ytestcnn,
    ytestffn, ytestffn,
    y_test, y_test, y_test]
utilities.plotRocCurve(
    mdls, mdllbls,
    mdlxte, mdlyte,
    'roccurvecrispr8x23.pdf')
print('\n!!!! precision recall curve on crispor data !!!\n')
utilities.plotPrecisionRecallCurve(
    mdls, mdllbls,
    mdlxte, mdlyte,
    'precisionrecallcurvecrispr8x23.pdf')
#
#
preds = utilities.collectPreds(mdls, mdlxte)
# correct predictions of Linn et al.
# preds.yscore[-4][:, 1] = np.abs(preds.yscore[-4][:, 1])
# for n in range(len(preds.yscore[-4])):
#     under = preds.yscore[-4][n, 0] + preds.yscore[-4][n, 1]
#     preds.yscore[-4][n, 0] = preds.yscore[-4][n, 0] / (under)
#     preds.yscore[-4][n, 1] = preds.yscore[-4][n, 1] / (under)
#
objfuns = [
    utilities.brierScore, accuracy_score,
    f1_score, precision_score, recall_score]
for objfun in objfuns:
    if 'brier' in str(objfun):
        utilities.computeScore(
            objfun,
            y_test,
            preds.yscore,
            mdllbls)
    else:
        utilities.computeScore(
            objfun,
            y_test,
            preds.ypred,
            mdllbls)
#
utilities.printTopPreds(
    cnn3,
    xtestcnn,
    y_test,
    loaddata.target_names,
    imgrows)
#
print('\n\n')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!       PREDICTIONS ON GUIDE SEQ           !!!')
print('!!!       RESULTS FOR PUBLICATION            !!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('\n!!! DATA PIPELINE !!!\n')
print('\n!!! reload dl for transfer learning !!!\n')
loaddata = utilities.importData(
    flpath=flpath,
    encoding=str(imgrows) + 'x' + str(imgcols),
    sim='crispor',
    tl=True)
x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images,
    pd.Series(loaddata.target), #loaddata.target,
    test_size=0.3,
    shuffle=True, 
    random_state=42)
xtrainrf, xtestrf, ytrainrf, ytestrf = mltrees.transformImages(
    x_train, x_test,
    y_train, y_test,
    imgrows, imgcols)
#
ffn3 = ffns.ffnthree(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
ffn5 = ffns.ffnfive(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
ffn10 = ffns.ffnten(
    xtrainffn, ytrainfnn,
    xtestffn, ytestffn,
    inputshapeffn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
lstmrnn = rnns.lstmmdl(
    x_train,
    ytrainfnn,
    x_test,
    ytestffn,
    num_classes,
    batch_size,
    epochs,
    callbacks,
    imgrows,
    ismodelsaved=ismodelsaved,
    tl=True)
grurnn = rnns.grumdl(
    x_train,
    ytrainfnn,
    x_test,
    ytestffn,
    num_classes,
    batch_size,
    epochs,
    callbacks,
    imgrows,
    ismodelsaved=ismodelsaved,
    tl=True)
cnn3 = cnns.cnnthree(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
cnn5 = cnns.cnnfive(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
cnn10 = cnns.cnnten(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
cnnlin = cnns.cnnlin(
    xtraincnn, ytraincnn,
    xtestcnn, ytestcnn,
    inputshapecnn, num_classes,
    batch_size, epochs,
    callbacks,
    ismodelsaved=ismodelsaved,
    tl=True)
rf = mltrees.initFitRF(xtrainrf, ytrainrf)
nb = mltrees.initFitNaiveBayes(xtrainrf, ytrainrf)
lr = mltrees.initFitLogReg(xtrainrf, ytrainrf)
#
print('\n!!! import guideseq data !!!\n')
loadguideseq = utilities.importData(
    flpath=flpath,
    encoding=str(imgrows) + 'x' + str(imgcols),
    sim='guideseq',
    tl=True)
gseq = utilities.transformGuideSeqImages(
		loadguideseq,
		num_classes,
		imgrows, imgcols)
print('\n!!!! roc curve on guideseq data !!!\n')
mdls = [
    ffn3, ffn5, ffn10,
    cnn3, cnn5,
    #cnn10, cnnlin,
    lstmrnn, grurnn,
    rf, nb, lr]
mdlxte = [
    gseq.xgseqffn, gseq.xgseqffn, gseq.xgseqffn,
    gseq.xgseqcnn, gseq.xgseqcnn,
    #gseq.xgseqcnn, gseq.xgseqcnn,
    loadguideseq.images, loadguideseq.images,
    gseq.xgseqrf, gseq.xgseqrf, gseq.xgseqrf]
mdlyte = [
    gseq.ygseqffn, gseq.ygseqffn, gseq.ygseqffn,
    gseq.ygseqcnn, gseq.ygseqcnn,
    #gseq.ygseqcnn, gseq.ygseqcnn,
    gseq.ygseqffn, gseq.ygseqffn,
    gseq.ygseqrf, gseq.ygseqrf, gseq.ygseqrf]
utilities.plotRocCurve(
    mdls, mdllbls, mdlxte, mdlyte,
    'roccurveguideseq8x23.pdf')
print('\n!!!! precision recall curve on guideseq data !!!\n')
utilities.plotPrecisionRecallCurve(
    mdls, mdllbls, mdlxte, mdlyte,
    'precisionrecallcurveguideseq8x23.pdf')
predsgseq = utilities.collectPreds(mdls, mdlxte)
# correct predictions of Linn et al.
# predsgseq.yscore[-4][:, 1] = np.abs(predsgseq.yscore[-4][:, 1])
# for n in range(len(predsgseq.yscore[-4])):
#     under = predsgseq.yscore[-4][n, 0] + predsgseq.yscore[-4][n, 1]
#     predsgseq.yscore[-4][n, 0] = predsgseq.yscore[-4][n, 0] / (under)
#     predsgseq.yscore[-4][n, 1] = predsgseq.yscore[-4][n, 1] / (under)
#
for objfun in objfuns:
    if 'brier' in str(objfun):
        utilities.computeScore(
            objfun,
            loadguideseq.target,
            predsgseq.yscore,
            mdllbls)
    else:
        utilities.computeScore(
            objfun,
            loadguideseq.target,
            predsgseq.ypred,
            mdllbls)
#
utilities.printTopPreds(
    cnn3,
    xtestcnn,
    gseq.ygseqdf,
    loadguideseq.target_names,
    imgrows)
#
# Last card of module offtargetsexp.
#
