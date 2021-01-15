"""Machine Learning Tree-Based Methods.
===============================================
Version |Date |   Author|   Comment
-----------------------------------------------
0.0 | 31 Oct 2020 | J. Charlier | initial version
0.1 | 13 Dec 2020 | J. Charlier | update for NB & LR
===============================================
"""
#
#
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
#
#
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols):
    xtrain = xtrain.astype('float32').reshape(-1, imgrows*imgcols)
    xtest = xtest.astype('float32').reshape(-1, imgrows*imgcols)
    xtrain /= 255
    xtest /= 255
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    return xtrain, xtest, ytrain, ytest
#
#
def initFitRF(xtrain, ytrain):
    rf = RandomForestClassifier(
        bootstrap=True, ccp_alpha=0.0, class_weight=None,
        criterion='gini', max_depth=None, max_features='auto',
        max_leaf_nodes=None, max_samples=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=1000,
        n_jobs=None, oob_score=False, random_state=42,
        verbose=0, warm_start=False
        )
    rf.fit(xtrain, ytrain)
    print("RF Training: Done")
    return rf
#
#
def initFitNaiveBayes(xtrain, ytrain):
    nb = ComplementNB(
        alpha=1.0,
        class_prior=None,
        fit_prior=True,
        norm=False
    )
    nb.fit(xtrain, ytrain)
    print("Naive Bayes Training: Done")
    return nb
#
#
def initFitLogReg(xtrain, ytrain):
    lr = LogisticRegression(
        C=0.01,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=42,
        solver='lbfgs',
        tol=0.0001,
        verbose=0,
        warm_start=False
    )
    lr.fit(xtrain, ytrain)
    print("Logistic Regression Training: Done")
    return lr
#
# Last card of module mltrees.
#
