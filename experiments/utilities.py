"""Utilities Functions for DL experiments.
===============================================
Version |Date | Author| Comment
-----------------------------------------------
0.0 | 31 Oct 2020 | J. Charlier | initial version
===============================================
"""
#
#
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from sklearn.metrics import (
	classification_report, roc_auc_score,
	confusion_matrix, f1_score,
	roc_curve, precision_score, recall_score,
	auc, average_precision_score, 
	precision_recall_curve, accuracy_score)
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import pickle as pkl
p = print
#
#
def importData(flpath='', encoding='8x23', sim='crispor', tl=False):
	if encoding == '4x23':
		if sim == 'crispor':
			data = pkl.load(
				open(flpath+'encoded4x23withoutTsai.pkl','rb'),
				encoding='latin1'
			)
		else:
			data = pkl.load(
				open(flpath+'guideseq4x23.pkl','rb'),
				encoding='latin1'
			)
	else:
		if sim == 'crispor':
			if tl is False:
				data = pkl.load(
					open(flpath+'encoded8x23withoutTsai.pkl','rb'),
					encoding='latin1'
				)
			else:
				data = pkl.load(
					open(flpath+'encoded8x23withouttsai2.pkl','rb'),
					encoding='latin1'
				)
		else:
			data = pkl.load(
				open(flpath+'guideseq8x23.pkl','rb'),
				encoding='latin1'
			)
	return data
#
#
def dispConfMatrixAsArray(y_test, ypred, disp=True):
	confData = {'y':y_test, 'ypred':ypred}
	confdf = pd.DataFrame(confData, columns=['y', 'ypred'])
	confmatrix = pd.crosstab(
		confdf.y, confdf.ypred, 
		rownames=['target'],
		colnames=['predicted']
	)
	if disp == True:
		p('\nConfusion Matrix')
		if len(np.unique(ypred)) >= 2:
			p("%-3s" % 'TN:', "%-5s" % confmatrix.loc[0,0],
			"|%-3s" % 'FP:', "%-5s" % confmatrix.loc[0,1])
			p("%-3s" % 'FN:', "%-5s" % confmatrix.loc[1,0],
			"|%-3s" % 'TP:', "%-5s" % confmatrix.loc[1,1])
		else:
			p("%-3s" % 'TN:', "%-5s" % confmatrix.loc[0,0])
			p("%-3s" % 'FN:', "%-5s" % confmatrix.loc[1,0])
	return confmatrix
#
#
def plotConfusionMatrix(y_test, ypred):
	confData = {'y':y_test, 'ypred':ypred}
	confdf = pd.DataFrame(confData, columns=['y', 'ypred'])
	confMatrix = pd.crosstab(
		confdf.y, confdf.ypred, 
		rownames=['Target'],
		colnames=['Predicted']
	)
	plt.figure(figsize=(12, 8))
	sns.heatmap(confMatrix, annot=True, fmt='d')
	plt.show()
#
#
def getClassificationMetrics(y_test, yscore, ypred):
	posLabel = np.unique(y_test)
	p("\nModel Metrics:")
	p("%-40s" % ("ROC AUC Score:"), "{:.3f}".format(roc_auc_score(
		y_test, yscore[:,1])))
	for n in posLabel:
		p("%-40s" % ("F1 Score Class " + str(n) + " :"), 
		"{:.3f}".format(f1_score(
			y_test, 
			ypred, 
			pos_label=n))
		)
		p("%-40s" % ("Recall Score Class " + str(n) + " :"), 
			"{:.3f}".format(recall_score(
				y_test, 
				ypred, 
				pos_label=n))
		)
		p("%-40s" % ("Avrg Precision Score Class "+str(n)+" :"), 
			"{:.3f}".format(average_precision_score(
				y_test,
				yscore[:,1],
				pos_label=n))
		)
#
#
def brierScore(y_test, yscore):
	"""
	compute the Brier score
	0 is the best achievable, 1 the worst
	"""
	bscore = (1/len(y_test))
	bscore *= np.sum(np.power(yscore[:,1] - y_test, 2))
	return bscore
#
#
def getFprTpr(estimator, xtest, ytest, disp=True):
    preds = estimator.predict(xtest)
    if len(preds.shape) == 2:
        ypreds = preds[:,1]
    else:
        ypreds = preds
    fprs, tprs, _ = roc_curve(ytest, ypreds)
    if disp:
        print('false positive rates:\n', fprs)
        print('true positive rates:\n', tprs)
    return fprs, tprs
#
#
def plotRocCurve(
		estimators, labels,
		xtests, ytests,
		flnm, icol=1):
	indx = 0
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	for estimator in estimators:
		if len(ytests[indx].shape) == 2:
			fprs, tprs, _ = roc_curve(
				ytests[indx][:,icol],
				estimator.predict(xtests[indx])[:,icol]
			)
		else:
			fprs, tprs, _ = roc_curve(
				ytests[indx],
				estimator.predict_proba(xtests[indx])[:,icol]
			)
		#
		plt.plot(
			fprs, tprs,
			label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
				np.round(auc(fprs, tprs), 3))
		)
		indx += 1
	#
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.legend(loc='best')
	plt.savefig(flnm)
#
#
def plotPrecisionRecallCurve(
		estimators, labels,
		xtests, ytests,
		flnm, icol=1):
	indx = 0
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	for estimator in estimators:
		if len(ytests[indx].shape) == 2:
			pre, rec, _ = precision_recall_curve(
				ytests[indx][:,icol],
				estimator.predict(xtests[indx])[:,icol],
				pos_label=icol)
		else:
			pre, rec, _ = precision_recall_curve(
				ytests[indx],
				estimator.predict_proba(xtests[indx])[:,icol],
				pos_label=icol)
		#
		plt.plot(
			rec, pre,
			label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
				np.round(auc(rec, pre), 3))
		)
		indx += 1
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(flnm)
#
#
def collectPreds(estimators, xtests):
    yscore = []
    ypred = []
    for m in range(len(estimators)):
        estimator = estimators[m]
        if ('RandomForest' in str(estimator)) or (
            	'NB' in str(estimator)) or (
            	'Logistic' in str(estimator)):
            yscore.append(estimator.predict_proba(xtests[m]))
            ypred.append(estimator.predict(xtests[m]))
        else:
            yscore.append(estimator.predict(xtests[m]))
            ypred.append(np.argmax(yscore[m], axis=1))
    return Bunch(yscore=yscore, ypred=ypred)
#
#
def computeScore(objfun, true, pred, estmtrlbl):
	p(str(objfun))
	k = 0
	for estimator in estmtrlbl:
		if 'precision_score' in str(objfun):
			p(estimator, np.round(objfun(true, pred[k], zero_division=0), 3))
		else:
			p(estimator, np.round(objfun(true, pred[k]), 3))
		k += 1
#
#
def transformGuideSeqImages(
		gseq,
		num_classes,
		imgrows, imgcols):
	x_guideseq = gseq.images
	y_guideseq = gseq.target
	y_guideseqffn = to_categorical(y_guideseq, num_classes)
	y_guideseqcnn = to_categorical(y_guideseq, num_classes)
	yguideseqdf = pd.DataFrame(gseq.target, columns=['target'])
	yguideseqdf['targetnames'] = gseq.target_names
	#
	if K.image_data_format() == 'channels_first':
		x_guideseqffn = x_guideseq.reshape(
			x_guideseq.shape[0],
			imgrows*imgcols
		)
		x_guideseqcnn = x_guideseq.reshape(
			x_guideseq.shape[0],
			1,
			imgrows,
			imgcols
		)
	else:
		x_guideseqffn = x_guideseq.reshape(
			x_guideseq.shape[0],
			imgrows*imgcols
		)
		x_guideseqcnn = x_guideseq.reshape(
			x_guideseq.shape[0],
			imgrows,
			imgcols,
			1
		)
	#
	x_guideseqffn = x_guideseqffn.astype('float32')
	x_guideseqffn /= 255
	x_guideseqcnn = x_guideseqcnn.astype('float32')
	x_guideseqcnn /= 255
	x_guideseqrf = x_guideseq.astype('float32').reshape(-1, imgrows*imgcols)
	x_guideseqrf /= 255
	return Bunch(
		xgseqffn=x_guideseqffn, ygseqffn=y_guideseqffn,
		xgseqcnn= x_guideseqcnn, ygseqcnn=y_guideseqcnn,
		xgseqrf=x_guideseqrf, ygseqrf=y_guideseq,
		ygseqdf=yguideseqdf
		)
#
#
def printTopPreds(
		estimator,
		xtest, ytest,
		targetnames,
		imgrows,
		ispaper=False):
	"""Print the top results based of class 1 based on ascending proba of class 1."""
	print('Warning: ispaper is False, no ouput of results for paper publication')
	predictions = estimator.predict(xtest)
	classpredictions = np.argmax(predictions, axis=1)
	#
	class1predproba = []
	class1predproba_row = []
	k = 0
	for n in classpredictions:
		if n == 1:
			class1predproba.append(predictions[k, n])
			class1predproba_row.append(k)
	k += 1
	#
	maxpredproba_indx = np.argsort(class1predproba)[::-1]
	class1predproba_sort = np.asarray(class1predproba)[maxpredproba_indx]
	print('\ntop predictions for off-targets:')
	print(ytest.iloc[np.asarray(class1predproba_row)[maxpredproba_indx]])
	#
	# we decode the encoded off-targets (for paper publication)
	if ispaper is True:
		indx = np.asarray(class1predproba_row)[maxpredproba_indx]
		nindx = len(indx)
		dc = ['A', 'G', 'C', 'T', 'A', 'G', 'C', 'T']
		seq_sgRNA_DNA = np.chararray((2*nindx,23))
		seq_sgRNA_DNA[:] = ''
		indx_counter = 0
		indx_seq = 0
		for iline in range(nindx):
			arr = xtest[indx[iline]]
			if imgrows == 4:
				arr = arr.reshape((4,23), order='F')
			else:
				arr = arr.reshape((8,23), order='F')
			for n in range(arr.shape[1]):
				loc_bp = np.where(arr[:,n] == 254)[0]
				indx_seq = 0
				for indx_loc_bp in loc_bp:
					seq_sgRNA_DNA[indx_counter + indx_seq, n] = dc[indx_loc_bp]
					if len(loc_bp) == 254:
						seq_sgRNA_DNA[indx_counter + indx_seq + 1, n] = seq_sgRNA_DNA[indx_counter + indx_seq, n]
					indx_seq += 1
			indx_counter += 2
		#
		# we post process the encoded 8x23
		for iline in range(0, nindx*2, 2):
			for n in range(23):
				if (seq_sgRNA_DNA[iline, n] == seq_sgRNA_DNA[iline+1, n]):
					seq_sgRNA_DNA[iline+1, n] = ''
		#
		indxnames = ytest.keys()[np.asarray(class1predproba_row)[maxpredproba_indx]]
		print('\ntarget names (if na, FPs):')
		print(targetnames[np.asarray(indxnames)])
#
# Last card of module utilities.
#
