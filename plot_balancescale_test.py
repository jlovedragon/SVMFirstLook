'''
===================================
A Test model about Non-linear model
===================================

To explain the performance of the svm is sensitive to the different kernel

'''

print(__doc__)

#########################################################
# author   : Quentin_Hsu                                #
# email    : jlove.dragon@gmail.com                     #
# date     : 2014.09.15                                 #
#########################################################

import numpy as np
import operator
from os import listdir
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# parse data from file
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = np.zeros((numberOfLines, 4))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[1:5]
		# print listFromLine[0]
		if listFromLine[0] == 'L':
			listFromLine[0] = 0
		elif listFromLine[0] == 'R':
			listFromLine[0] = 1
		else:
			listFromLine[0] = 2
		classLabelVector.append(int(listFromLine[0]))
		index += 1

	return returnMat, classLabelVector

# generate dataset
def createDataSet():
	X, y = file2matrix("balancescale_lostb1.txt")

	n_samples = X.shape[0]

	X_train = []; y_train = []
	X_test  = []; y_test  = []

	dataIndex = range(n_samples)
	for i in range(n_samples - 100):
		randIndex = int(np.random.uniform(0, len(dataIndex)))
		# print randIndex
		X_train.append([int(X[randIndex][0]), int(X[randIndex][1]), int(X[randIndex][2]), int(X[randIndex][3])])
		y_train.append([y[randIndex]])
		del(dataIndex[randIndex])

	for i in dataIndex:
		X_test.append([int(X[i][0]), int(X[i][1]), int(X[i][2]), int(X[i][3])])
		y_test.append([y[i]])

	y_train = np.array(y_train)
	y_test = np.array(y_test)

	return X_train, y_train, X_test, y_test

# get classifier from training data
def getClassifiers(X_train, y_train):
	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	C = 1.0  # SVM regularization parameter
	svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train.ravel())
	lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train.ravel())
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train.ravel())
	poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train.ravel())
	return svc, lin_svc, rbf_svc, poly_svc

# predict data
def predictData(X_data, y_data, svc, lin_svc, rbf_svc, poly_svc):
	Z1_data = svc.predict(X_data)
	Z2_data = lin_svc.predict(X_data)
	Z3_data = rbf_svc.predict(X_data)
	Z4_data = poly_svc.predict(X_data)

	errorCount1 = 0.0
	errorCount2 = 0.0
	errorCount3 = 0.0
	errorCount4 = 0.0

	for i in range(len(X_data)):
		if Z1_data[i] != y_data[i]:
			errorCount1 += 1
		if Z2_data[i] != y_data[i]:
			errorCount2 += 1
		if Z3_data[i] != y_data[i]:
			errorCount3 += 1
		if Z4_data[i] != y_data[i]:
			errorCount4 += 1

	return errorCount1, errorCount2, errorCount3, errorCount4

def test():
	# sum the error of training data
	trainErrorCount1 = 0.0
	trainErrorCount2 = 0.0
	trainErrorCount3 = 0.0
	trainErrorCount4 = 0.0

	# loop 100 to get average
	for i in range(100):
		X_train, y_train, X_test, y_test = createDataSet()
		svc, lin_svc, rbf_svc, poly_svc = getClassifiers(X_train, y_train)
		errorCount1, errorCount2, errorCount3, errorCount4 = predictData(X_train, y_train, svc, lin_svc, rbf_svc, poly_svc)
		trainErrorCount1 += errorCount1
		trainErrorCount2 += errorCount2
		trainErrorCount3 += errorCount3
		trainErrorCount4 += errorCount4

	# sum the error of testing data
	testErrorCount1 = 0.0
	testErrorCount2 = 0.0
	testErrorCount3 = 0.0
	testErrorCount4 = 0.0

	# loop 100 to get average
	for i in range(100):
		X_train, y_train, X_test, y_test = createDataSet()
		svc, lin_svc, rbf_svc, poly_svc = getClassifiers(X_train, y_train)
		errorCount1, errorCount2, errorCount3, errorCount4 = predictData(X_test, y_test, svc, lin_svc, rbf_svc, poly_svc)
		testErrorCount1 += errorCount1
		testErrorCount2 += errorCount2
		testErrorCount3 += errorCount3
		testErrorCount4 += errorCount4


	print "+=============================+"
	print "|dataType\t|kernel\t\t|rate |"
	print "|-----------+-----------+-----|"
	print "|training\t|svc\t\t|%.2f%%|" % ((trainErrorCount1 / (len(X_train)) / 100) * 100)
	print "|training\t|lin_svc\t|%.2f%%|" % ((trainErrorCount2 / (len(X_train)) / 100) * 100)
	print "|training\t|rbf_svc\t|%.2f%%|" % ((trainErrorCount3 / (len(X_train)) / 100) * 100)
	print "|training\t|poly_svc\t|%.2f%%|" % ((trainErrorCount4 / (len(X_train)) / 100) * 100)

	print "|-----------+-----------+-----|"


	print "|testing\t|svc\t\t|%.2f%%|" % ((testErrorCount1 / len(X_test) / 100) * 100)
	print "|testing\t|lin_svc\t|%.2f%%|" % ((testErrorCount2 / len(X_test) / 100) * 100)
	print "|testing\t|rbf_svc\t|%.2f%%|" % ((testErrorCount3 / len(X_test) / 100) * 100)
	print "|testing\t|poly_svc\t|%.2f%%|" % ((testErrorCount4 / len(X_test) / 100) * 100)
	print "+=============================+"

# python plot_balancescale_test.py
if __name__ == '__main__':
	test()

# import plot_balancescale_test or reload(plot_balancescale_test)
if __name__ == 'plot_balancescale_test':
	print 'Testing......'
	test()
