'''
===============================
A Test model about Linear model
===============================

SVM with linear kernel VS Logistic Regression

'''

print(__doc__)

#########################################################
# author   : Quentin_Hsu                                #
# email    : jlove.dragon@gmail.com                     #
# date     : 2014.09.17                                 #
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm

def createDataSet():
	# create 40 separable points
	np.random.seed(0)
	X = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
	Y = [0] * 50 + [1] * 50
	return X, Y

def getClf(model):
	if model == "Perceptron":
		clf = linear_model.Perceptron()
	else:
		clf  = svm.SVC(kernel='linear', C=1)
	return clf


def fitModel(clff, X, Y):
	clff.fit(X, Y)
	w = clff.coef_[0]
	a = -w[0] / w[1]
	xx = np.linspace(-5, 5)
	yy = a * xx - (clff.intercept_[0]) / w[1]
	return xx, yy, a

def test():
	X, Y = createDataSet()

	# get tge Perceptron classifier
	perc_clf = getClf("Perceptron")
	perc_xx, perc_yy, perc_a = fitModel(perc_clf, X, Y)

	# get the svm linear classifier
	svm_clf = getClf("svm")
	svm_xx, svm_yy, svm_a = fitModel(svm_clf, X, Y)

	# get tge magin
	b = svm_clf.support_vectors_[0]
	yy_down = svm_a * svm_xx + (b[1] - svm_a * b[0])
	b = svm_clf.support_vectors_[-1]
	yy_up = svm_a * svm_xx + (b[1] - svm_a * b[0])

	# plot the support vectors
	plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1],
	            s=80, facecolors='none')

	# plot the data node
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, label='data')
	plt.hold('on')

	# plot the decision boundary
	plt.plot(perc_xx, perc_yy, 'b-',  label='perceptron')

	# plot the magin
	plt.plot(svm_xx, svm_yy, 'r-', label='SVM')
	plt.plot(svm_xx, yy_down, 'r--', label='SVM Margin')
	plt.plot(svm_xx, yy_up, 'r--', label='SVM Margin')

	plt.title('SVM VS Perceptron')

	# a very useful tools: show label
	plt.legend()

	plt.axis('tight')
	plt.show()

# python plot_linear_model.py
if __name__ == '__main__':
	test()

# import plot_linear_model or reload(plot_linear_model)
if __name__ == 'plot_linear_model':
	print 'Testing......'
	test()
